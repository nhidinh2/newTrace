"""Ingestion pipeline: fetch -> normalize -> deduplicate -> persist -> audit.

Every run is recorded as an :class:`~newstrace.models.IngestionRun` including the
configuration, seed and git commit, so a run can be reproduced.  A failure part
way through never leaves the database inconsistent: each article is committed in
its own nested transaction and failures are counted, not raised.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from newstrace.config import Settings, get_settings, load_source_policy, load_topics
from newstrace.ingestion.base import IngestionResult, RawArticle
from newstrace.ingestion.deduplicate import DuplicateDetector, index_article
from newstrace.ingestion.normalize import NormalizedArticle, normalize_article
from newstrace.logging import get_logger
from newstrace.models import Article, EmbeddingRecord, IngestionRun
from newstrace.retrieval.fulltext import index_article_text
from newstrace.utils import git_commit, utcnow

logger = get_logger(__name__)


@dataclass
class IngestOptions:
    source: str = "fixtures"
    topic: str | None = None
    timespan: str = "24h"
    max_records: int = 250
    windows: int = 1
    refresh: bool = False
    index: bool = True


def _open_run(session: Session, options: IngestOptions, settings: Settings) -> IngestionRun:
    run = IngestionRun(
        source=options.source,
        topic=options.topic,
        status="running",
        started_at=utcnow(),
        git_commit=git_commit(),
        random_seed=settings.random_seed,
        config={
            "source": options.source,
            "topic": options.topic,
            "timespan": options.timespan,
            "max_records": options.max_records,
            "windows": options.windows,
            "cluster_threshold": settings.cluster_threshold,
            "cluster_window_hours": settings.cluster_window_hours,
            "embedding_model": settings.embedding_model,
        },
    )
    session.add(run)
    session.flush()
    return run


def collect_raw_articles(options: IngestOptions, settings: Settings) -> list[RawArticle]:
    """Fetch raw articles from the configured source adapter."""
    topics = load_topics()
    selected = (
        {options.topic: topics[options.topic]}
        if options.topic and options.topic in topics
        else topics
    )

    if options.source == "fixtures":
        from newstrace.ingestion.fixtures import FixtureAdapter

        with FixtureAdapter() as adapter:
            return list(adapter.fetch(topic=options.topic))

    if options.source == "gdelt":
        from newstrace.ingestion.gdelt import GdeltAdapter

        with GdeltAdapter() as adapter:
            return list(
                adapter.fetch_topics(
                    selected,
                    timespan=options.timespan,
                    max_records=options.max_records,
                    refresh=options.refresh,
                    windows=options.windows,
                )
            )

    if options.source == "rss":
        from newstrace.ingestion.rss import RssAdapter

        articles: list[RawArticle] = []
        with RssAdapter() as adapter:
            for key, topic in selected.items():
                if not topic.feeds:
                    continue
                articles.extend(
                    adapter.fetch(feeds=topic.feeds, topic=key, refresh=options.refresh)
                )
        return articles

    raise ValueError(f"Unknown ingestion source: {options.source!r}")


def persist_articles(
    session: Session,
    raw_articles: Sequence[RawArticle],
    *,
    run: IngestionRun | None = None,
    settings: Settings | None = None,
) -> IngestionResult:
    """Normalise, deduplicate and store raw articles idempotently."""
    settings = settings or get_settings()
    policy = load_source_policy()
    detector = DuplicateDetector(
        session,
        title_threshold=settings.near_duplicate_title_threshold,
        window_hours=settings.near_duplicate_window_hours,
        max_candidates=settings.near_duplicate_max_candidates,
    )
    result = IngestionResult(fetched=len(raw_articles))

    for raw in raw_articles:
        try:
            normalized = normalize_article(
                raw, max_excerpt_chars=policy.extraction.max_excerpt_chars
            )
        except ValueError as exc:
            result.skipped += 1
            result.errors.append(f"normalize: {exc}")
            continue

        try:
            verdict = detector.check(normalized)
            if verdict.reason == "canonical_url":
                # Already stored: idempotent re-ingestion, not a new duplicate row.
                # A later adapter may still carry richer metadata (GDELT's artlist
                # mode has no description; the feed does), so fill in blanks only.
                if verdict.duplicate_of_id is not None and _enrich_existing(
                    session, verdict.duplicate_of_id, normalized
                ):
                    result.article_ids.append(verdict.duplicate_of_id)
                result.skipped += 1
                continue

            article = Article(
                url=normalized.url,
                canonical_url=normalized.canonical_url,
                title=normalized.title,
                description=normalized.description,
                excerpt=normalized.excerpt,
                source_domain=normalized.source_domain,
                language=normalized.language,
                topic=normalized.topic,
                published_at=normalized.published_at,
                retrieved_at=utcnow(),
                raw_payload_hash=normalized.raw_payload_hash,
                normalized_text_hash=normalized.normalized_text_hash,
                normalized_title_hash=normalized.normalized_title_hash,
                extraction_method=normalized.extraction_method,
                source_adapter=normalized.source_adapter,
                is_near_duplicate=verdict.is_duplicate,
                duplicate_of_article_id=verdict.duplicate_of_id,
                duplicate_reason=verdict.reason,
                duplicate_similarity=verdict.similarity if verdict.is_duplicate else None,
                ingestion_run_id=run.id if run is not None else None,
                extra=dict(normalized.extra),
            )
            with session.begin_nested():
                session.add(article)
            session.flush()
            index_article(session, article)
            index_article_text(session, article)
            session.flush()
        except SQLAlchemyError as exc:
            session.rollback()
            result.failures += 1
            result.errors.append(f"persist: {exc.__class__.__name__}: {exc}")
            continue

        result.article_ids.append(article.id)
        if verdict.is_duplicate:
            result.duplicates += 1
        else:
            result.inserted += 1

    return result


def _enrich_existing(session: Session, article_id: int, normalized: NormalizedArticle) -> bool:
    """Backfill empty fields on an already-stored article without overwriting data.

    Returns ``True`` when the article text changed, in which case its cached
    embedding is stale and is dropped so the next indexing pass recomputes it.
    """
    article = session.get(Article, article_id)
    if article is None:
        return False
    changed = False
    for field_name in ("description", "excerpt"):
        incoming = str(getattr(normalized, field_name) or "")
        if incoming and not getattr(article, field_name):
            setattr(article, field_name, incoming)
            changed = True
    if not article.published_at and normalized.published_at:
        article.published_at = normalized.published_at
        changed = True
    if changed:
        article.normalized_text_hash = normalized.normalized_text_hash
        session.query(EmbeddingRecord).filter(EmbeddingRecord.article_id == article_id).delete(
            synchronize_session=False
        )
        # The text this article is blocked and searched on just changed.
        index_article(session, article)
        index_article_text(session, article)
        session.flush()
    return changed


def run_ingestion(
    session: Session,
    options: IngestOptions,
    *,
    settings: Settings | None = None,
    raw_articles: Iterable[RawArticle] | None = None,
) -> tuple[IngestionRun, IngestionResult]:
    """Execute one full ingestion run and persist its audit record."""
    settings = settings or get_settings()
    run = _open_run(session, options, settings)
    session.commit()

    try:
        fetched = (
            list(raw_articles)
            if raw_articles is not None
            else collect_raw_articles(options, settings)
        )
    except Exception as exc:
        logger.error("Ingestion fetch failed: %s", exc)
        run.status = "failed"
        run.finished_at = utcnow()
        run.failure_count = 1
        run.errors = {"fetch": f"{exc.__class__.__name__}: {exc}"}
        session.commit()
        return run, IngestionResult(failures=1, errors=[str(exc)])

    result = persist_articles(session, fetched, run=run, settings=settings)

    run.fetched_count = result.fetched
    run.inserted_count = result.inserted
    run.duplicate_count = result.duplicates
    run.skipped_count = result.skipped
    run.failure_count = result.failures
    run.errors = {"messages": result.errors[:50]} if result.errors else {}
    run.status = "completed" if result.failures == 0 else "completed_with_errors"
    run.finished_at = utcnow()
    session.commit()

    if options.index and result.article_ids:
        from newstrace.pipeline import index_articles

        try:
            index_articles(session, result.article_ids)
        except Exception as exc:
            logger.error("Indexing failed after ingestion: %s", exc)
            run.errors = {**(run.errors or {}), "index": str(exc)}
            run.status = "completed_with_errors"
            session.commit()

    logger.info(
        "Ingestion run %s finished: fetched=%s inserted=%s duplicates=%s skipped=%s failures=%s",
        run.id,
        result.fetched,
        result.inserted,
        result.duplicates,
        result.skipped,
        result.failures,
    )
    return run, result


def latest_run(session: Session) -> IngestionRun | None:
    return session.execute(
        select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(1)
    ).scalar_one_or_none()


def run_summary(run: IngestionRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "source": run.source,
        "topic": run.topic,
        "status": run.status,
        "fetched": run.fetched_count,
        "inserted": run.inserted_count,
        "duplicates": run.duplicate_count,
        "skipped": run.skipped_count,
        "failures": run.failure_count,
    }
