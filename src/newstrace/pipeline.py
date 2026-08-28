"""Post-ingestion indexing: embed -> project -> cluster -> extract claims.

This is the streaming half of the system: articles arrive, are embedded once
(cached), assigned to a story with a bounded-window online clusterer, and their
claims extracted.  Per-article update latency is recorded so throughput and
latency can be reported.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.claims.evidence import persist_claims
from newstrace.clustering.online import Assignment, OnlineClusterer
from newstrace.config import Settings, get_settings
from newstrace.logging import get_logger
from newstrace.models import Article
from newstrace.representations.embedder import from_blob, get_embedder
from newstrace.representations.registry import ensure_embeddings
from newstrace.stories import load_story
from newstrace.utils import ensure_utc, utcnow

logger = get_logger(__name__)


@dataclass
class IndexResult:
    """Counters and latency measurements for one indexing pass."""

    embedded: int = 0
    assigned: int = 0
    new_stories: int = 0
    claims: int = 0
    per_article_latency_ms: list[float] = field(default_factory=list)
    total_seconds: float = 0.0

    @property
    def throughput_articles_per_second(self) -> float:
        return (self.assigned / self.total_seconds) if self.total_seconds > 0 else 0.0

    def latency_summary(self) -> dict[str, float]:
        if not self.per_article_latency_ms:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(self.per_article_latency_ms)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
        }

    def as_dict(self) -> dict[str, float | int]:
        return {
            "embedded": self.embedded,
            "assigned": self.assigned,
            "new_stories": self.new_stories,
            "claims": self.claims,
            "total_seconds": round(self.total_seconds, 4),
            "throughput_articles_per_second": round(self.throughput_articles_per_second, 3),
            **{k: round(v, 3) for k, v in self.latency_summary().items()},
        }


def index_articles(
    session: Session,
    article_ids: Sequence[int],
    *,
    settings: Settings | None = None,
    extract_claims: bool = True,
) -> IndexResult:
    """Embed, cluster and (optionally) extract claims for the given articles."""
    settings = settings or get_settings()
    result = IndexResult()
    if not article_ids:
        return result

    articles = list(
        session.execute(select(Article).where(Article.id.in_(list(article_ids)))).scalars()
    )
    if not articles:
        return result

    started = time.perf_counter()
    embedder = get_embedder(settings)
    records = ensure_embeddings(session, articles, embedder=embedder, settings=settings)
    result.embedded = len(records)
    vector_by_article = {r.article_id: from_blob(r.vector_blob, r.dimension) for r in records}

    clusterer = OnlineClusterer(session, settings=settings, embedding_method="full")
    ordered = sorted(articles, key=lambda a: (ensure_utc(a.published_at) or utcnow(), a.id))
    assignments: list[Assignment] = []
    for article in ordered:
        vector = vector_by_article.get(article.id)
        if vector is None:
            continue
        tick = time.perf_counter()
        assignment = clusterer.assign(article, vector)
        result.per_article_latency_ms.append((time.perf_counter() - tick) * 1000.0)
        assignments.append(assignment)
        result.assigned += 1
        result.new_stories += int(assignment.created_new)
    session.flush()

    if extract_claims:
        for story_id in sorted({a.story_id for a in assignments}):
            bundle = load_story(session, story_id)
            if bundle is None:
                continue
            result.claims += len(persist_claims(session, story_id, bundle.articles))

    session.commit()
    result.total_seconds = time.perf_counter() - started
    logger.info("Indexed %s articles: %s", result.assigned, result.as_dict())
    return result


def reindex_all(session: Session, *, settings: Settings | None = None) -> IndexResult:
    """Re-run indexing over every stored article (used after config changes)."""
    ids = list(session.execute(select(Article.id).order_by(Article.id)).scalars())
    return index_articles(session, ids, settings=settings)
