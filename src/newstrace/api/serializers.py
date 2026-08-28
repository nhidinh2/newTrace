"""ORM -> schema conversion shared by the API, CLI and Streamlit UI."""

from __future__ import annotations

from sqlalchemy.orm import Session

from newstrace.claims.compare import ClaimGroup
from newstrace.models import Article, Claim, ClaimEvidence, IngestionRun, StoryCluster
from newstrace.schemas import (
    ArticleResponse,
    ClaimEvidenceResponse,
    ClaimResponse,
    IngestionRunResponse,
    SearchResultResponse,
    StoryDetailResponse,
    StorySummaryResponse,
    TimelineEntryResponse,
)
from newstrace.search import SearchResultItem
from newstrace.stories import StoryArticles, TimelineEntry, story_keywords
from newstrace.utils import ensure_utc


def article_to_schema(article: Article) -> ArticleResponse:
    return ArticleResponse(
        id=article.id,
        title=article.title,
        url=article.url,
        source_domain=article.source_domain,
        language=article.language,
        topic=article.topic,
        published_at=ensure_utc(article.published_at),
        retrieved_at=ensure_utc(article.retrieved_at),
        excerpt=article.excerpt or article.description,
        is_near_duplicate=bool(article.is_near_duplicate),
        duplicate_of_article_id=article.duplicate_of_article_id,
        extraction_method=article.extraction_method,
    )


def story_to_schema(
    story: StoryCluster, bundle: StoryArticles | None = None
) -> StorySummaryResponse:
    independent = len(bundle.independent) if bundle else story.article_count
    duplicates = len(bundle.duplicates) if bundle else 0
    domains = bundle.independent_domains if bundle else list((story.extra or {}).get("domains", []))
    return StorySummaryResponse(
        id=story.id,
        display_title=story.display_title,
        topic=story.topic,
        article_count=story.article_count,
        independent_article_count=independent,
        distinct_domain_count=story.distinct_domain_count,
        duplicate_count=duplicates,
        first_published_at=ensure_utc(story.first_published_at),
        last_published_at=ensure_utc(story.last_published_at),
        keywords=story_keywords(bundle) if bundle else [],
        domains=domains,
    )


def story_detail_to_schema(bundle: StoryArticles) -> StoryDetailResponse:
    base = story_to_schema(bundle.story, bundle)
    return StoryDetailResponse(
        **base.model_dump(),
        articles=[article_to_schema(a) for a in bundle.sorted_by_time(include_duplicates=True)],
    )


def timeline_entry_to_schema(entry: TimelineEntry) -> TimelineEntryResponse:
    return TimelineEntryResponse(
        article=article_to_schema(entry.article),
        published_at=entry.published_at,
        similarity=entry.similarity,
        is_near_duplicate=entry.is_near_duplicate,
        duplicate_of_article_id=entry.duplicate_of_article_id,
        is_first_report=entry.is_first_report,
        new_sentences=entry.new_sentences,
        repeated_sentences=entry.repeated_sentences,
    )


def claim_to_schema(session: Session, claim: Claim, evidence: list[ClaimEvidence]) -> ClaimResponse:
    articles = {
        a.id: a
        for a in session.query(Article).filter(Article.id.in_([e.article_id for e in evidence]))
    }
    rows: list[ClaimEvidenceResponse] = []
    independent_domains: set[str] = set()
    duplicates = 0
    disputed = False
    for ev in evidence:
        article = articles.get(ev.article_id)
        if article is None:
            continue
        if article.is_near_duplicate:
            duplicates += 1
        else:
            independent_domains.add(article.source_domain)
        disputed = disputed or ev.stance in ("disputes", "unclear")
        rows.append(
            ClaimEvidenceResponse(
                article_id=article.id,
                title=article.title,
                source_domain=article.source_domain,
                url=article.url,
                published_at=ensure_utc(article.published_at),
                excerpt=ev.excerpt,
                stance=ev.stance,
                confidence=ev.confidence,
                is_near_duplicate=bool(article.is_near_duplicate),
            )
        )
    if disputed:
        status = "unclear"
    elif len(independent_domains) > 1:
        status = "repeated"
    else:
        status = "single_source"
    return ClaimResponse(
        id=claim.id,
        normalized_claim=claim.normalized_claim,
        subject=claim.subject,
        predicate=claim.predicate,
        object=claim.object,
        event_time=ensure_utc(claim.event_time),
        extraction_method=claim.extraction_method,
        confidence=claim.confidence,
        status=status,  # type: ignore[arg-type]
        independent_domain_count=len(independent_domains),
        duplicate_evidence_count=duplicates,
        evidence=rows,
    )


def search_item_to_schema(item: SearchResultItem) -> SearchResultResponse:
    return SearchResultResponse(
        article_id=item.article.id,
        story_id=item.story_id,
        title=item.article.title,
        source_domain=item.article.source_domain,
        published_at=ensure_utc(item.article.published_at),
        url=item.article.url,
        score=item.hit.score,
        ranking_method=item.hit.method,
        signals={k: float(v) for k, v in item.hit.signals.items()},
        excerpt=item.excerpt,
        explanation=item.explanation,
        is_near_duplicate=bool(item.article.is_near_duplicate),
    )


def run_to_schema(run: IngestionRun) -> IngestionRunResponse:
    return IngestionRunResponse(
        id=run.id,
        source=run.source,
        topic=run.topic,
        status=run.status,
        started_at=ensure_utc(run.started_at),  # type: ignore[arg-type]
        finished_at=ensure_utc(run.finished_at),
        fetched_count=run.fetched_count,
        inserted_count=run.inserted_count,
        duplicate_count=run.duplicate_count,
        skipped_count=run.skipped_count,
        failure_count=run.failure_count,
        errors=run.errors or {},
        git_commit=run.git_commit,
        random_seed=run.random_seed,
    )


def group_status(group: ClaimGroup) -> str:
    return group.status()
