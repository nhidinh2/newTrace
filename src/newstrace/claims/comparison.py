"""Source-by-source inspection of stored evidence, without new inference."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.models import Article, Claim, ClaimEvidence
from newstrace.schemas import (
    ClaimEvidenceResponse,
    ComparisonCell,
    ComparisonClaim,
    ComparisonSource,
    StoryComparisonResponse,
)
from newstrace.stories import StoryArticles
from newstrace.utils import ensure_utc


def build_source_comparison(session: Session, bundle: StoryArticles) -> StoryComparisonResponse:
    """Read claims/evidence in one joined query; keep evidence-free claims.

    Cells are sparse: absent domains have no extracted evidence, not a negative
    stance. Article counts deduplicate repeated excerpts from the same report.
    """
    by_domain: dict[str, dict[int, Article]] = defaultdict(dict)
    for article in bundle.articles:
        by_domain[article.source_domain][article.id] = article

    claims: dict[int, Claim] = {}
    evidence: dict[int, dict[str, list[ClaimEvidenceResponse]]] = defaultdict(
        lambda: defaultdict(list)
    )
    rows = session.execute(
        select(Claim, ClaimEvidence, Article)
        .outerjoin(ClaimEvidence, ClaimEvidence.claim_id == Claim.id)
        .outerjoin(Article, Article.id == ClaimEvidence.article_id)
        .where(Claim.story_cluster_id == bundle.story.id)
        .order_by(Claim.confidence.desc(), Claim.id, ClaimEvidence.id)
    )
    for claim, ev, article in rows:
        claims[claim.id] = claim
        if ev is None or article is None:
            continue
        # Persisted provenance remains inspectable even if membership changed.
        by_domain[article.source_domain][article.id] = article
        evidence[claim.id][article.source_domain].append(
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

    result: list[ComparisonClaim] = []
    for claim in claims.values():
        cells: list[ComparisonCell] = []
        for domain, passages in sorted(evidence[claim.id].items()):
            cells.append(
                ComparisonCell(
                    source_domain=domain,
                    independent_article_count=len(
                        {p.article_id for p in passages if not p.is_near_duplicate}
                    ),
                    duplicate_article_count=len(
                        {p.article_id for p in passages if p.is_near_duplicate}
                    ),
                    stances=sorted({p.stance for p in passages if not p.is_near_duplicate}),
                    evidence=passages,
                )
            )
        result.append(
            ComparisonClaim(
                claim_id=claim.id,
                text=claim.normalized_claim,
                extraction_method=claim.extraction_method,
                confidence=claim.confidence,
                independent_domain_count=sum(
                    bool(c.source_domain) and c.independent_article_count > 0 for c in cells
                ),
                cells=cells,
            )
        )
    return StoryComparisonResponse(
        story_id=bundle.story.id,
        sources=[
            ComparisonSource(
                source_domain=domain,
                article_count=len(articles),
                duplicate_article_count=sum(bool(a.is_near_duplicate) for a in articles.values()),
            )
            for domain, articles in sorted(by_domain.items())
        ],
        claims=result,
    )
