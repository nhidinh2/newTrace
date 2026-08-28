"""Persisting claims and their evidence, and reading them back for a story."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.claims.compare import ClaimGroup, ClaimOccurrence, group_claims
from newstrace.claims.extract import extract_claims
from newstrace.models import Article, Claim, ClaimEvidence
from newstrace.utils import ensure_utc, truncate

STANCE_REPORTS = "reports"
STANCE_SUPPORTS = "supports"
STANCE_DISPUTES = "disputes"
STANCE_UNCLEAR = "unclear"


def collect_occurrences(articles: Sequence[Article]) -> list[ClaimOccurrence]:
    """Extract claims from every article's available text."""
    occurrences: list[ClaimOccurrence] = []
    for article in articles:
        for claim in extract_claims(article.body_text):
            occurrences.append(
                ClaimOccurrence(
                    article_id=article.id,
                    source_domain=article.source_domain,
                    is_near_duplicate=bool(article.is_near_duplicate),
                    claim=claim,
                )
            )
    return occurrences


def stance_for(group: ClaimGroup, occurrence: ClaimOccurrence) -> str:
    """Conservative stance label for one occurrence within its group."""
    if group.is_conflicting():
        return STANCE_UNCLEAR if occurrence.claim.is_negated else STANCE_DISPUTES
    if len(group.independent_domains) > 1 and not occurrence.is_near_duplicate:
        return STANCE_SUPPORTS
    return STANCE_REPORTS


def persist_claims(
    session: Session, story_id: int, articles: Sequence[Article], *, max_claims: int = 40
) -> list[Claim]:
    """Extract, group and store claims for one story (idempotent by claim hash)."""
    groups = group_claims(collect_occurrences(articles))[:max_claims]
    existing = {
        c.claim_hash: c
        for c in session.execute(select(Claim).where(Claim.story_cluster_id == story_id)).scalars()
    }
    stored: list[Claim] = []
    for group in groups:
        rep = group.representative
        claim = existing.get(rep.claim_hash)
        published = [
            stamp
            for stamp in (ensure_utc(a.published_at) for a in _articles_for(articles, group))
            if stamp is not None
        ]
        earliest = min(published) if published else None
        if claim is None:
            claim = Claim(
                story_cluster_id=story_id,
                normalized_claim=rep.text,
                claim_hash=rep.claim_hash,
                subject=rep.subject,
                predicate=rep.predicate,
                object=rep.object,
                event_time=earliest,
                extraction_method=rep.extraction_method,
                confidence=group.confidence(),
            )
            session.add(claim)
            session.flush()
        else:
            claim.confidence = group.confidence()
            claim.event_time = earliest
        stored.append(claim)

        known = {
            (e.article_id, e.excerpt_start)
            for e in session.execute(
                select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id)
            ).scalars()
        }
        for occurrence in group.occurrences:
            key = (occurrence.article_id, occurrence.claim.start)
            if key in known:
                continue
            session.add(
                ClaimEvidence(
                    claim_id=claim.id,
                    article_id=occurrence.article_id,
                    excerpt=truncate(occurrence.claim.text, 400),
                    excerpt_start=occurrence.claim.start,
                    excerpt_end=occurrence.claim.end,
                    stance=stance_for(group, occurrence),
                    confidence=occurrence.claim.confidence,
                )
            )
    session.flush()
    return stored


def _articles_for(articles: Sequence[Article], group: ClaimGroup) -> list[Article]:
    ids = {o.article_id for o in group.occurrences}
    return [a for a in articles if a.id in ids]


def load_claims(session: Session, story_id: int) -> list[tuple[Claim, list[ClaimEvidence]]]:
    """Load a story's claims with their evidence, ordered by confidence."""
    claims = list(
        session.execute(
            select(Claim)
            .where(Claim.story_cluster_id == story_id)
            .order_by(Claim.confidence.desc(), Claim.id)
        ).scalars()
    )
    out: list[tuple[Claim, list[ClaimEvidence]]] = []
    for claim in claims:
        evidence = list(
            session.execute(
                select(ClaimEvidence)
                .where(ClaimEvidence.claim_id == claim.id)
                .order_by(ClaimEvidence.id)
            ).scalars()
        )
        out.append((claim, evidence))
    return out
