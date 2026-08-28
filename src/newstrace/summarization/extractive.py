"""Deterministic, citation-grounded extractive summarizer.

The output is assembled entirely from sentences that exist in the retrieved
material.  Nothing is generated.  Each statement carries the evidence ids of the
articles it was lifted from, and statements whose evidence does not resolve are
dropped before the summary is returned.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.orm import Session

from newstrace.claims.compare import ClaimGroup, find_conflicts, group_claims
from newstrace.claims.evidence import collect_occurrences
from newstrace.models import Article
from newstrace.stories import StoryArticles, build_timeline, load_story
from newstrace.summarization.base import (
    Evidence,
    StorySummary,
    SummarySection,
    SummaryStatement,
)
from newstrace.utils import ensure_utc, jaccard, sha256_text, token_set, truncate

MAX_PER_SECTION = 5


def evidence_id(article_id: int, excerpt: str = "") -> str:
    """Evidence ids are per (article, excerpt), not per article.

    A statement must cite the exact passage it was lifted from; citing the
    article generally would let a summary point at text that does not contain
    what the statement says.
    """
    if not excerpt:
        return f"ev-{article_id}"
    return f"ev-{article_id}-{sha256_text(excerpt)[:8]}"


class EvidenceBuilder:
    """Collects the exact excerpts cited by statements, deduplicated by id."""

    def __init__(self, articles: Sequence[Article]) -> None:
        self._articles = {a.id: a for a in articles}
        self._records: dict[str, Evidence] = {}

    def cite(self, article_id: int, excerpt: str, *, stance: str = "reports") -> str | None:
        article = self._articles.get(article_id)
        if article is None:
            return None
        text = truncate(excerpt or article.excerpt or article.description or article.title, 400)
        eid = evidence_id(article_id, text)
        if eid not in self._records:
            self._records[eid] = Evidence(
                evidence_id=eid,
                article_id=article_id,
                title=article.title,
                source_domain=article.source_domain,
                url=article.url,
                published_at=ensure_utc(article.published_at),
                excerpt=text,
                is_near_duplicate=bool(article.is_near_duplicate),
                stance=stance,
            )
        return eid

    def records(self) -> list[Evidence]:
        return sorted(self._records.values(), key=lambda e: (e.article_id, e.evidence_id))


def _statement(
    section: SummarySection,
    text: str,
    evidence_ids: Sequence[str],
    *,
    distinct_domains: int,
    is_single_source: bool = False,
    is_disputed: bool = False,
    confidence: float = 0.0,
) -> SummaryStatement:
    return SummaryStatement(
        section=section,
        text=truncate(text, 400),
        evidence_ids=list(dict.fromkeys(evidence_ids)),
        distinct_domains=distinct_domains,
        is_single_source=is_single_source,
        is_disputed=is_disputed,
        confidence=confidence,
    )


class ExtractiveSummarizer:
    """Builds the five-part grounded summary described in the README."""

    name = "extractive"

    def __init__(self, session: Session, *, max_per_section: int = MAX_PER_SECTION) -> None:
        self.session = session
        self.max_per_section = max_per_section

    def summarize(self, story_id: int) -> StorySummary:
        bundle = load_story(self.session, story_id)
        if bundle is None:
            raise LookupError(f"story {story_id} not found")
        return self.summarize_bundle(bundle)

    def summarize_bundle(self, bundle: StoryArticles) -> StorySummary:
        builder = EvidenceBuilder(bundle.articles)
        statements: list[SummaryStatement] = []
        seen_tokens: list[set[str]] = []

        def add(statement: SummaryStatement) -> None:
            """Append a statement unless it repeats one already in the summary."""
            tokens = token_set(statement.text)
            if any(jaccard(tokens, prior) > 0.7 for prior in seen_tokens):
                return
            seen_tokens.append(tokens)
            statements.append(statement)

        # 1. New developments -- sentences the timeline marked as previously unseen.
        timeline = build_timeline(bundle, include_duplicates=False)
        new_count = 0
        for entry in reversed(timeline):
            for sentence in entry.new_sentences[:2]:
                if new_count >= self.max_per_section:
                    break
                eid = builder.cite(entry.article.id, sentence)
                if eid is None:
                    continue
                before = len(statements)
                add(
                    _statement(
                        SummarySection.NEW_DEVELOPMENTS,
                        sentence,
                        [eid],
                        distinct_domains=1,
                        is_single_source=True,
                        confidence=0.4,
                    )
                )
                new_count += len(statements) - before
            if new_count >= self.max_per_section:
                break

        # 2/3/4. Claim groups -> repeated, single-source and unclear statements.
        groups = group_claims(collect_occurrences(bundle.articles))
        repeated: list[ClaimGroup] = []
        single: list[ClaimGroup] = []
        unclear: list[ClaimGroup] = []
        for group in groups:
            status = group.status()
            if status == "repeated":
                repeated.append(group)
            elif status == "unclear":
                unclear.append(group)
            else:
                single.append(group)

        def cite_group(group: ClaimGroup, *, independent_only: bool, stance: str) -> list[str]:
            ids: list[str] = []
            for occurrence in group.occurrences:
                if independent_only and occurrence.is_near_duplicate:
                    continue
                eid = builder.cite(occurrence.article_id, occurrence.claim.text, stance=stance)
                if eid is not None:
                    ids.append(eid)
            return ids

        for group in repeated[: self.max_per_section]:
            ids = cite_group(group, independent_only=True, stance="supports")
            if not ids:
                continue
            add(
                _statement(
                    SummarySection.REPEATED_REPORTING,
                    group.representative.text,
                    ids,
                    distinct_domains=len(group.independent_domains),
                    confidence=group.confidence(),
                )
            )

        for group in single[: self.max_per_section]:
            ids = cite_group(group, independent_only=False, stance="reports")
            if not ids:
                continue
            add(
                _statement(
                    SummarySection.SINGLE_SOURCE,
                    group.representative.text,
                    ids,
                    distinct_domains=len(group.independent_domains),
                    is_single_source=True,
                    confidence=group.confidence(),
                )
            )

        for group in unclear[: self.max_per_section]:
            ids = cite_group(group, independent_only=False, stance="unclear")
            if not ids:
                continue
            add(
                _statement(
                    SummarySection.DISAGREEMENTS,
                    group.representative.text,
                    ids,
                    distinct_domains=len(group.independent_domains),
                    is_disputed=True,
                    confidence=group.confidence(),
                )
            )

        # Cross-claim disagreements: two passages about the same thing that the
        # available evidence cannot reconcile. Both sides are quoted; neither is
        # endorsed.
        for conflict in find_conflicts(groups)[: self.max_per_section]:
            ids = cite_group(conflict.left, independent_only=False, stance="unclear")
            ids += cite_group(conflict.right, independent_only=False, stance="disputes")
            if len(ids) < 2:
                continue
            detail = f" ({conflict.detail})" if conflict.detail else ""
            add(
                _statement(
                    SummarySection.DISAGREEMENTS,
                    f"Sources do not agree{detail}: "
                    f"\u201c{conflict.left.representative.text}\u201d versus "
                    f"\u201c{conflict.right.representative.text}\u201d",
                    ids,
                    distinct_domains=len(conflict.independent_domains()),
                    is_disputed=True,
                    confidence=0.0,
                )
            )

        evidence = builder.records()
        summary = StorySummary(
            story_id=bundle.story.id,
            title=bundle.story.display_title,
            method=self.name,
            generated_at=None,
            statements=statements,
            evidence=evidence,
            article_count=len(bundle.articles),
            independent_source_count=len(bundle.independent_domains),
            duplicate_count=len(bundle.duplicates),
            notes=[
                "Statements are quoted from the retrieved reporting; nothing is generated.",
                "Agreement between sources is not evidence of truth, and copied or "
                "syndicated articles are excluded from independent-source counts.",
            ],
        )
        return drop_uncited(summary)


def drop_uncited(summary: StorySummary) -> StorySummary:
    """Remove any statement whose evidence ids do not all resolve."""
    known = set(summary.evidence_index())
    summary.statements = [
        s for s in summary.statements if s.evidence_ids and set(s.evidence_ids) <= known
    ]
    return summary
