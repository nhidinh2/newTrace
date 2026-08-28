"""Summary and evidence metrics.

Citation *precision* here means: the cited excerpt actually contains the words of
the statement it supports.  It does not mean the statement is true.
"""

from __future__ import annotations

from dataclasses import dataclass

from newstrace.summarization.base import StorySummary
from newstrace.utils import jaccard, token_set

SUPPORT_THRESHOLD = 0.6


@dataclass
class SummaryMetrics:
    citation_coverage: float = 0.0
    citation_precision: float = 0.0
    unsupported_claim_rate: float = 0.0
    redundancy_rate: float = 0.0
    source_domain_diversity: int = 0
    statements: int = 0
    single_source_statements: int = 0
    disputed_statements: int = 0

    def as_dict(self) -> dict[str, float]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def statement_supported(statement_text: str, excerpts: list[str]) -> bool:
    """A statement is supported when its tokens are covered by a cited excerpt."""
    tokens = token_set(statement_text)
    if not tokens:
        return False
    for excerpt in excerpts:
        excerpt_tokens = token_set(excerpt)
        if not excerpt_tokens:
            continue
        coverage = len(tokens & excerpt_tokens) / len(tokens)
        if coverage >= SUPPORT_THRESHOLD:
            return True
    return False


def evaluate_summary(summary: StorySummary) -> SummaryMetrics:
    index = summary.evidence_index()
    total = len(summary.statements)
    if total == 0:
        return SummaryMetrics()

    cited = 0
    supported = 0
    domains: set[str] = set()
    seen_tokens: list[set[str]] = []
    redundant = 0

    for statement in summary.statements:
        excerpts = []
        for eid in statement.evidence_ids:
            evidence = index.get(eid)
            if evidence is None:
                continue
            excerpts.append(evidence.excerpt or evidence.title)
            if not evidence.is_near_duplicate:
                domains.add(evidence.source_domain)
        if statement.evidence_ids:
            cited += 1
        if statement_supported(statement.text, excerpts):
            supported += 1
        tokens = token_set(statement.text)
        if any(jaccard(tokens, prior) > 0.8 for prior in seen_tokens):
            redundant += 1
        seen_tokens.append(tokens)

    return SummaryMetrics(
        citation_coverage=cited / total,
        citation_precision=supported / max(1, cited),
        unsupported_claim_rate=(cited - supported) / max(1, cited),
        redundancy_rate=redundant / total,
        source_domain_diversity=len(domains),
        statements=total,
        single_source_statements=sum(1 for s in summary.statements if s.is_single_source),
        disputed_statements=sum(1 for s in summary.statements if s.is_disputed),
    )
