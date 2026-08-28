"""Summary data structures shared by the extractive and LLM summarizers.

Every summary statement must carry at least one evidence record.  A statement
without valid evidence is dropped -- never rendered.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field, field_validator


class SummarySection(StrEnum):
    """The five-part answer structure required by the README."""

    NEW_DEVELOPMENTS = "new_developments"
    REPEATED_REPORTING = "repeated_reporting"
    SINGLE_SOURCE = "single_source_claims"
    DISAGREEMENTS = "disagreements_or_uncertainty"


class Evidence(BaseModel):
    """A pointer back to the reporting that supports a statement."""

    evidence_id: str
    article_id: int
    title: str
    source_domain: str
    url: str
    published_at: datetime | None = None
    excerpt: str = ""
    is_near_duplicate: bool = False
    stance: str = "reports"


class SummaryStatement(BaseModel):
    """One statement plus its supporting evidence."""

    section: SummarySection
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    distinct_domains: int = 0
    is_single_source: bool = False
    is_disputed: bool = False
    confidence: float = 0.0

    @field_validator("evidence_ids")
    @classmethod
    def _require_evidence(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("every summary statement must cite at least one evidence id")
        return value


class StorySummary(BaseModel):
    """The full grounded summary of one story cluster."""

    story_id: int
    title: str
    method: str = "extractive"
    generated_at: datetime | None = None
    statements: list[SummaryStatement] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    article_count: int = 0
    independent_source_count: int = 0
    duplicate_count: int = 0
    notes: list[str] = Field(default_factory=list)

    def evidence_index(self) -> dict[str, Evidence]:
        return {e.evidence_id: e for e in self.evidence}

    def citation_coverage(self) -> float:
        if not self.statements:
            return 0.0
        cited = sum(1 for s in self.statements if s.evidence_ids)
        return cited / len(self.statements)

    def validate_citations(self) -> list[str]:
        """Return the ids of statements whose citations do not resolve."""
        known = set(self.evidence_index())
        return [s.text for s in self.statements if not set(s.evidence_ids) <= known]


class Summarizer(Protocol):
    """Interface implemented by the extractive and LLM summarizers."""

    name: str

    def summarize(self, story_id: int) -> StorySummary: ...
