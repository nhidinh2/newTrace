"""Shared ingestion interface.

Every adapter yields :class:`RawArticle` records.  Normalisation, deduplication
and persistence are handled once, in :mod:`newstrace.ingestion.pipeline`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator


class RawArticle(BaseModel):
    """A single article as reported by a source adapter, before normalisation."""

    url: str
    title: str = ""
    description: str = ""
    excerpt: str = ""
    source_domain: str = ""
    language: str = "unknown"
    published_at: datetime | None = None
    topic: str | None = None
    source_adapter: str = "unknown"
    extraction_method: str = "metadata"
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("published_at", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: Any) -> Any:
        """Accept publisher timestamp formats Pydantic will not parse itself.

        GDELT uses ``YYYYMMDDTHHMMSSZ`` and feeds use RFC 822; adapters should
        not have to normalise before constructing a record.
        """
        if isinstance(value, str):
            from newstrace.utils import to_utc

            return to_utc(value)
        return value


@dataclass
class IngestionResult:
    """Counters and errors for one ingestion run."""

    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    skipped: int = 0
    failures: int = 0
    errors: list[str] = field(default_factory=list)
    article_ids: list[int] = field(default_factory=list)

    def merge(self, other: IngestionResult) -> IngestionResult:
        self.fetched += other.fetched
        self.inserted += other.inserted
        self.duplicates += other.duplicates
        self.skipped += other.skipped
        self.failures += other.failures
        self.errors.extend(other.errors)
        self.article_ids.extend(other.article_ids)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "fetched": self.fetched,
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "skipped": self.skipped,
            "failures": self.failures,
            "errors": self.errors[:50],
        }


class SourceAdapter(ABC):
    """Common interface for GDELT, RSS and fixture sources."""

    name: str = "unknown"

    @abstractmethod
    def fetch(self, **kwargs: Any) -> Iterable[RawArticle]:
        """Yield raw articles.  Implementations must not raise on empty results."""
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default no-op
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
