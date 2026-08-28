"""GDELT DOC 2.0 adapter.

Docs: https://www.gdeltproject.org/data.html
      https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

GDELT coverage is neither complete nor unbiased; experiment reports must say so.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from newstrace.ingestion.base import RawArticle, SourceAdapter
from newstrace.ingestion.http import HttpFetcher
from newstrace.ingestion.normalize import extract_domain
from newstrace.logging import get_logger
from newstrace.utils import to_utc

logger = get_logger(__name__)

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_RECORDS_LIMIT = 250
GDELT_DATETIME_FORMAT = "%Y%m%d%H%M%S"
TIMESPAN_UNITS = {
    "min": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "hours": timedelta(hours=1),
    "d": timedelta(days=1),
    "days": timedelta(days=1),
    "w": timedelta(weeks=1),
    "weeks": timedelta(weeks=1),
    "m": timedelta(days=30),
    "months": timedelta(days=30),
}


def parse_timespan(timespan: str) -> timedelta:
    """Parse a GDELT timespan such as ``24h``, ``30min`` or ``7d``.

    Raises ``ValueError`` for anything GDELT would not accept, so a typo fails
    at the call site instead of silently querying the wrong window.
    """
    match = re.fullmatch(r"\s*(\d+)\s*(min|hours|h|days|d|weeks|w|months|m)\s*", timespan.lower())
    if not match:
        raise ValueError(f"Unrecognised GDELT timespan: {timespan!r}")
    return int(match.group(1)) * TIMESPAN_UNITS[match.group(2)]


def time_windows(
    timespan: str,
    windows: int,
    *,
    end: datetime | None = None,
) -> list[tuple[datetime, datetime]]:
    """Split ``timespan`` into ``windows`` consecutive UTC slices, oldest first.

    GDELT caps one ``artlist`` response at 250 records, so a busy query is
    sampled by asking for consecutive sub-windows instead of one wide window.
    Every slice is closed-open, and the slices exactly tile the timespan.
    """
    count = max(1, int(windows))
    stop = (end or datetime.now(tz=UTC)).astimezone(UTC)
    total = parse_timespan(timespan)
    step = total / count
    start = stop - total
    slices: list[tuple[datetime, datetime]] = []
    for index in range(count):
        window_start = start + step * index
        window_end = stop if index == count - 1 else start + step * (index + 1)
        slices.append((window_start, window_end))
    return slices


def window_params(start: datetime, end: datetime) -> dict[str, str]:
    """GDELT ``startdatetime``/``enddatetime`` parameters for one window."""
    return {
        "startdatetime": start.astimezone(UTC).strftime(GDELT_DATETIME_FORMAT),
        "enddatetime": end.astimezone(UTC).strftime(GDELT_DATETIME_FORMAT),
    }


def build_params(
    query: str,
    *,
    timespan: str = "24h",
    max_records: int = 250,
    mode: str = "artlist",
    sort: str = "datedesc",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build GDELT query parameters (encoded later by httpx, never concatenated).

    ``timespan`` and ``startdatetime``/``enddatetime`` are mutually exclusive in
    GDELT DOC 2.0, so an explicit window in ``extra`` drops ``timespan``.
    """
    params: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "format": "json",
        "maxrecords": min(int(max_records), MAX_RECORDS_LIMIT),
        "timespan": timespan,
        "sort": sort,
    }
    if extra:
        params.update(extra)
    if "startdatetime" in params or "enddatetime" in params:
        params.pop("timespan", None)
    return params


def parse_articles(payload: Any, *, topic: str | None = None) -> list[RawArticle]:
    """Parse a GDELT ``artlist`` payload defensively."""
    if not isinstance(payload, dict):
        logger.warning("Unexpected GDELT payload type: %s", type(payload).__name__)
        return []
    records = payload.get("articles")
    if not isinstance(records, list):
        return []

    articles: list[RawArticle] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        url = str(record.get("url") or "").strip()
        if not url:
            continue
        domain = str(record.get("domain") or "").strip() or extract_domain(url)
        articles.append(
            RawArticle(
                url=url,
                title=str(record.get("title") or "").strip(),
                description="",
                excerpt="",
                source_domain=domain,
                language=str(record.get("language") or "unknown"),
                published_at=to_utc(str(record.get("seendate") or "")),
                topic=topic,
                source_adapter="gdelt",
                extraction_method="metadata",
                raw={k: v for k, v in record.items() if isinstance(v, (str, int, float))},
            )
        )
    return articles


class GdeltAdapter(SourceAdapter):
    """Fetch article metadata from the GDELT DOC 2.0 API."""

    name = "gdelt"
    endpoint = GDELT_DOC_ENDPOINT

    def __init__(self, fetcher: HttpFetcher | None = None) -> None:
        self._fetcher = fetcher
        self._owns_fetcher = fetcher is None

    @property
    def fetcher(self) -> HttpFetcher:
        if self._fetcher is None:
            self._fetcher = HttpFetcher()
        return self._fetcher

    def fetch(  # type: ignore[override]
        self,
        *,
        query: str,
        topic: str | None = None,
        timespan: str = "24h",
        max_records: int = 250,
        refresh: bool = False,
        window: tuple[datetime, datetime] | None = None,
        **_: Any,
    ) -> Iterable[RawArticle]:
        extra = window_params(*window) if window else None
        params = build_params(query, timespan=timespan, max_records=max_records, extra=extra)
        payload = self.fetcher.get_json(self.endpoint, params=params, refresh=refresh)
        return parse_articles(payload, topic=topic)

    def fetch_topics(
        self,
        topics: dict[str, Any],
        *,
        timespan: str = "24h",
        max_records: int = 250,
        refresh: bool = False,
        windows: int = 1,
        end: datetime | None = None,
    ) -> Iterator[RawArticle]:
        """Fetch every configured topic, optionally sub-window by sub-window.

        ``windows > 1`` splits the timespan so one topic can yield more than the
        250 records a single ``artlist`` response carries.
        """
        slices = time_windows(timespan, windows, end=end) if windows > 1 else [None]
        for key, topic in topics.items():
            query = getattr(topic, "gdelt_query", "") or ""
            if not query:
                continue
            for window in slices:
                yield from self.fetch(
                    query=query,
                    topic=key,
                    timespan=timespan,
                    max_records=max_records,
                    refresh=refresh,
                    window=window,
                )

    def close(self) -> None:
        if self._owns_fetcher and self._fetcher is not None:
            self._fetcher.close()
