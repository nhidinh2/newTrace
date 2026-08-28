"""RSS/Atom adapter built on feedparser.

Feed lists are user-owned YAML.  Private or credentialed feed URLs must never be
committed to the repository.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import feedparser

from newstrace.ingestion.base import RawArticle, SourceAdapter
from newstrace.ingestion.http import HttpFetcher
from newstrace.ingestion.normalize import extract_domain
from newstrace.logging import get_logger
from newstrace.utils import normalize_unicode, to_utc

logger = get_logger(__name__)


def _entry_text(entry: Any, key: str) -> str:
    value = entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
    if isinstance(value, list) and value:
        value = value[0].get("value") if isinstance(value[0], dict) else value[0]
    if isinstance(value, dict):
        value = value.get("value", "")
    return normalize_unicode(str(value or ""))


def _strip_html(text: str) -> str:
    import re

    return normalize_unicode(re.sub(r"<[^>]+>", " ", text or ""))


def parse_feed(text: str, *, topic: str | None = None, feed_url: str = "") -> list[RawArticle]:
    """Parse feed XML into raw articles.  Malformed feeds yield an empty list."""
    parsed = feedparser.parse(text)
    entries = getattr(parsed, "entries", []) or []
    feed_title = _entry_text(getattr(parsed, "feed", {}) or {}, "title")

    articles: list[RawArticle] = []
    for entry in entries:
        url = _entry_text(entry, "link") or _entry_text(entry, "id")
        if not url:
            continue
        published = (
            _entry_text(entry, "published")
            or _entry_text(entry, "updated")
            or _entry_text(entry, "created")
        )
        summary = _strip_html(_entry_text(entry, "summary"))
        content = _strip_html(_entry_text(entry, "content"))
        articles.append(
            RawArticle(
                url=url,
                title=_entry_text(entry, "title"),
                description=summary,
                excerpt=content or summary,
                source_domain=extract_domain(url),
                language=_entry_text(entry, "language") or "unknown",
                published_at=to_utc(published),
                topic=topic,
                source_adapter="rss",
                extraction_method="feed_metadata",
                raw={"feed_title": feed_title, "feed_url": feed_url},
            )
        )
    return articles


class RssAdapter(SourceAdapter):
    """Fetch and parse a list of permitted RSS/Atom feeds."""

    name = "rss"

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
        feeds: Iterable[str],
        topic: str | None = None,
        refresh: bool = False,
        **_: Any,
    ) -> Iterable[RawArticle]:
        articles: list[RawArticle] = []
        for feed_url in feeds:
            try:
                text = self.fetcher.get_text(feed_url, refresh=refresh)
            except Exception as exc:
                logger.warning("Failed to fetch feed %s: %s", feed_url, exc)
                continue
            articles.extend(parse_feed(text, topic=topic, feed_url=feed_url))
        return articles

    def close(self) -> None:
        if self._owns_fetcher and self._fetcher is not None:
            self._fetcher.close()
