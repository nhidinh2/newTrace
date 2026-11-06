"""A compact, citation-first digest of recent stories.

This is the export other tools read: a morning-news note, a static website. Every
story carries the reporting it came from, and its source count is the number of
*independent* domains, so a wire story syndicated nine times still counts once.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from newstrace.stories import StoryArticles, list_stories, load_story
from newstrace.utils import ensure_utc

# RSS feeds rarely declare a language, so "unknown" is kept; a story whose earliest
# report is declared in another language is left out of an English digest.
ENGLISH = {"english", "en", "unknown", ""}


def _iso(value: datetime | None) -> str | None:
    aware = ensure_utc(value)
    return aware.isoformat() if aware else None


def _source(article: Any) -> dict[str, Any]:
    return {
        "domain": article.source_domain,
        "title": article.title,
        "url": article.url,
        "published_at": _iso(article.published_at),
    }


def story_entry(bundle: StoryArticles, *, max_sources: int = 5) -> dict[str, Any] | None:
    """Summarise one story, or return None when it has no independent report."""
    independent = bundle.sorted_by_time()
    if not independent:
        return None
    lead = independent[0]
    # One link per domain: the earliest report from each outlet.
    per_domain: dict[str, Any] = {}
    for article in independent:
        per_domain.setdefault(article.source_domain, article)
    domains = bundle.independent_domains
    return {
        "id": bundle.story.id,
        "title": lead.title or bundle.story.display_title,
        "topic": bundle.story.topic,
        "summary": (lead.description or lead.excerpt or "").strip()[:280],
        "independent_sources": len(domains),
        "articles": len(bundle.articles),
        "duplicates": len(bundle.duplicates),
        "single_source": len(domains) <= 1,
        "first_published_at": _iso(bundle.story.first_published_at),
        "last_published_at": _iso(bundle.story.last_published_at),
        "earliest_report": _source(lead),
        "sources": [_source(a) for a in list(per_domain.values())[:max_sources]],
    }


def build_digest(
    session: Session,
    *,
    topics: list[str],
    since_hours: float = 48,
    per_topic: int = 5,
    min_sources: int = 1,
    english_only: bool = True,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return the top recent stories per topic, ranked by independent sources."""
    now = ensure_utc(now) or datetime.now(UTC)
    start = now - timedelta(hours=since_hours)
    sections: dict[str, list[dict[str, Any]]] = {}
    for topic in topics:
        entries: list[dict[str, Any]] = []
        # Over-fetch: language and source filters drop some candidates.
        for story in list_stories(session, topic=topic, start_time=start, limit=per_topic * 6):
            bundle = load_story(session, story.id)
            entry = story_entry(bundle) if bundle else None
            if entry is None or entry["independent_sources"] < min_sources:
                continue
            if english_only and bundle is not None:
                language = (bundle.sorted_by_time()[0].language or "").lower()
                if language not in ENGLISH:
                    continue
            entries.append(entry)
            if len(entries) >= per_topic:
                break
        sections[topic] = entries
    newest = max(
        (
            s["last_published_at"]
            for items in sections.values()
            for s in items
            if s["last_published_at"]
        ),
        default=None,
    )
    return {
        "generated_at": now.isoformat(),
        "window_hours": since_hours,
        "newest_article_at": newest,
        "topics": sections,
    }
