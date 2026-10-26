"""Story read-model and timeline behaviour."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from newstrace.models import Article, StoryCluster
from newstrace.stories import (
    DailyStoryActivity,
    StoryActivity,
    StoryArticles,
    build_story_activity,
    build_timeline,
)
from tests.conftest import BASE_TIME


def _article(article_id: int, *, published: bool) -> Article:
    return Article(
        id=article_id,
        url=f"https://source{article_id}.example/report",
        canonical_url=f"https://source{article_id}.example/report",
        title=f"Report {article_id}",
        description="This report contains enough text to produce a timeline sentence.",
        source_domain=f"source{article_id}.example",
        published_at=BASE_TIME if published else None,
        is_near_duplicate=False,
    )


def test_timeline_places_articles_without_publication_dates_last() -> None:
    """A missing feed timestamp must not crash an otherwise dated story."""
    dated = _article(1, published=True)
    undated = _article(2, published=False)
    bundle = StoryArticles(
        story=StoryCluster(id=1, display_title="A developing story"),
        articles=[undated, dated],
    )

    assert bundle.sorted_by_time(include_duplicates=True) == [dated, undated]

    entries = build_timeline(bundle)
    assert [entry.article.id for entry in entries] == [dated.id, undated.id]
    assert entries[0].is_first_report is True
    assert entries[1].published_at is None


def test_story_activity_counts_articles_by_utc_day_and_separates_copies() -> None:
    articles = [_article(i, published=True) for i in range(1, 6)]
    # These two local dates are the same UTC day; SQLite's naive times are UTC.
    articles[0].published_at = datetime(2026, 8, 18, 23, 30)
    articles[1].published_at = datetime(2026, 8, 19, 1, 30, tzinfo=timezone(timedelta(hours=2)))
    articles[1].source_domain = articles[0].source_domain
    articles[2].published_at = datetime(2026, 8, 18, 22, 30, tzinfo=timezone(timedelta(hours=-2)))
    articles[3].published_at = datetime(2026, 8, 20, 6, tzinfo=UTC)
    articles[3].is_near_duplicate = True
    articles[4].published_at = articles[0].published_at
    articles[4].is_near_duplicate = True
    bundle = StoryArticles(story=StoryCluster(id=1), articles=list(reversed(articles)))

    activity = build_story_activity(bundle)

    assert activity == StoryActivity(
        days=[
            DailyStoryActivity(date(2026, 8, 18), report_count=2, duplicate_count=1),
            DailyStoryActivity(date(2026, 8, 19), report_count=1),
            DailyStoryActivity(date(2026, 8, 20), duplicate_count=1),
        ]
    )
    assert articles[1].published_at.utcoffset() == timedelta(hours=2)


@pytest.mark.parametrize("include_dated", [False, True])
def test_story_activity_keeps_undated_articles_out_of_daily_counts(include_dated: bool) -> None:
    undated = _article(1, published=False)
    undated_copy = _article(2, published=False)
    undated_copy.is_near_duplicate = True
    articles = [undated, undated_copy]
    if include_dated:
        articles.append(_article(3, published=True))
    bundle = StoryArticles(story=StoryCluster(id=1), articles=articles)

    activity = build_story_activity(bundle)

    assert activity == StoryActivity(
        days=[DailyStoryActivity(BASE_TIME.date(), report_count=1)] if include_dated else [],
        undated_report_count=1,
        undated_duplicate_count=1,
    )
    assert sum(d.report_count + d.duplicate_count for d in activity.days) + 2 == len(articles)


def test_story_activity_for_an_empty_story() -> None:
    assert build_story_activity(StoryArticles(story=StoryCluster(id=1))) == StoryActivity()
