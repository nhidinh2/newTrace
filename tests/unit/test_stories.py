"""Story read-model and timeline behaviour."""

from __future__ import annotations

from newstrace.models import Article, StoryCluster
from newstrace.stories import StoryArticles, build_timeline
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
