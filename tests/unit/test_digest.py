"""Digest export: recent stories with independent-source counts and links."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from newstrace.digest import build_digest
from newstrace.models import Article, ClusterMembership, StoryCluster
from tests.conftest import BASE_TIME


def _add_story(
    session: Session,
    story_id: int,
    *,
    topic: str,
    domains: list[str],
    duplicates: int = 0,
    language: str = "unknown",
    age_hours: float = 2,
) -> None:
    published = BASE_TIME - timedelta(hours=age_hours)
    story = StoryCluster(
        id=story_id,
        display_title=f"Story {story_id}",
        topic=topic,
        article_count=len(domains) + duplicates,
        distinct_domain_count=len(set(domains)),
        first_published_at=published,
        last_published_at=published + timedelta(minutes=len(domains)),
    )
    session.add(story)
    rows = [(d, False) for d in domains] + [("wire.example", True)] * duplicates
    for i, (domain, is_dup) in enumerate(rows):
        article = Article(
            url=f"https://{domain}/s{story_id}/{i}",
            canonical_url=f"https://{domain}/s{story_id}/{i}",
            title=f"Headline {story_id} from {domain}",
            description="What happened.",
            source_domain=domain,
            language=language,
            topic=topic,
            published_at=published + timedelta(minutes=i),
            normalized_text_hash=f"t{story_id}-{i}",
            normalized_title_hash=f"h{story_id}-{i}",
            is_near_duplicate=is_dup,
        )
        session.add(article)
        session.flush()
        session.add(
            ClusterMembership(article_id=article.id, story_cluster_id=story_id, similarity=0.9)
        )
    session.commit()


def test_digest_counts_independent_sources_and_links_earliest_report(session: Session) -> None:
    _add_story(
        session, 1, topic="ai_safety", domains=["a.example", "b.example", "a.example"], duplicates=4
    )
    result = build_digest(session, topics=["ai_safety"], now=BASE_TIME)

    [story] = result["topics"]["ai_safety"]
    assert story["independent_sources"] == 2  # syndicated copies do not count
    assert story["duplicates"] == 4
    assert story["single_source"] is False
    assert story["earliest_report"]["url"] == "https://a.example/s1/0"
    assert [s["domain"] for s in story["sources"]] == ["a.example", "b.example"]
    assert result["newest_article_at"] is not None


def test_digest_ranks_by_sources_and_filters_window_language_and_minimum(
    session: Session,
) -> None:
    _add_story(session, 1, topic="ai_models", domains=["a.example"])
    _add_story(session, 2, topic="ai_models", domains=["a.example", "b.example", "c.example"])
    _add_story(session, 3, topic="ai_models", domains=["x.example", "y.example"], age_hours=100)
    _add_story(
        session, 4, topic="ai_models", domains=["p.example", "q.example"], language="chinese"
    )

    result = build_digest(session, topics=["ai_models"], since_hours=48, now=BASE_TIME)
    assert [s["id"] for s in result["topics"]["ai_models"]] == [2, 1]

    strict = build_digest(session, topics=["ai_models"], min_sources=2, now=BASE_TIME)
    assert [s["id"] for s in strict["topics"]["ai_models"]] == [2]


def test_digest_reports_nothing_recent(session: Session) -> None:
    result = build_digest(session, topics=["ai_safety"], now=BASE_TIME)
    assert result["topics"] == {"ai_safety": []}
    assert result["newest_article_at"] is None
