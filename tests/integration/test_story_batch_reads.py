"""Feed content and database work stay stable as page sizes grow."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from newstrace.api.app import create_app
from newstrace.api.serializers import story_to_schema
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.stories import list_stories, load_story, load_story_bundles
from tests.conftest import BASE_TIME


@contextmanager
def select_queries(session: Session) -> Iterator[list[str]]:
    queries: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            queries.append(statement)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        yield queries
    finally:
        event.remove(engine, "before_cursor_execute", record)


def seed_stories(session: Session, count: int) -> list[int]:
    stories = [
        StoryCluster(display_title=f"Story {i}", article_count=1, distinct_domain_count=1)
        for i in range(count)
    ]
    session.add_all(stories)
    session.flush()
    for story in stories:
        url = f"https://source.example/{story.id}"
        article = Article(
            url=url,
            canonical_url=url,
            normalized_text_hash=url,
            normalized_title_hash=url,
            title=f"Report {story.id}",
            source_domain="source.example",
            published_at=BASE_TIME,
            description="Researchers publish detailed evidence about a developing news event.",
        )
        session.add(article)
        session.flush()
        session.add(
            ClusterMembership(
                article_id=article.id,
                story_cluster_id=story.id,
                similarity=0.75,
            )
        )
    session.commit()
    return [s.id for s in stories]


@pytest.mark.parametrize("count", [0, 1, 40, 200])
def test_api_feed_query_budget_and_content(session: Session, count: int) -> None:
    seed_stories(session, count)
    stories = list_stories(session, limit=200)
    expected = [
        story_to_schema(s, load_story(session, s.id)).model_dump(mode="json") for s in stories
    ]
    # A fresh request session must not rely on previously hydrated identity maps.
    session.expunge_all()
    with TestClient(create_app()) as client, select_queries(session) as queries:
        response = client.get("/stories", params={"limit": 200})
    assert response.status_code == 200
    assert response.json() == expected
    assert len(queries) == (2 if count else 1)


def test_batch_retains_empty_duplicate_and_shared_memberships(session: Session) -> None:
    ids = seed_stories(session, 3)
    empty = StoryCluster(display_title="Empty story")
    session.add(empty)
    session.flush()
    article = session.scalar(select(Article).order_by(Article.id))
    assert article is not None
    article.is_near_duplicate = True
    session.add(ClusterMembership(article_id=article.id, story_cluster_id=ids[1], similarity=0.4))
    session.commit()
    empty_id, article_id = empty.id, article.id
    session.expunge_all()
    stories = list(session.scalars(select(StoryCluster).order_by(StoryCluster.id.desc())))
    with select_queries(session) as queries:
        bundles = load_story_bundles(session, stories + stories[:1])
        assert list(bundles) == [s.id for s in stories]
        assert bundles[empty_id].articles == []
        assert bundles[ids[0]].similarity_by_article[article_id] == 0.75
        assert bundles[ids[1]].similarity_by_article[article_id] == 0.4
        assert len(bundles[ids[0]].duplicates) == 1
        assert bundles[ids[0]].independent_domains == []
        assert bundles[ids[1]].independent_domains == ["source.example"]
        assert [a.id for a in bundles[ids[1]].articles] == sorted(
            a.id for a in bundles[ids[1]].articles
        )
        for story in stories:
            story_to_schema(story, bundles[story.id])
    assert len(queries) == 1
    assert load_story(session, 999999) is None


def test_large_batch_bounds_sql_parameters(session: Session) -> None:
    seed_stories(session, 1201)
    session.expunge_all()
    stories = list(session.scalars(select(StoryCluster)))
    with select_queries(session) as queries:
        bundles = load_story_bundles(session, stories)
    assert len(queries) == 4
    assert len(bundles) == 1201
    assert all(len(bundle.articles) == 1 for bundle in bundles.values())
    with select_queries(session) as queries:
        assert load_story_bundles(session, []) == {}
    assert queries == []


def test_batch_api_keeps_filters_and_page_boundaries(session: Session) -> None:
    ids = seed_stories(session, 8)
    with TestClient(create_app()) as client, select_queries(session) as queries:
        body = client.get("/stories", params={"limit": 2, "offset": 3}).json()
    assert [s["id"] for s in body] == list(reversed(ids))[3:5]
    assert len(queries) == 2
    with TestClient(create_app()) as client, select_queries(session) as queries:
        assert client.get("/stories", params={"topic": "absent"}).json() == []
    assert len(queries) == 1


def test_streamlit_feed_uses_one_article_query(session: Session) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    seed_stories(session, 20)
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"),
        default_timeout=30,
    )
    with select_queries(session) as queries:
        app.run()
    assert not app.exception
    article_reads = [q for q in queries if "JOIN articles" in q or "FROM articles JOIN" in q]
    assert len(article_reads) == 1
    assert "Researchers publish detailed evidence" in " ".join(m.value for m in app.markdown)
