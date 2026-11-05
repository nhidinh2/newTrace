"""Story exploration without losing article provenance or ordinary search behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from newstrace.api.app import create_app
from newstrace.cli import app
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.representations.registry import ensure_embeddings
from newstrace.retrieval import fulltext
from newstrace.retrieval.exact import ScoredHit
from newstrace.search import SearchFilters, SearchRequest, search
from tests.conftest import BASE_TIME

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def fixed_recency_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare ranking signals across separate API/CLI calls at the same instant."""
    monkeypatch.setattr("newstrace.retrieval.rerank.utcnow", lambda: BASE_TIME)


def seed_reports(session: Session, *, crowded: int = 12, embed: bool = True) -> list[Article]:
    """Many high-scoring reports, a second story and two unclustered reports.

    Explicit memberships isolate result presentation from clustering heuristics.
    Distinct stories deliberately share a domain; this is not source diversity.
    """
    stories = [StoryCluster(display_title=title) for title in ("Weather updates", "Housing update")]
    session.add_all(stories)
    session.flush()
    reports = []
    for i in range(crowded + 3):
        title = (
            "Weather"
            if i < crowded
            else f"Weather {['housing', 'energy', 'transport'][i - crowded]}"
        )
        article = Article(
            url=f"https://shared.example/report/{i}",
            canonical_url=f"https://shared.example/report/{i}",
            title=title,
            excerpt="Weather climate"
            if i < crowded
            else f"Weather climate {title} policy analysis",
            source_domain="shared.example",
            topic="ai_models" if i < crowded else "chips_and_compute",
            language="en",
            published_at=BASE_TIME,
            normalized_text_hash=str(i),
            normalized_title_hash=str(i),
        )
        session.add(article)
        session.flush()
        fulltext.index_article_text(session, article)
        if i <= crowded:
            session.add(
                ClusterMembership(
                    article_id=article.id,
                    story_cluster_id=stories[0 if i < crowded else 1].id,
                )
            )
        reports.append(article)
    if embed:
        ensure_embeddings(session, reports)
    session.commit()
    return reports


@pytest.mark.parametrize("method", ["full", "bm25", "tfidf"])
@pytest.mark.parametrize("pagerank", [False, True])
def test_search_fills_with_distinct_stories_after_reranking(
    session: Session, method: str, pagerank: bool
) -> None:
    reports = seed_reports(session, crowded=24)
    ranked = search(
        session,
        SearchRequest(query="weather", method=method, rerank_with_pagerank=pagerank, top_k=27),
    )
    # A well-covered event dominates ordinary article results.
    if not pagerank:
        assert len({item.story_id for item in ranked.items[:4]}) == 1
    diverse = search(
        session,
        SearchRequest(
            query="weather",
            method=method,
            rerank_with_pagerank=pagerank,
            top_k=4,
            one_per_story=True,
        ),
    )
    assert diverse.one_per_story
    # The next story lies beyond the ordinary four-result candidate budget (16).
    assert diverse.candidates_considered == 27
    assert diverse.collapsed_candidates == 23
    assert len(diverse.items) == 4
    assert len({item.story_id for item in diverse.items if item.story_id is not None}) == 2
    assert {item.article.id for item in diverse.items if item.story_id is None} == {
        article.id for article in reports[-2:]
    }
    expected = []
    seen = set()
    for item in ranked.items:
        if item.story_id is None or item.story_id not in seen:
            expected.append(item)
            seen.add(item.story_id)
    assert [item.article.id for item in diverse.items] == [item.article.id for item in expected]
    assert [item.hit.rank for item in diverse.items] == [1, 2, 3, 4]
    for result, original in zip(diverse.items, expected, strict=True):
        assert result.hit.score == original.hit.score
        assert result.hit.signals == original.hit.signals
        assert result.hit.method == original.hit.method
        assert f"rank {result.hit.rank} via" in result.explanation
        assert result.excerpt
        assert result.article.url.startswith("https://")
    ordinary = search(session, SearchRequest(query="weather", method=method, top_k=4))
    assert not ordinary.one_per_story
    assert ordinary.collapsed_candidates == 0
    assert len(ordinary.items) == 4
    assert len({item.story_id for item in ordinary.items}) == 1


def test_grouping_preserves_filters_and_duplicate_policy(session: Session) -> None:
    reports = seed_reports(session)
    reports[0].is_near_duplicate = True
    session.commit()
    filters = SearchFilters(topic="ai_models", source_domain="shared.example", language="en")
    filtered = search(
        session,
        SearchRequest(query="weather", method="bm25", filters=filters, one_per_story=True, top_k=4),
    )
    assert len(filtered.items) == 1
    assert filtered.items[0].article.id != reports[0].id
    assert filtered.candidates_considered == 11
    assert filtered.collapsed_candidates == 10
    assert any("Only 1 result remains" in note for note in filtered.notes)
    filters.include_duplicates = True
    included = search(
        session,
        SearchRequest(query="weather", method="bm25", filters=filters, one_per_story=True, top_k=4),
    )
    assert len(included.items) == 1
    assert included.candidates_considered == 12
    assert included.collapsed_candidates == 11
    # Grouping is applied after duplicate penalties, so it still prefers an original.
    assert not included.items[0].article.is_near_duplicate
    filters.entity = "absent entity"
    empty = search(session, SearchRequest(query="weather", filters=filters, one_per_story=True))
    assert empty.one_per_story
    assert empty.items == []
    assert empty.collapsed_candidates == 0


def test_multi_story_memberships_do_not_repeat_any_story(session: Session) -> None:
    reports = seed_reports(session)
    story_ids = list(session.scalars(select(StoryCluster.id).order_by(StoryCluster.id)))
    session.add(ClusterMembership(article_id=reports[0].id, story_cluster_id=story_ids[1]))
    session.commit()
    result = search(session, SearchRequest(query="weather", method="bm25", one_per_story=True))
    assert len(result.items) == 3
    assert result.items[0].article.id == reports[0].id
    assert result.items[0].story_id == min(story_ids)
    assert result.collapsed_candidates == 12
    assert all(item.story_id is None for item in result.items[1:])


def test_bounded_pool_discloses_incomplete_story_coverage(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    import newstrace.search as service

    reports = seed_reports(session, crowded=1000, embed=False)
    requested = []

    class RankedRetriever:
        def search(self, query, *, k, allowed_ids):
            requested.append(k)
            return [
                ScoredHit(article.id, 1.0 - i / 2000, i + 1, "cosine:full", {"cosine": 1.0})
                for i, article in enumerate(reports[:k])
                if article.id in allowed_ids
            ]

    monkeypatch.setattr(service, "get_index", lambda *args, **kwargs: RankedRetriever())
    queries = []

    def record(_conn, _cursor, statement, _params, _context, _many):
        if statement.lstrip().upper().startswith("SELECT") and "cluster_memberships" in statement:
            queries.append(statement)

    event.listen(session.get_bind(), "before_cursor_execute", record)
    try:
        result = search(session, SearchRequest(query="weather", top_k=100, one_per_story=True))
    finally:
        event.remove(session.get_bind(), "before_cursor_execute", record)
    assert requested == [1000]
    assert result.candidates_considered == 1000
    assert len(queries) == 1  # No per-article or per-story membership queries.
    assert len(result.items) == 1
    assert result.collapsed_candidates == 999
    assert any("limited to 1000" in note and "outside this pool" in note for note in result.notes)


def test_no_embeddings_and_no_matches_keep_requested_mode(session: Session) -> None:
    seed_reports(session, embed=False)
    unavailable = search(session, SearchRequest(query="weather", one_per_story=True))
    assert unavailable.one_per_story
    assert unavailable.items == []
    assert "no embeddings available" in unavailable.notes
    no_match = search(
        session, SearchRequest(query="nonexistent", method="bm25", one_per_story=True)
    )
    assert no_match.one_per_story
    assert no_match.items == []
    assert no_match.candidates_considered == no_match.collapsed_candidates == 0


@pytest.mark.parametrize("method", ["svd", "bm25"])
def test_grouping_retains_representation_fallbacks(
    session: Session, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    seed_reports(session)
    if method == "bm25":
        monkeypatch.setattr(fulltext, "available", lambda session: False)
    result = search(
        session,
        SearchRequest(
            query="weather",
            method=method,
            dimension=32,
            one_per_story=True,
            apply_signal_rerank=False,
            top_k=4,
        ),
    )
    assert result.one_per_story
    assert len(result.items) == 4
    assert result.collapsed_candidates == 11
    assert any("baseline" in note for note in result.notes)
    assert [item.hit.rank for item in result.items] == [1, 2, 3, 4]
    assert all("combined" not in item.hit.signals for item in result.items)


def test_api_and_cli_expose_mode_counts_and_provenance(session: Session) -> None:
    seed_reports(session)
    with TestClient(create_app()) as client:
        ordinary = client.post("/search", json={"query": "weather", "method": "bm25", "top_k": 4})
        assert ordinary.status_code == 200
        assert ordinary.json()["one_per_story"] is False
        response = client.post(
            "/search",
            json={"query": "weather", "method": "bm25", "top_k": 4, "one_per_story": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["one_per_story"] is True
        assert body["collapsed_candidates"] == 11
        assert len(body["results"]) == 4
        assert all(row["url"] and row["excerpt"] and row["explanation"] for row in body["results"])
        assert all(row["signals"] and row["ranking_method"] for row in body["results"])
        story_ids = [row["story_id"] for row in body["results"] if row["story_id"] is not None]
        assert len(story_ids) == len(set(story_ids)) == 2
    runner = CliRunner()
    output = runner.invoke(
        app, ["search", "weather", "--method", "bm25", "--top-k", "4", "--one-per-story", "--json"]
    )
    assert output.exit_code == 0, output.output
    cli = json.loads(output.output)
    assert cli["one_per_story"] is True
    assert cli["collapsed_candidates"] == 11
    assert cli["results"] == body["results"]
    table = runner.invoke(app, ["search", "weather", "--method", "bm25", "--one-per-story"])
    assert table.exit_code == 0, table.output
    assert "One result per story" in table.output


def test_search_ui_toggle_empty_state_and_story_navigation(session: Session) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    seed_reports(session)
    # Run the real page, with a navigation target that renders the selected story.
    source = (ROOT / "app/streamlit_app.py").read_text().rsplit("\nmain()", 1)[0]
    source += """
page = st.navigation([SEARCH_PAGE, STORY_PAGE], position="hidden")
page.run()
"""
    test = AppTest.from_string(source).run(timeout=30)
    assert not test.exception
    test.text_input[0].set_value("weather")
    test.number_input[0].set_value(4)
    test.selectbox[1].select("bm25").run()
    assert not test.exception
    assert len([button for button in test.button if button.label == "Story details"]) == 4
    next(box for box in test.checkbox if box.label == "One result per story").check().run()
    assert not test.exception
    assert len([button for button in test.button if button.label == "Story details"]) == 2
    assert any("11 additional reports grouped" in caption.value for caption in test.caption)
    assert any(
        "does not establish source independence" in caption.value for caption in test.caption
    )
    test.text_input[0].set_value("nonexistent").run()
    assert not test.exception
    assert any("No results" in info.value for info in test.info)
    test.text_input[0].set_value("weather").run()
    next(button for button in test.button if button.label == "Story details").click().run()
    assert not test.exception
    assert test.header[0].value == "Story"
    assert test.selectbox[0].value == test.session_state["story_id"]
    assert test.query_params["story"] == [str(test.session_state["story_id"])]
