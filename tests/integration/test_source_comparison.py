"""Comparison provenance, sparse cells, API contracts and interactive selection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlalchemy.orm import Session

from newstrace.api.app import create_app
from newstrace.claims.comparison import build_source_comparison
from newstrace.models import Article, Claim, ClaimEvidence, ClusterMembership, StoryCluster
from newstrace.stories import load_story


@pytest.fixture
def comparison_story(session: Session) -> int:
    story = StoryCluster(display_title="Synthetic source comparison", article_count=5)
    other = StoryCluster(display_title="Unrelated story")
    session.add_all([story, other])
    session.flush()
    articles = []
    for i, domain in enumerate(
        ["alpha.example", "alpha.example", "beta.example", "copy.example", "silent.example"]
    ):
        article = Article(
            url=f"https://{domain}/{i}",
            canonical_url=f"https://{domain}/{i}",
            title=f"Synthetic report {i}",
            source_domain=domain,
            normalized_text_hash=f"text-{i}",
            normalized_title_hash=f"title-{i}",
            is_near_duplicate=domain == "copy.example",
            published_at=None if i == 2 else datetime(2026, 11, 2, 10, tzinfo=UTC),
        )
        session.add(article)
        session.flush()
        articles.append(article)
        session.add(ClusterMembership(article_id=article.id, story_cluster_id=story.id))
    claims = []
    for i in range(3):
        claim = Claim(
            story_cluster_id=story.id,
            normalized_claim=f"Synthetic claim {i}",
            claim_hash=f"claim-{i}",
            confidence=0.8 if i < 2 else 0.1,
        )
        session.add(claim)
        session.flush()
        claims.append(claim)
    # Multiple excerpts and multiple articles from alpha must still count as one domain.
    for i, (claim, article, stance) in enumerate(
        [
            (claims[0], articles[0], "reports"),
            (claims[0], articles[0], "supports"),
            (claims[0], articles[1], "reports"),
            (claims[0], articles[2], "disputes"),
            (claims[0], articles[3], "unclear"),
            (claims[1], articles[3], "disputes"),
        ]
    ):
        session.add(
            ClaimEvidence(
                claim_id=claim.id,
                article_id=article.id,
                excerpt=f"Exact synthetic passage {i}",
                excerpt_start=i,
                stance=stance,
                confidence=0.7,
            )
        )
    session.add(
        Claim(
            story_cluster_id=other.id,
            normalized_claim="Unrelated claim",
            claim_hash="other",
            confidence=1.0,
        )
    )
    session.commit()
    return story.id


def test_comparison_preserves_provenance_and_distinct_counts(session, comparison_story):
    with Session(session.get_bind()) as cold:
        bundle = load_story(cold, comparison_story)
        view = build_source_comparison(cold, bundle)
    assert [c.text for c in view.claims] == [f"Synthetic claim {i}" for i in range(3)]
    assert [s.source_domain for s in view.sources] == [
        "alpha.example",
        "beta.example",
        "copy.example",
        "silent.example",
    ]
    assert view.sources[0].article_count == 2
    assert view.sources[2].duplicate_article_count == 1
    first = view.claims[0]
    assert first.independent_domain_count == 2
    alpha, beta, copy = first.cells
    assert alpha.independent_article_count == 2
    assert len(alpha.evidence) == 3
    assert alpha.stances == ["reports", "supports"]
    assert beta.stances == ["disputes"]
    assert beta.evidence[0].published_at is None
    assert copy.stances == []
    assert copy.independent_article_count == 0
    assert copy.duplicate_article_count == 1
    assert copy.evidence[0].stance == "unclear"
    assert copy.evidence[0].is_near_duplicate
    assert view.claims[1].independent_domain_count == 0
    assert view.claims[2].cells == []
    assert [e.excerpt for cell in first.cells for e in cell.evidence] == [
        f"Exact synthetic passage {i}" for i in range(5)
    ]
    assert all(e.url.startswith("https://") for cell in first.cells for e in cell.evidence)


def test_comparison_api_contract(session, comparison_story):
    with TestClient(create_app()) as client:
        response = client.get(f"/stories/{comparison_story}/comparison")
        assert response.status_code == 200
        body = response.json()
        assert len(body["claims"]) == 3
        assert "does not mean a source denies" in body["note"]
        assert "proxy for independence" in body["note"]
        assert client.get("/stories/999999/comparison").status_code == 404
        schema = client.get("/openapi.json").json()
        assert "/stories/{story_id}/comparison" in schema["paths"]


def test_comparison_keeps_historical_evidence_and_unknown_domains(session, comparison_story):
    bundle = load_story(session, comparison_story)
    article = next(a for a in bundle.articles if a.source_domain == "beta.example")
    session.query(ClusterMembership).filter_by(article_id=article.id).delete()
    article.source_domain = ""
    session.commit()
    bundle = load_story(session, comparison_story)
    view = build_source_comparison(session, bundle)
    assert view.sources[0].source_domain == ""
    assert view.sources[0].article_count == 1
    assert view.claims[0].independent_domain_count == 1
    historical = view.claims[0].cells[0].evidence[0]
    assert historical.article_id == article.id
    assert historical.url == "https://beta.example/2"


@pytest.mark.parametrize("count", [0, 1, 40, 1201])
def test_comparison_query_budget(session, count):
    story = StoryCluster(display_title="Query budget")
    session.add(story)
    session.flush()
    article = Article(
        url="https://budget.example/report",
        canonical_url="https://budget.example/report",
        source_domain="budget.example",
        normalized_text_hash="budget",
        normalized_title_hash="budget",
    )
    session.add(article)
    session.flush()
    session.add(ClusterMembership(article_id=article.id, story_cluster_id=story.id))
    for i in range(count):
        claim = Claim(
            story_cluster_id=story.id,
            normalized_claim=f"Claim {i}",
            claim_hash=str(i),
            confidence=0.5,
        )
        session.add(claim)
        session.flush()
        session.add(
            ClaimEvidence(claim_id=claim.id, article_id=article.id, excerpt=f"Evidence {i}")
        )
    session.commit()
    selects = []

    def record(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            selects.append(statement)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        with Session(engine) as cold:
            bundle = load_story(cold, story.id)
            view = build_source_comparison(cold, bundle)
            view.model_dump(mode="json")
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert len(selects) == 3
    assert len(view.claims) == count
    assert [s.source_domain for s in view.sources] == ["budget.example"]
    assert all(len(c.cells[0].evidence) == 1 for c in view.claims)


def test_comparison_ui_selection_filters_and_empty_states(session, comparison_story):
    app_test = pytest.importorskip("streamlit.testing.v1").AppTest
    source = (Path(__file__).parents[2] / "app/streamlit_app.py").read_text()
    app = app_test.from_string(source.rsplit("\nmain()", 1)[0] + "\npage_story_detail()\n")
    app.query_params["story"] = str(comparison_story)
    app.run(timeout=30)
    assert not app.exception
    assert "Compare sources" in [tab.label for tab in app.tabs]
    matrix = app.dataframe[0].value
    assert len(matrix) == 3
    assert matrix.iloc[1]["Source: copy.example"] == "Copies only · 1 article"
    assert "Exact synthetic passage 3" in [m.value for m in app.markdown]
    assert any("https://beta.example/2" in m.value for m in app.markdown)
    app.multiselect[0].set_value(["silent.example"]).run()
    assert not app.exception
    assert any(i.value == "No extracted evidence" for i in app.info)
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    app.multiselect[0].set_value(["copy.example"]).run()
    assert any("near-duplicate, excluded" in c.value for c in app.caption)
    app.multiselect[0].set_value([]).run()
    assert any("Choose at least one source" in i.value for i in app.info)
    # Filter changes after selecting an evidence-free claim must not leave a stale widget ID.
    app.multiselect[0].set_value(["alpha.example"]).run()
    app.checkbox[0].uncheck().run()
    picker = next(s for s in app.selectbox if s.label == "Inspect claim evidence")
    picker.select(picker.options[-1]).run()
    app.checkbox[0].check().run()
    assert not app.exception
    # A story without any flagged evidence needs an explicit empty filter state.
    session.query(ClaimEvidence).update({ClaimEvidence.stance: "reports"})
    session.commit()
    app.run()
    assert not app.exception
    assert any("No claims match this filter" in i.value for i in app.info)


@pytest.mark.parametrize("with_claim", [False, True])
def test_comparison_ui_no_claims_or_sources(session, with_claim):
    app_test = pytest.importorskip("streamlit.testing.v1").AppTest
    story = StoryCluster(display_title="No claims")
    session.add(story)
    session.flush()
    if with_claim:
        session.add(
            Claim(story_cluster_id=story.id, normalized_claim="No evidence", claim_hash="empty")
        )
    session.commit()
    source = (Path(__file__).parents[2] / "app/streamlit_app.py").read_text()
    app = app_test.from_string(source.rsplit("\nmain()", 1)[0] + "\npage_story_detail()\n")
    app.query_params["story"] = str(story.id)
    app.run(timeout=30)
    assert not app.exception
    message = (
        "No source evidence is available" if with_claim else "Source articles remain available"
    )
    assert any(message in i.value for i in app.info)
