"""API contract: response schemas, filters, evidence and error handling."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from newstrace.models import Article, StoryCluster


def test_health_reports_system_state(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["article_count"] == 4
    assert body["duplicate_count"] == 1
    assert body["story_count"] >= 1
    assert body["llm_provider"] == "none"
    assert body["embedding_dimension"] > 0
    assert body["device"] in {"cpu", "cuda", "mps"}
    assert isinstance(body["representations"], list)


def test_topics_endpoint(api_client: TestClient) -> None:
    body = api_client.get("/topics").json()
    assert isinstance(body, list)
    keys = {t["key"] for t in body}
    assert "ai_models" in keys
    for topic in body:
        assert {"key", "display_name", "gdelt_query", "article_count"} <= set(topic)


def test_stories_list_and_detail(api_client: TestClient, populated_session: Session) -> None:
    stories = api_client.get("/stories").json()
    assert stories
    story = stories[0]
    assert story["article_count"] >= 1
    assert story["distinct_domain_count"] == len(story["domains"])

    detail = api_client.get(f"/stories/{story['id']}").json()
    assert detail["id"] == story["id"]
    assert detail["articles"]
    for article in detail["articles"]:
        assert article["url"].startswith("https://")
        assert "source_domain" in article


def test_story_detail_404(api_client: TestClient) -> None:
    assert api_client.get("/stories/999999").status_code == 404
    assert api_client.get("/stories/999999/timeline").status_code == 404
    assert api_client.get("/stories/999999/claims").status_code == 404


def test_timeline_is_ordered_and_marks_the_earliest_report(
    api_client: TestClient, populated_session: Session
) -> None:
    story_id = populated_session.query(StoryCluster.id).first()[0]
    body = api_client.get(f"/stories/{story_id}/timeline").json()
    entries = body["entries"]
    assert entries
    stamps = [e["published_at"] for e in entries if e["published_at"]]
    assert stamps == sorted(stamps)
    assert sum(1 for e in entries if e["is_first_report"]) == 1
    assert "not proof of who reported a claim first" in body["note"]


def test_claims_endpoint_labels_status_without_asserting_truth(
    api_client: TestClient, populated_session: Session
) -> None:
    story_id = populated_session.query(StoryCluster.id).first()[0]
    body = api_client.get(f"/stories/{story_id}/claims").json()
    assert "not assess whether a claim is true" in body["note"]
    for claim in body["claims"]:
        assert claim["status"] in {"repeated", "single_source", "unclear"}
        assert claim["evidence"]
        for evidence in claim["evidence"]:
            assert evidence["stance"] in {"reports", "supports", "disputes", "unclear"}
            assert evidence["url"]


def test_search_returns_evidence_and_explanations(api_client: TestClient) -> None:
    response = api_client.post("/search", json={"query": "Atlas 3 agent benchmark", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    for result in body["results"]:
        assert result["title"]
        assert result["source_domain"]
        assert result["url"].startswith("https://")
        assert result["excerpt"]
        assert result["explanation"]
        assert result["ranking_method"]
        assert isinstance(result["score"], float)
        assert "story_id" in result
    ranks = [r["score"] for r in body["results"]]
    assert ranks == sorted(ranks, reverse=True)


def test_search_over_bm25(api_client: TestClient) -> None:
    response = api_client.post(
        "/search", json={"query": "Atlas 3 agent benchmark", "method": "bm25", "top_k": 5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"]
    assert all("bm25" in r["ranking_method"] for r in body["results"])


def test_health_reports_the_index_state(api_client: TestClient) -> None:
    body = api_client.get("/health").json()
    assert body["full_text_index"] is True
    assert "entries" in body["retrieval_cache"]


def test_search_excludes_duplicates_by_default(api_client: TestClient) -> None:
    default = api_client.post("/search", json={"query": "Atlas 3", "top_k": 10}).json()
    assert all(not r["is_near_duplicate"] for r in default["results"])
    with_dupes = api_client.post(
        "/search", json={"query": "Atlas 3", "top_k": 10, "include_duplicates": True}
    ).json()
    assert len(with_dupes["results"]) >= len(default["results"])


def test_search_filters_by_topic_and_domain(api_client: TestClient) -> None:
    by_topic = api_client.post(
        "/search", json={"query": "demand", "topic": "chips_and_compute", "top_k": 10}
    ).json()
    assert all(r["source_domain"] == "gamma.example" for r in by_topic["results"])

    by_domain = api_client.post(
        "/search", json={"query": "Atlas", "source_domain": "beta.example", "top_k": 10}
    ).json()
    assert all(r["source_domain"] == "beta.example" for r in by_domain["results"])


def test_search_with_unknown_representation_falls_back_with_a_note(
    api_client: TestClient,
) -> None:
    body = api_client.post(
        "/search", json={"query": "Atlas 3", "method": "svd", "dimension": 32}
    ).json()
    assert body["notes"]
    assert any("falling back" in n for n in body["notes"])


def test_search_with_pagerank_rerank(api_client: TestClient) -> None:
    body = api_client.post(
        "/search",
        json={"query": "Atlas 3 agent benchmark", "rerank_with_pagerank": True, "top_k": 5},
    ).json()
    assert body["results"]
    assert any("PageRank" in note for note in body["notes"])
    assert all("pagerank" in r["signals"] for r in body["results"])


def test_search_validates_input(api_client: TestClient) -> None:
    assert api_client.post("/search", json={"query": ""}).status_code == 422
    assert api_client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert api_client.post("/search", json={"query": "x", "method": "bogus"}).status_code == 422


def test_tfidf_search_method(api_client: TestClient) -> None:
    body = api_client.post(
        "/search", json={"query": "gigawatts demand", "method": "tfidf", "top_k": 5}
    ).json()
    assert body["results"]
    assert body["results"][0]["ranking_method"].startswith("tfidf")


def test_summary_endpoint_is_fully_cited(
    api_client: TestClient, populated_session: Session
) -> None:
    story_id = populated_session.query(StoryCluster.id).first()[0]
    body = api_client.post(f"/summaries/story/{story_id}", json={"method": "extractive"}).json()
    summary = body["summary"]
    assert body["citation_coverage"] == 1.0
    known = {e["evidence_id"] for e in summary["evidence"]}
    assert summary["statements"]
    for statement in summary["statements"]:
        assert statement["evidence_ids"]
        assert set(statement["evidence_ids"]) <= known
    assert any("not evidence of truth" in n for n in summary["notes"])


def test_summary_llm_method_falls_back_without_a_key(
    api_client: TestClient, populated_session: Session
) -> None:
    story_id = populated_session.query(StoryCluster.id).first()[0]
    body = api_client.post(f"/summaries/story/{story_id}", json={"method": "llm"}).json()
    assert body["summary"]["method"] == "extractive"
    assert any("unavailable" in n for n in body["summary"]["notes"])


def test_summary_404(api_client: TestClient) -> None:
    assert api_client.post("/summaries/story/999999", json={}).status_code == 404


def test_ingestion_run_endpoints(api_client: TestClient) -> None:
    created = api_client.post("/ingestion/runs", json={"source": "fixtures", "index": False})
    assert created.status_code == 201
    run = created.json()
    assert run["status"].startswith("completed")
    assert run["fetched_count"] > 0
    assert run["random_seed"] == 549

    fetched = api_client.get(f"/ingestion/runs/{run['id']}").json()
    assert fetched["id"] == run["id"]
    assert api_client.get("/ingestion/runs").json()
    assert api_client.get("/ingestion/runs/999999").status_code == 404


def test_openapi_documents_every_required_route(api_client: TestClient) -> None:
    paths = api_client.get("/openapi.json").json()["paths"]
    required = {
        ("/health", "get"),
        ("/topics", "get"),
        ("/stories", "get"),
        ("/stories/{story_id}", "get"),
        ("/stories/{story_id}/timeline", "get"),
        ("/stories/{story_id}/claims", "get"),
        ("/search", "post"),
        ("/summaries/story/{story_id}", "post"),
        ("/ingestion/runs", "post"),
        ("/ingestion/runs/{run_id}", "get"),
        ("/experiments", "post"),
        ("/experiments/{experiment_id}", "get"),
    }
    for path, method in required:
        assert path in paths, f"missing route {path}"
        assert method in paths[path], f"missing {method.upper()} {path}"


def test_experiment_endpoints(api_client: TestClient) -> None:
    created = api_client.post(
        "/experiments",
        json={"methods": ["full", "gaussian_rp"], "dimensions": [8], "include_pagerank": False},
    )
    assert created.status_code == 201
    body = created.json()
    assert body["status"] in {"completed", "failed"}
    assert api_client.get(f"/experiments/{body['id']}").json()["id"] == body["id"]
    assert api_client.get("/experiments/999999").status_code == 404


def test_articles_are_never_hidden_behind_prose(
    api_client: TestClient, populated_session: Session
) -> None:
    """Every surfaced article must expose its source URL."""
    story_id = populated_session.query(StoryCluster.id).first()[0]
    detail = api_client.get(f"/stories/{story_id}").json()
    urls = {a["url"] for a in detail["articles"]}
    stored = {
        a.url
        for a in populated_session.query(Article).all()
        if any(m.story_cluster_id == story_id for m in a.memberships)
    }
    assert urls == stored
