"""A tied candidate cutoff must not discard lower-ID reports before reranking."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from newstrace.api.app import create_app
from newstrace.db import get_db
from newstrace.models import Article
from newstrace.representations.registry import ensure_embeddings
from newstrace.retrieval.index import cache_stats


def _article(article_id: int) -> Article:
    return Article(
        id=article_id,
        url=f"https://source.example/{article_id}",
        canonical_url=f"https://source.example/{article_id}",
        title="Shared news report",
        excerpt="Researchers publish a shared news report on retrieval.",
        source_domain="even.example" if article_id % 2 == 0 else "odd.example",
        language="english",
        normalized_text_hash=f"text-{article_id}",
        normalized_title_hash=f"title-{article_id}",
        is_near_duplicate=article_id in (100, 101),
    )


@pytest.mark.parametrize("method", ["full", "tfidf"])
def test_search_api_cutoff_ties_survive_filtering_reranking_and_cache_growth(
    session: Session, method: str
) -> None:
    articles = [_article(i) for i in range(100, 148)]
    session.add_all(articles)
    session.flush()
    # Deliberately store vectors in an order different from article identity.
    ensure_embeddings(session, articles[::-1])
    session.commit()
    app = create_app()
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        for source, expected in (
            (None, [102, 103, 104, 105, 106]),
            ("even.example", [102, 104, 106, 108, 110]),
            ("odd.example", [103, 105, 107, 109, 111]),
        ):
            payload = {"query": "shared news report", "method": method, "top_k": 5}
            if source:
                payload["source_domain"] = source
            # Cold and warm requests both pass through the real signal reranker.
            for _ in range(2):
                response = client.post("/search", json=payload)
                assert response.status_code == 200
                body = response.json()
                assert [r["article_id"] for r in body["results"]] == expected
                assert body["candidates_considered"] == 20
                assert all("signals" in r["ranking_method"] for r in body["results"])
                assert all(not r["is_near_duplicate"] for r in body["results"])

        # A later embedding record with a lower article ID must win ties even
        # when the process cache extends its existing rows instead of rebuilding.
        new_article = _article(50)
        session.add(new_article)
        session.flush()
        ensure_embeddings(session, [new_article])
        session.commit()
        extensions = cache_stats()["extends"]
        response = client.post(
            "/search", json={"query": "shared news report", "method": method, "top_k": 5}
        )
        assert response.status_code == 200
        assert [r["article_id"] for r in response.json()["results"]] == [50, 102, 103, 104, 105]
        if method == "full":
            assert cache_stats()["extends"] == extensions + 1
        response = client.post(
            "/search",
            json={
                "query": "shared news report",
                "method": method,
                "source_domain": "absent.example",
            },
        )
        assert response.status_code == 200
        assert response.json()["results"] == []
