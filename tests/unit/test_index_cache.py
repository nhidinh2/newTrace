"""The cached retrieval index must never serve a stale corpus."""

from __future__ import annotations

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article
from newstrace.pipeline import index_articles
from newstrace.representations.registry import load_vectors, store_vectors
from newstrace.retrieval import index as index_module
from newstrace.search import SearchRequest, search
from tests.conftest import BASE_TIME


def test_index_is_built_once_and_reused(populated_session: Session) -> None:
    index_module.invalidate()
    first = index_module.get_index(populated_session, method="full")
    second = index_module.get_index(populated_session, method="full")
    assert first is not None
    assert first is second
    assert index_module.cache_stats()["hits"] >= 1


def test_index_grows_when_articles_are_ingested(populated_session: Session) -> None:
    index_module.invalidate()
    before = index_module.get_index(populated_session, method="full")
    assert before is not None
    original = len(before.article_ids)

    result = persist_articles(
        populated_session,
        [
            RawArticle(
                url="https://epsilon.example/new-story",
                title="Regulator opens consultation on model evaluations",
                description="The regulator asked for comment on evaluation standards.",
                source_domain="epsilon.example",
                language="English",
                published_at=BASE_TIME,
                topic="ai_models",
                source_adapter="test",
            )
        ],
    )
    populated_session.commit()
    index_articles(populated_session, result.article_ids)

    after = index_module.get_index(populated_session, method="full")
    assert after is not None
    assert len(after.article_ids) == original + 1
    # Grown in place rather than rebuilt from every blob.
    assert after is before
    assert after.matrix.shape[0] == len(after.article_ids)


def test_extended_index_matches_a_full_rebuild(populated_session: Session) -> None:
    index_module.invalidate()
    index_module.get_index(populated_session, method="full")
    result = persist_articles(
        populated_session,
        [
            RawArticle(
                url="https://zeta.example/another",
                title="Chip maker reports record lithography yields",
                description="The manufacturer reported a record yield on its newest line.",
                source_domain="zeta.example",
                language="English",
                published_at=BASE_TIME,
                source_adapter="test",
            )
        ],
    )
    populated_session.commit()
    index_articles(populated_session, result.article_ids)
    grown = index_module.get_index(populated_session, method="full")
    assert grown is not None

    index_module.invalidate()
    rebuilt = index_module.get_index(populated_session, method="full")
    assert rebuilt is not None
    assert grown.article_ids == rebuilt.article_ids
    assert np.allclose(grown.matrix, rebuilt.matrix, atol=1e-6)


def test_rewritten_vectors_invalidate_the_cache(populated_session: Session) -> None:
    """An in-place rewrite changes neither the row count nor the highest id."""
    vectors = load_vectors(populated_session, method="full")
    store_vectors(
        populated_session,
        vectors.article_ids,
        np.zeros_like(vectors.matrix),
        model_name="test",
        method="svd",
        fit_version="test-fit",
    )
    populated_session.commit()
    first = index_module.get_index(populated_session, method="svd")
    assert first is not None

    store_vectors(
        populated_session,
        vectors.article_ids,
        np.ones_like(vectors.matrix),
        model_name="test",
        method="svd",
        fit_version="test-fit",
    )
    populated_session.commit()
    second = index_module.get_index(populated_session, method="svd")
    assert second is not None
    assert second is not first
    assert not np.allclose(first.matrix, second.matrix)


def test_missing_representation_returns_none(populated_session: Session) -> None:
    assert index_module.get_index(populated_session, method="does_not_exist") is None


def test_deleted_embeddings_shrink_the_index(populated_session: Session) -> None:
    index_module.invalidate()
    before = index_module.get_index(populated_session, method="full")
    assert before is not None
    article = populated_session.execute(select(Article).limit(1)).scalars().one()
    populated_session.delete(article)
    populated_session.commit()

    after = index_module.get_index(populated_session, method="full")
    assert after is not None
    assert len(after.article_ids) == len(before.article_ids) - 1


def test_cache_respects_the_configured_size(
    populated_session: Session, monkeypatch: object
) -> None:
    from newstrace.config import get_settings

    settings = get_settings().model_copy(update={"retrieval_cache_entries": 1})
    index_module.invalidate()
    vectors = load_vectors(populated_session, method="full")
    for name in ("svd", "gaussian_rp"):
        store_vectors(
            populated_session,
            vectors.article_ids,
            vectors.matrix[:, :4],
            model_name="test",
            method=name,
            fit_version=f"{name}-fit",
        )
    populated_session.commit()
    index_module.get_index(populated_session, method="svd", settings=settings)
    index_module.get_index(populated_session, method="gaussian_rp", settings=settings)
    assert index_module.cache_stats()["entries"] == 1


def test_search_uses_the_cache_and_still_filters(populated_session: Session) -> None:
    index_module.invalidate()
    response = search(populated_session, SearchRequest(query="agent benchmark", top_k=3))
    assert response.items
    assert index_module.cache_stats()["entries"] >= 1
    repeated = search(populated_session, SearchRequest(query="agent benchmark", top_k=3))
    assert [i.article.id for i in repeated.items] == [i.article.id for i in response.items]
