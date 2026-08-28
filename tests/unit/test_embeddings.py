"""Embedding cache keys, determinism and vector serialisation."""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from newstrace.models import Article, EmbeddingRecord
from newstrace.representations.embedder import (
    HashingEmbedder,
    cache_key,
    from_blob,
    l2_normalize,
    to_blob,
)
from newstrace.representations.registry import ensure_embeddings, load_vectors


def test_cache_key_depends_on_model_preprocessing_and_text() -> None:
    base = cache_key("hello world", "model-a", "v1")
    assert base == cache_key("hello world", "model-a", "v1")
    assert base != cache_key("hello world", "model-b", "v1")
    assert base != cache_key("hello world", "model-a", "v2")
    assert base != cache_key("hello there", "model-a", "v1")


def test_cache_key_ignores_incidental_whitespace() -> None:
    assert cache_key("hello   world", "m", "v1") == cache_key(" hello world ", "m", "v1")


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder(64)
    a = embedder.encode(["Atlas 3 agent model release"])
    b = embedder.encode(["Atlas 3 agent model release"])
    assert np.array_equal(a, b)
    assert a.shape == (1, 64)
    assert np.isclose(np.linalg.norm(a[0]), 1.0, atol=1e-6)


def test_hashing_embedder_ranks_related_text_higher() -> None:
    embedder = HashingEmbedder(256)
    vectors = embedder.encode(
        [
            "Northwind Labs releases Atlas 3 agent model",
            "Northwind Labs announced the Atlas 3 agent model",
            "Grid operator warns of data centre power demand",
        ]
    )
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_empty_text_gives_zero_vector() -> None:
    assert np.allclose(HashingEmbedder(32).encode([""]), 0.0)


def test_l2_normalize_handles_zero_rows() -> None:
    out = l2_normalize(np.zeros((2, 4), dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_blob_roundtrip() -> None:
    vector = np.arange(8, dtype=np.float32)
    assert np.array_equal(from_blob(to_blob(vector), 8), vector)
    assert np.allclose(from_blob(None, 4), 0.0)


def test_embeddings_are_cached_not_recomputed(session: Session, raw_articles: list) -> None:
    from newstrace.ingestion.pipeline import persist_articles

    persist_articles(session, raw_articles)
    session.commit()
    articles = session.query(Article).all()

    first = ensure_embeddings(session, articles)
    session.commit()
    assert len(first) == len(articles)

    before = session.query(EmbeddingRecord).count()
    second = ensure_embeddings(session, articles)
    session.commit()
    assert session.query(EmbeddingRecord).count() == before
    assert {r.id for r in second} == {r.id for r in first}


def test_load_vectors_returns_aligned_matrix(session: Session, raw_articles: list) -> None:
    from newstrace.ingestion.pipeline import persist_articles

    persist_articles(session, raw_articles)
    session.commit()
    ensure_embeddings(session, session.query(Article).all())
    session.commit()

    vectors = load_vectors(session, method="full")
    assert len(vectors) == len(raw_articles)
    assert vectors.matrix.shape[0] == len(vectors.article_ids)
    assert vectors.memory_bytes() == vectors.matrix.nbytes
    assert sorted(vectors.index_by_article) == sorted(vectors.article_ids)


def test_load_vectors_empty_is_safe(session: Session) -> None:
    vectors = load_vectors(session, method="svd", dimension=32)
    assert len(vectors) == 0
    assert vectors.matrix.shape == (0, 32)
