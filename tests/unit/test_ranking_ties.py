"""Cutoff ties must be selected before truncating, across retrieval backends."""

from __future__ import annotations

import numpy as np
import pytest

from newstrace.retrieval.ann import AnnConfig, IvfIndex
from newstrace.retrieval.exact import DenseRetriever, TfidfRetriever, top_k


def test_top_k_resolves_the_entire_cutoff_tie() -> None:
    scores = np.ones(32, dtype=np.float32)
    assert top_k(scores, 5).tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("k", [0, -1, -40])
def test_top_k_nonpositive_budget_is_empty(k: int) -> None:
    assert top_k(np.ones(32, dtype=np.float32), k).tolist() == []


def test_top_k_matches_full_sort_with_and_without_identity_keys() -> None:
    rng = np.random.default_rng(549)
    for size in (0, 1, 2, 32, 101):
        scores = rng.integers(-3, 4, size=size).astype(np.float32)
        rows = np.arange(size, dtype=np.int64)
        # Repeated keys exercise the final row-index tiebreaker too.
        identities = rng.integers(0, 20, size=size, dtype=np.int64)
        for k in (0, 1, 5, size, size + 4):
            for keys in (None, identities):
                expected = (
                    np.lexsort((rows, -scores))
                    if keys is None
                    else np.lexsort((rows, keys, -scores))
                )
                actual = top_k(scores, k) if keys is None else top_k(scores, k, tie_breaker=keys)
                assert actual.dtype == np.int64
                assert actual.tolist() == expected[:k].tolist()


def test_top_k_preserves_numpy_order_for_nonfinite_scores() -> None:
    scores = np.array([np.nan, -np.inf, 1, np.inf, 1, np.nan], dtype=np.float32)
    keys = np.array([6, 5, 4, 3, 2, 1], dtype=np.int64)
    for k in range(1, 8):
        assert (
            top_k(scores, k, tie_breaker=keys).tolist() == np.lexsort((keys, -scores))[:k].tolist()
        )


@pytest.mark.parametrize("backend", ["dense", "tfidf"])
@pytest.mark.parametrize("permutation", ["forward", "reverse", "shuffle"])
def test_exact_ties_follow_article_identity_after_filtering(backend: str, permutation: str) -> None:
    ids = np.arange(100, 132, dtype=np.int64)
    if permutation == "reverse":
        ids = ids[::-1]
    elif permutation == "shuffle":
        ids = np.random.default_rng(549).permutation(ids)
    if backend == "dense":
        retriever = DenseRetriever(ids.tolist(), np.tile([1, 0], (32, 1)).astype(np.float32))
        query = np.array([1, 0], dtype=np.float32)
    else:
        retriever = TfidfRetriever(ids.tolist(), ["shared news report"] * 32)
        query = "shared news"
    for allowed in (None, set(range(107, 130)), set(), {115}):
        expected = sorted(set(ids.tolist()) if allowed is None else allowed)
        for k in (0, -1, 1, 5, 50):
            hits = retriever.search(query, k=k, allowed_ids=allowed)
            assert [hit.article_id for hit in hits] == expected[: max(0, k)]
            assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_dense_ties_remain_stable_after_extending_the_index() -> None:
    retriever = DenseRetriever([50, 40], np.ones((2, 2), dtype=np.float32))
    retriever.extend([30, 20, 10], np.ones((3, 2), dtype=np.float32))
    assert [h.article_id for h in retriever.search(np.ones(2, dtype=np.float32), k=3)] == [
        10,
        20,
        30,
    ]


@pytest.mark.parametrize(("subvectors", "lists"), [(0, 1), (1, 1), (0, 4), (1, 2)])
def test_ivf_cutoff_ties_follow_ids_across_cells_and_filters(subvectors: int, lists: int) -> None:
    ids = np.random.default_rng(549).permutation(np.arange(100, 132)).tolist()
    matrix = np.eye(4, dtype=np.float32)[np.arange(32) % 4]
    index = IvfIndex(
        AnnConfig(n_lists=lists, n_probe=lists, n_subvectors=subvectors, n_bits=2)
    ).fit(ids, matrix)
    for query in (np.zeros(4, dtype=np.float32), matrix[0]):
        # Exhaustive backend output supplies actual distances, including PQ rounding.
        complete = index.search(query, k=len(ids))
        reference = sorted(complete, key=lambda hit: (-hit.score, hit.article_id))
        for allowed in (None, set(range(107, 130)), set(), {115}):
            expected = [
                hit.article_id for hit in reference if allowed is None or hit.article_id in allowed
            ]
            for k in (0, -1, 1, 3, 50):
                hits = index.search(query, k=k, allowed_ids=allowed)
                assert [hit.article_id for hit in hits] == expected[: max(0, k)]
                assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
        if subvectors == 0:
            exact = DenseRetriever(ids, matrix).search(query, k=3)
            assert [h.article_id for h in index.search(query, k=3)] == [h.article_id for h in exact]
