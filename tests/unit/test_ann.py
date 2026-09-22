"""IVF and IVF-PQ: an approximate index still has to be approximately right."""

from __future__ import annotations

import numpy as np
import pytest

from newstrace.retrieval.ann import AnnConfig, IvfIndex
from newstrace.retrieval.exact import DenseRetriever


def _clustered_corpus(
    n_clusters: int = 8, per_cluster: int = 40, dimension: int = 32, seed: int = 549
) -> tuple[list[int], np.ndarray]:
    """Vectors with real cluster structure, which is what IVF assumes."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dimension)).astype(np.float32)
    rows = []
    for centre in centres:
        rows.append(centre + 0.12 * rng.normal(size=(per_cluster, dimension)).astype(np.float32))
    matrix = np.vstack(rows).astype(np.float32)
    return list(range(1, matrix.shape[0] + 1)), matrix


def _recall(exact: list[int], approximate: list[int]) -> float:
    reference = set(exact)
    return len(reference & set(approximate)) / max(1, len(reference))


def test_probing_every_cell_recovers_exact_ranking() -> None:
    """IVF-Flat with n_probe = n_lists is exhaustive, so it cannot be wrong."""
    ids, matrix = _clustered_corpus()
    index = IvfIndex(AnnConfig(n_lists=8, n_probe=8, n_subvectors=0)).fit(ids, matrix)
    exact = DenseRetriever(ids, matrix)

    for row in range(0, matrix.shape[0], 37):
        query = matrix[row]
        expected = [h.article_id for h in exact.search(query, k=5)]
        got = [h.article_id for h in index.search(query, k=5, n_probe=8)]
        assert got == expected


def test_recall_rises_with_n_probe() -> None:
    ids, matrix = _clustered_corpus()
    index = IvfIndex(AnnConfig(n_lists=16, n_probe=1)).fit(ids, matrix)
    exact = DenseRetriever(ids, matrix)
    queries = [matrix[row] for row in range(0, matrix.shape[0], 23)]

    scores = []
    for probe in (1, 4, 16):
        recalls = [
            _recall(
                [h.article_id for h in exact.search(q, k=10)],
                [h.article_id for h in index.search(q, k=10, n_probe=probe)],
            )
            for q in queries
        ]
        scores.append(float(np.mean(recalls)))
    assert scores[0] <= scores[1] <= scores[2]
    assert scores[-1] > 0.95


def test_pq_trades_memory_for_recall() -> None:
    ids, matrix = _clustered_corpus(dimension=32)
    flat = IvfIndex(AnnConfig(n_lists=8, n_probe=8, n_subvectors=0)).fit(ids, matrix)
    quantised = IvfIndex(AnnConfig(n_lists=8, n_probe=8, n_subvectors=8, n_bits=4)).fit(ids, matrix)

    assert quantised.memory_bytes() < flat.memory_bytes()
    exact = DenseRetriever(ids, matrix)
    queries = [matrix[row] for row in range(0, matrix.shape[0], 29)]
    recall = float(
        np.mean(
            [
                _recall(
                    [h.article_id for h in exact.search(q, k=10)],
                    [h.article_id for h in quantised.search(q, k=10)],
                )
                for q in queries
            ]
        )
    )
    # Quantisation costs recall; it must not destroy it.
    assert recall > 0.5


def test_scores_are_cosine_like() -> None:
    ids, matrix = _clustered_corpus()
    index = IvfIndex(AnnConfig(n_lists=8, n_probe=8)).fit(ids, matrix)
    hits = index.search(matrix[0], k=5)
    assert hits
    assert hits[0].article_id == ids[0]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert all(-1.01 <= h.score <= 1.01 for h in hits)
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]


def test_allowed_ids_filter_is_applied() -> None:
    ids, matrix = _clustered_corpus()
    index = IvfIndex(AnnConfig(n_lists=8, n_probe=8)).fit(ids, matrix)
    allowed = {ids[5], ids[6]}
    hits = index.search(matrix[0], k=10, allowed_ids=allowed)
    assert {h.article_id for h in hits} <= allowed


def test_subvectors_must_divide_the_dimension() -> None:
    ids, matrix = _clustered_corpus(dimension=30)
    with pytest.raises(ValueError, match="does not divide"):
        IvfIndex(AnnConfig(n_lists=4, n_subvectors=7)).fit(ids, matrix)


def test_empty_index_returns_nothing() -> None:
    index = IvfIndex(AnnConfig(n_lists=4))
    assert index.search(np.zeros(8, dtype=np.float32), k=5) == []


def test_mismatched_ids_are_rejected() -> None:
    ids, matrix = _clustered_corpus()
    with pytest.raises(ValueError, match="disagree"):
        IvfIndex(AnnConfig(n_lists=4)).fit(ids[:-1], matrix)


def test_stats_account_for_every_vector() -> None:
    ids, matrix = _clustered_corpus(n_clusters=4, per_cluster=25)
    index = IvfIndex(AnnConfig(n_lists=4, n_probe=2)).fit(ids, matrix)
    stats = index.stats.as_dict()
    assert stats["train_vectors"] == len(ids)
    assert sum(rows.size for rows in index.lists.values()) == len(ids)
    assert stats["index_memory_mib"] > 0
