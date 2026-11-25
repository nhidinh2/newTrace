"""Rebuilding an approximate index must replace a complete, coherent corpus."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from newstrace.retrieval import ann
from newstrace.retrieval.ann import AnnConfig, IvfIndex
from newstrace.retrieval.exact import DenseRetriever


def _corpus(count: int, dimension: int, first_id: int, seed: int = 17):
    matrix = np.random.default_rng(seed).normal(size=(count, dimension)).astype(np.float32)
    return list(range(first_id, first_id + count)), matrix


def _assert_current_corpus(index: IvfIndex, ids: list[int], matrix: np.ndarray) -> None:
    rows = np.concatenate(list(index.lists.values()))
    assert sorted(rows.tolist()) == list(range(len(ids)))
    assert index.lists.keys() == index.payload.keys()
    assert all(0 <= cell < len(index.centroids) for cell in index.lists)
    assert index.article_ids.tolist() == ids
    assert index.stats.train_vectors == len(ids)
    assert index.stats.lists_used == len(index.lists)
    assert index.stats.mean_list_size == pytest.approx(len(ids) / len(index.lists))
    assert index.stats.max_list_size == max(len(rows) for rows in index.lists.values())
    assert index.stats.payload_bytes == sum(block.nbytes for block in index.payload.values())
    assert index.stats.codebook_bytes == index.centroids.nbytes + index.codebooks.nbytes
    assert index.memory_bytes() == (
        index.centroids.nbytes
        + index.codebooks.nbytes
        + sum(block.nbytes for block in index.payload.values())
    )
    for query in matrix[:3]:
        hits = index.search(query, k=len(ids), n_probe=len(index.centroids))
        assert sorted(hit.article_id for hit in hits) == ids
        assert len({hit.article_id for hit in hits}) == len(hits)
        assert index.search(query, allowed_ids={-1}) == []
        assert {h.article_id for h in index.search(query, allowed_ids={ids[0]})} == {ids[0]}


@pytest.mark.parametrize("subvectors", [0, 2])
def test_refit_removes_old_cells_when_corpus_shrinks(subvectors: int) -> None:
    ids, matrix = _corpus(24, 8, 1)
    index = IvfIndex(AnnConfig(n_lists=6, n_subvectors=subvectors, n_bits=1)).fit(ids, matrix)
    ids, matrix = _corpus(2, 8, 101)
    assert index.fit(ids, matrix) is index
    _assert_current_corpus(index, ids, matrix)


@pytest.mark.parametrize("subvectors", [0, 2])
def test_refit_removes_cells_that_become_empty(subvectors: int) -> None:
    ids, matrix = _corpus(24, 8, 1)
    index = IvfIndex(AnnConfig(n_lists=4, n_subvectors=subvectors, n_bits=1)).fit(ids, matrix)
    ids = list(range(101, 125))
    matrix = np.ones((24, 8), dtype=np.float32)
    with pytest.warns(ConvergenceWarning):
        index.fit(ids, matrix)
    _assert_current_corpus(index, ids, matrix)
    assert index.stats.lists_used == 1


@pytest.mark.parametrize("before,after", [(2, 0), (0, 2), (2, 3), (0, 0)])
def test_refit_changes_layout_and_dimension_like_a_fresh_index(before: int, after: int) -> None:
    ids, matrix = _corpus(16, 8, 1)
    config = AnnConfig(n_lists=4, n_subvectors=before, n_bits=1)
    index = IvfIndex(config).fit(ids, matrix)
    config.n_subvectors = after
    ids, matrix = _corpus(12, 12, 101)
    index.fit(ids, matrix)
    fresh = IvfIndex(replace(config)).fit(ids, matrix)
    _assert_current_corpus(index, ids, matrix)
    np.testing.assert_array_equal(index.codebooks, fresh.codebooks)
    assert index.memory_bytes() == fresh.memory_bytes()
    for query in matrix[:3]:
        assert index.search(query, k=6, n_probe=4) == fresh.search(query, k=6, n_probe=4)
        if after == 0:
            exact = DenseRetriever(ids, matrix).search(query, k=6)
            assert [h.article_id for h in index.search(query, k=6, n_probe=4)] == [
                h.article_id for h in exact
            ]
    if after == 0:
        assert index.codebooks.size == 0


@pytest.mark.parametrize("subvectors", [0, 2])
@pytest.mark.parametrize("failure", ["ids", "coarse", "encode"])
def test_failed_refit_preserves_the_previous_index(
    subvectors: int, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids, matrix = _corpus(16, 8, 1)
    index = IvfIndex(AnnConfig(n_lists=4, n_subvectors=subvectors, n_bits=1)).fit(ids, matrix)
    expected = index.search(matrix[0], k=8, n_probe=4)
    old_stats = index.stats
    old_memory = index.memory_bytes()
    new_ids, new_matrix = _corpus(6, 12, 101)
    # A changed requested layout must not reinterpret the old fitted payload.
    index.config.n_subvectors = 3

    def fail(*args, **kwargs):
        raise RuntimeError("training interrupted")

    with monkeypatch.context() as patch:
        if failure == "ids":
            new_ids = new_ids[:-1]
            error, message = ValueError, "disagree"
        else:
            if failure == "coarse":
                patch.setattr(ann, "_kmeans", fail)
            else:
                patch.setattr(IvfIndex, "_encode", fail)
            error, message = RuntimeError, "training interrupted"
        with pytest.raises(error, match=message):
            index.fit(new_ids, new_matrix)

    assert index.article_ids.tolist() == ids
    assert index.stats is old_stats
    assert index.memory_bytes() == old_memory
    assert index.search(matrix[0], k=8, n_probe=4) == expected
    _assert_current_corpus(index, ids, matrix)
    # The request is still editable, and a valid retry publishes the new corpus.
    new_ids, new_matrix = _corpus(6, 12, 101)
    index.fit(new_ids, new_matrix)
    _assert_current_corpus(index, new_ids, new_matrix)


def test_failed_first_fit_remains_empty_and_can_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    index = IvfIndex(AnnConfig(n_lists=2, n_subvectors=2, n_bits=1))
    ids, matrix = _corpus(6, 8, 1)

    def fail(*args, **kwargs):
        raise RuntimeError("encoding interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(IvfIndex, "_encode", fail)
        with pytest.raises(RuntimeError, match="encoding interrupted"):
            index.fit(ids, matrix)
    assert index.article_ids.size == 0
    assert index.centroids.size == 0
    assert index.codebooks.size == 0
    assert index.memory_bytes() == 0
    assert index.search(matrix[0]) == []
    index.fit(ids, matrix)
    _assert_current_corpus(index, ids, matrix)


@pytest.mark.parametrize("subvectors", [0, 2])
def test_pending_layout_changes_preserve_search_but_probe_count_remains_live(
    subvectors: int,
) -> None:
    ids, matrix = _corpus(24, 8, 1)
    config = AnnConfig(n_lists=4, n_probe=1, n_subvectors=subvectors, n_bits=1)
    index = IvfIndex(config).fit(ids, matrix)
    method = f"ann:{config.name}"
    expected = index.search(matrix[0], k=24, n_probe=4)
    assert len(index.search(matrix[0], k=24)) < len(ids)

    config.n_probe = 4
    config.n_subvectors = 3
    config.n_bits = 2
    assert index.search(matrix[0], k=24) == expected
    assert all(hit.method == method for hit in expected)
    assert len(index.search(matrix[0], k=24, n_probe=1)) < len(ids)
