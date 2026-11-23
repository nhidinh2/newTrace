"""Invalid ANN experiments must fail before training or producing artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from newstrace.evaluation.ann_sweep import AnnSweepConfig, run_ann_sweep
from newstrace.retrieval.ann import AnnConfig, IvfIndex


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_lists", 0),
        ("n_lists", -1),
        ("n_probe", 0),
        ("n_probe", -1),
        ("n_subvectors", -1),
        ("n_bits", 0),
        ("n_bits", 9),
        ("seed", -1),
        ("seed", 2**32),
        ("n_lists", 1.5),
        ("n_probe", True),
    ],
)
def test_invalid_index_config_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        AnnConfig(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_lists", []),
        ("n_probes", []),
        ("n_subvectors", []),
        ("n_lists", [1, 0]),
        ("n_probes", [-1]),
        ("n_subvectors", [-1]),
        ("n_bits", 9),
        ("top_k", 0),
        ("max_queries", 0),
        ("latency_repeats", 0),
        ("since_days", 0),
        ("since_days", -1),
        ("compare_dimensions", [0]),
        ("seed", -1),
        ("seed", 2**32),
        ("n_lists", [True]),
        ("max_queries", 1.5),
    ],
)
def test_invalid_sweep_config_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        AnnSweepConfig(**{field: value})


def test_mutated_sweep_config_is_rechecked_before_persistence() -> None:
    config = AnnSweepConfig()
    config.n_probes.clear()
    # No session calls, RNG changes, database rows or artifact creation.
    with pytest.raises(ValueError, match="n_probes"):
        run_ann_sweep(None, config)


def test_mutated_index_config_is_rechecked_before_training(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AnnConfig()
    config.n_bits = 9
    monkeypatch.setattr(
        "newstrace.retrieval.ann._kmeans",
        lambda *args: pytest.fail("invalid config reached k-means"),
    )
    with pytest.raises(ValueError, match="n_bits"):
        IvfIndex(config).fit([1, 2], np.eye(2, dtype=np.float32))


def test_incompatible_subvectors_fail_before_coarse_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "newstrace.retrieval.ann._kmeans",
        lambda *args: pytest.fail("invalid PQ layout reached coarse k-means"),
    )
    with pytest.raises(ValueError, match="does not divide"):
        IvfIndex(AnnConfig(n_subvectors=3)).fit([1, 2], np.eye(2, dtype=np.float32))


@pytest.mark.parametrize("matrix", [np.empty((2, 0)), np.zeros(2), np.array(1.0)])
def test_invalid_matrix_shape_is_rejected_before_training(matrix: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"two-dimensional.*positive"):
        IvfIndex().fit([1, 2], matrix)


def test_valid_boundary_configs_and_empty_comparison_arm() -> None:
    assert AnnConfig(n_lists=1, n_probe=100, n_subvectors=0, n_bits=1, seed=0)
    assert AnnConfig(n_bits=8, seed=2**32 - 1)
    assert AnnSweepConfig(compare_dimensions=[], n_subvectors=[0], n_bits=1, seed=0)


@pytest.mark.parametrize("n_bits", [1, 8])
def test_supported_pq_widths_preserve_codebook_entries(n_bits: int) -> None:
    index = IvfIndex(AnnConfig(n_subvectors=1, n_bits=n_bits))
    index._dimension = 2
    size = 2**n_bits
    book = np.column_stack((np.arange(size, dtype=np.float32), np.zeros(size, dtype=np.float32)))
    index.codebooks = book[None, :, :]
    codes = index._encode(book)
    assert codes.dtype == np.uint8
    np.testing.assert_array_equal(codes[:, 0], np.arange(size))
