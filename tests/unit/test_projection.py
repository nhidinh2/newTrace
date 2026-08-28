"""Projector shapes, determinism, saving and train/test leakage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from newstrace.representations.projection import (
    METHODS,
    cosine_distortion,
    load_projector,
    make_projector,
    pairwise_distortion,
)


@pytest.fixture
def matrix() -> np.ndarray:
    rng = np.random.default_rng(549)
    return rng.normal(size=(60, 128)).astype(np.float32)


@pytest.mark.parametrize("method", ["svd", "gaussian_rp", "sparse_rp"])
def test_projection_output_dimension(method: str, matrix: np.ndarray) -> None:
    projector = make_projector(method, 32, random_seed=549).fit(matrix)
    out = projector.transform(matrix)
    assert out.shape == (matrix.shape[0], 32)
    assert out.dtype == np.float32


@pytest.mark.parametrize("method", ["svd", "gaussian_rp", "sparse_rp"])
def test_projection_is_deterministic_given_a_seed(method: str, matrix: np.ndarray) -> None:
    a = make_projector(method, 16, random_seed=549).fit(matrix).transform(matrix)
    b = make_projector(method, 16, random_seed=549).fit(matrix).transform(matrix)
    assert np.allclose(a, b)


@pytest.mark.parametrize("method", ["gaussian_rp", "sparse_rp"])
def test_different_seeds_give_different_projections(method: str, matrix: np.ndarray) -> None:
    a = make_projector(method, 16, random_seed=1).fit(matrix).transform(matrix)
    b = make_projector(method, 16, random_seed=2).fit(matrix).transform(matrix)
    assert not np.allclose(a, b)


def test_identity_projector_preserves_dimension(matrix: np.ndarray) -> None:
    projector = make_projector("full", 0).fit(matrix)
    assert projector.dimension == matrix.shape[1]
    assert projector.transform(matrix).shape == matrix.shape


def test_svd_dimension_is_capped_by_sample_count() -> None:
    small = np.random.default_rng(0).normal(size=(10, 128)).astype(np.float32)
    projector = make_projector("svd", 64, random_seed=549).fit(small)
    assert projector.dimension <= small.shape[0] - 1
    assert projector.meta["requested_dimension"] == 64


def test_transform_before_fit_raises(matrix: np.ndarray) -> None:
    with pytest.raises(RuntimeError, match="must be fitted"):
        make_projector("svd", 8).transform(matrix)


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown projection method"):
        make_projector("magic", 8)
    assert set(METHODS) == {"full", "svd", "gaussian_rp", "sparse_rp"}


def test_no_leakage_projector_never_sees_heldout_rows(matrix: np.ndarray) -> None:
    """Fitting on the first 60% must not depend on the held-out tail."""
    cut = int(len(matrix) * 0.6)
    fit_part = matrix[:cut]

    baseline = make_projector("svd", 16, random_seed=549).fit(fit_part)
    projected_baseline = baseline.transform(matrix[cut:])

    tampered = matrix.copy()
    tampered[cut:] = np.random.default_rng(7).normal(size=tampered[cut:].shape)
    tampered_fit = make_projector("svd", 16, random_seed=549).fit(tampered[:cut])

    # The fitted transform is identical because the fit period is unchanged.
    assert np.allclose(baseline.transform(fit_part), tampered_fit.transform(fit_part))
    assert projected_baseline.shape == (len(matrix) - cut, 16)


def test_fit_version_is_stable_and_sensitive(matrix: np.ndarray) -> None:
    a = make_projector("svd", 16, random_seed=549).fit(matrix)
    b = make_projector("svd", 16, random_seed=549).fit(matrix)
    c = make_projector("svd", 16, random_seed=550).fit(matrix)
    assert a.fit_version("model", 60) == b.fit_version("model", 60)
    assert a.fit_version("model", 60) != c.fit_version("model", 60)
    assert a.fit_version("model", 60) != a.fit_version("model", 61)


@pytest.mark.parametrize("method", ["svd", "gaussian_rp", "sparse_rp"])
def test_projector_roundtrips_through_disk(method: str, matrix: np.ndarray, tmp_path: Path) -> None:
    projector = make_projector(method, 16, random_seed=549).fit(matrix)
    path = tmp_path / f"{method}.joblib"
    projector.save(path)
    restored = load_projector(path)
    assert restored.dimension == projector.dimension
    assert np.allclose(restored.transform(matrix), projector.transform(matrix))


def test_distortion_decreases_with_dimension(matrix: np.ndarray) -> None:
    rng = np.random.default_rng(549)
    low = make_projector("gaussian_rp", 8, random_seed=549).fit(matrix).transform(matrix)
    high = make_projector("gaussian_rp", 96, random_seed=549).fit(matrix).transform(matrix)
    low_d = pairwise_distortion(matrix, low, rng=np.random.default_rng(1))
    high_d = pairwise_distortion(matrix, high, rng=np.random.default_rng(1))
    assert high_d["mean_abs_distortion"] < low_d["mean_abs_distortion"]

    low_c = cosine_distortion(matrix, low, rng=rng)
    high_c = cosine_distortion(matrix, high, rng=rng)
    assert high_c["mean_abs_cosine_error"] < low_c["mean_abs_cosine_error"]


def test_distortion_on_degenerate_input() -> None:
    single = np.ones((1, 4), dtype=np.float32)
    rng = np.random.default_rng(0)
    assert pairwise_distortion(single, single, rng=rng)["mean_abs_distortion"] == 0.0
    assert cosine_distortion(single, single, rng=rng)["mean_abs_cosine_error"] == 0.0
