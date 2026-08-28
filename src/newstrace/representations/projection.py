"""Dimensionality reduction: identity, truncated SVD, Gaussian RP, sparse RP.

All projectors share one interface so retrieval and clustering never need to
know which representation they are running on.  Fits are seeded and saved with
version metadata so an experiment can be replayed exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from newstrace.representations.embedder import l2_normalize
from newstrace.utils import sha256_obj

Matrix = NDArray[np.float32]

METHODS = ("full", "svd", "gaussian_rp", "sparse_rp")


class Projector(Protocol):
    """Interface required by the README."""

    method: str
    dimension: int

    def fit(self, x: Matrix) -> Projector: ...
    def transform(self, x: Matrix) -> Matrix: ...
    def save(self, path: Path) -> None: ...


@dataclass
class BaseProjector:
    method: str = "full"
    dimension: int = 0
    random_seed: int = 549
    fitted: bool = False
    normalize_output: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def _post_transform(self, x: Matrix) -> Matrix:
        out = np.asarray(x, dtype=np.float32)
        return l2_normalize(out) if self.normalize_output else out

    def fit_version(self, source_model: str, fit_article_count: int) -> str:
        """Stable identifier for this fit, used as the embedding ``fit_version``."""
        return sha256_obj(
            {
                "method": self.method,
                "dimension": self.dimension,
                "seed": self.random_seed,
                "model": source_model,
                "n_fit": fit_article_count,
                "meta": self.meta,
            }
        )[:16]

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def fit(self, x: Matrix) -> BaseProjector:
        raise NotImplementedError

    def transform(self, x: Matrix) -> Matrix:
        raise NotImplementedError


@dataclass
class IdentityProjector(BaseProjector):
    """Full-dimensional baseline: passes embeddings through unchanged."""

    method: str = "full"

    def fit(self, x: Matrix) -> IdentityProjector:
        self.dimension = int(np.asarray(x).shape[1])
        self.fitted = True
        return self

    def transform(self, x: Matrix) -> Matrix:
        return self._post_transform(x)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"method": self.method, "dimension": self.dimension}), encoding="utf-8"
        )


@dataclass
class SklearnProjector(BaseProjector):
    """Shared implementation for the scikit-learn backed projectors."""

    model: Any = None

    def _build(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, x: Matrix) -> SklearnProjector:
        data = np.asarray(x, dtype=np.float32)
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("projector fit requires a non-empty 2-D matrix")
        max_dim = min(self.dimension, data.shape[1] - 1 if self.method == "svd" else data.shape[1])
        max_dim = min(max_dim, max(1, data.shape[0] - 1)) if self.method == "svd" else max_dim
        if max_dim < 1:
            raise ValueError("target dimension must be >= 1")
        if max_dim != self.dimension:
            self.meta["requested_dimension"] = self.dimension
            self.dimension = int(max_dim)
        self.model = self._build()
        self.model.fit(data)
        self.fitted = True
        return self

    def transform(self, x: Matrix) -> Matrix:
        if not self.fitted or self.model is None:
            raise RuntimeError(f"{self.method} projector must be fitted before transform")
        data = np.asarray(x, dtype=np.float32)
        return self._post_transform(self.model.transform(data))

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "method": self.method,
                "dimension": self.dimension,
                "random_seed": self.random_seed,
                "meta": self.meta,
                "model": self.model,
            },
            path,
        )


@dataclass
class SvdProjector(SklearnProjector):
    method: str = "svd"

    def _build(self) -> Any:
        return TruncatedSVD(
            n_components=self.dimension, random_state=self.random_seed, algorithm="randomized"
        )


@dataclass
class GaussianRpProjector(SklearnProjector):
    method: str = "gaussian_rp"

    def _build(self) -> Any:
        return GaussianRandomProjection(n_components=self.dimension, random_state=self.random_seed)


@dataclass
class SparseRpProjector(SklearnProjector):
    method: str = "sparse_rp"

    def _build(self) -> Any:
        return SparseRandomProjection(
            n_components=self.dimension, random_state=self.random_seed, dense_output=True
        )


def make_projector(method: str, dimension: int, *, random_seed: int = 549) -> BaseProjector:
    """Factory used by the experiment runner and the retrieval registry."""
    if method not in METHODS:
        raise ValueError(f"unknown projection method {method!r}; expected one of {METHODS}")
    if method == "full":
        return IdentityProjector(random_seed=random_seed)
    if dimension < 1:
        raise ValueError("dimension must be >= 1 for compressed methods")
    cls = {
        "svd": SvdProjector,
        "gaussian_rp": GaussianRpProjector,
        "sparse_rp": SparseRpProjector,
    }[method]
    return cls(dimension=int(dimension), random_seed=random_seed)


def load_projector(path: Path) -> BaseProjector:
    """Load a projector previously written by :meth:`BaseProjector.save`."""
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        proj = IdentityProjector(dimension=int(payload["dimension"]))
        proj.fitted = True
        return proj
    import joblib

    payload = joblib.load(path)
    projector = make_projector(payload["method"], int(payload["dimension"]))
    assert isinstance(projector, SklearnProjector)
    projector.model = payload["model"]
    projector.random_seed = int(payload.get("random_seed", 549))
    projector.meta = dict(payload.get("meta") or {})
    projector.fitted = True
    return projector


def pairwise_distortion(
    original: Matrix, projected: Matrix, *, rng: np.random.Generator, sample_pairs: int = 2000
) -> dict[str, float]:
    """Empirical distance distortion, the JL-style diagnostic from the README."""
    n = original.shape[0]
    if n < 2:
        return {"mean_abs_distortion": 0.0, "p95_abs_distortion": 0.0, "max_abs_distortion": 0.0}
    pairs = min(sample_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return {"mean_abs_distortion": 0.0, "p95_abs_distortion": 0.0, "max_abs_distortion": 0.0}
    d_orig = np.linalg.norm(original[i] - original[j], axis=1)
    d_proj = np.linalg.norm(projected[i] - projected[j], axis=1)
    safe = d_orig > 1e-8
    ratio = np.ones_like(d_orig)
    ratio[safe] = d_proj[safe] / d_orig[safe]
    distortion = np.abs(ratio - 1.0)
    return {
        "mean_abs_distortion": float(distortion.mean()),
        "p95_abs_distortion": float(np.percentile(distortion, 95)),
        "max_abs_distortion": float(distortion.max()),
        "mean_ratio": float(ratio.mean()),
        "pairs_sampled": int(i.size),
    }


def cosine_distortion(
    original: Matrix, projected: Matrix, *, rng: np.random.Generator, sample_pairs: int = 2000
) -> dict[str, float]:
    """Absolute cosine-similarity error before/after compression."""
    n = original.shape[0]
    if n < 2:
        return {"mean_abs_cosine_error": 0.0, "p95_abs_cosine_error": 0.0}
    pairs = min(sample_pairs, n * (n - 1) // 2)
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return {"mean_abs_cosine_error": 0.0, "p95_abs_cosine_error": 0.0}
    orig = l2_normalize(original)
    proj = l2_normalize(projected)
    cos_o = np.sum(orig[i] * orig[j], axis=1)
    cos_p = np.sum(proj[i] * proj[j], axis=1)
    err = np.abs(cos_o - cos_p)
    return {
        "mean_abs_cosine_error": float(err.mean()),
        "p95_abs_cosine_error": float(np.percentile(err, 95)),
    }
