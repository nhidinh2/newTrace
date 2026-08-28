"""Text embedding with a deterministic, dependency-free fallback.

``sentence-transformers`` is an optional extra.  When it is unavailable (no
model download, offline CI, minimal install) NewsTrace falls back to a
deterministic hashing embedder so the whole system -- ingest, cluster, search,
evaluate -- still runs and still produces reproducible numbers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from newstrace.config import Settings, get_settings
from newstrace.logging import get_logger
from newstrace.utils import normalize_unicode, sha256_text, tokenize

logger = get_logger(__name__)

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used everywhere else in the system."""

    name: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> Matrix: ...


def l2_normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def cache_key(text: str, model_name: str, preprocessing_version: str) -> str:
    """Embedding cache key: model + preprocessing version + normalized text."""
    return sha256_text(f"{model_name}|{preprocessing_version}|{normalize_unicode(text)}")


class HashingEmbedder:
    """Deterministic feature-hashing embedder over word and character n-grams.

    Not competitive with a trained sentence encoder, but it is stable across
    machines, needs no download, and gives a real cosine geometry to test with.
    """

    def __init__(self, dimension: int = 384, *, name: str = "hashing-ngram-v1") -> None:
        self.dimension = int(dimension)
        self.name = f"{name}-{self.dimension}"

    @staticmethod
    def _hash(token: str) -> int:
        return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        features = list(tokens)
        features += [f"{a}_{b}" for a, b in pairwise(tokens)]
        compact = "".join(tokens)
        features += [compact[i : i + 4] for i in range(max(0, len(compact) - 3))]
        return features

    def encode(self, texts: Sequence[str]) -> Matrix:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                h = self._hash(feature)
                index = h % self.dimension
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, index] += sign
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence-transformers`` (CPU by default, GPU if present)."""

    def __init__(self, model_name: str, *, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        # The accessor was renamed in sentence-transformers 5.x.
        getter: Callable[[], Any] = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        dimension = getter()
        if dimension is None:
            raise RuntimeError(f"{model_name} did not report an embedding dimension")
        self.dimension = int(dimension)

    def encode(self, texts: Sequence[str]) -> Matrix:
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def detect_device() -> str:
    """Report an available accelerator without ever requiring one."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_EMBEDDER: Embedder | None = None
_EMBEDDER_KEY: tuple[str, str] | None = None


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Construct the configured embedder, falling back to hashing when needed."""
    settings = settings or get_settings()
    backend = settings.embedding_backend
    if backend in ("auto", "sentence-transformers"):
        try:
            embedder = SentenceTransformerEmbedder(settings.embedding_model, device=detect_device())
            logger.info(
                "Using sentence-transformers embedder %s (dim=%s)",
                embedder.name,
                embedder.dimension,
            )
            return embedder
        except Exception as exc:
            if backend == "sentence-transformers":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s); using deterministic hashing embedder",
                exc.__class__.__name__,
            )
    return HashingEmbedder(settings.embedding_dimension_fallback)


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Process-wide cached embedder."""
    global _EMBEDDER, _EMBEDDER_KEY
    settings = settings or get_settings()
    key = (settings.embedding_backend, settings.embedding_model)
    if _EMBEDDER is None or key != _EMBEDDER_KEY:
        _EMBEDDER = build_embedder(settings)
        _EMBEDDER_KEY = key
    return _EMBEDDER


def reset_embedder() -> None:
    global _EMBEDDER, _EMBEDDER_KEY
    _EMBEDDER = None
    _EMBEDDER_KEY = None


def to_blob(vector: NDArray[np.float32]) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes | None, dimension: int) -> Vector:
    if not blob:
        return np.zeros(dimension, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32).reshape(dimension).copy()
