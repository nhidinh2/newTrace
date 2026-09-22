"""Exact retrieval baselines: vectorised cosine similarity and TF-IDF.

The exact NumPy baseline is validated first; an approximate index is only worth
adding once these numbers exist.
"""

from __future__ import annotations

import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from newstrace.representations.embedder import Embedder, get_embedder, l2_normalize

Matrix = NDArray[np.float32]


@dataclass
class ScoredHit:
    """One retrieved article with its score and ranking provenance."""

    article_id: int
    score: float
    rank: int
    method: str
    signals: dict[str, float] = field(default_factory=dict)


def cosine_scores(query: NDArray[np.float32], matrix: Matrix) -> NDArray[np.float32]:
    """Cosine similarity of one query vector against every row of ``matrix``."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    qn = float(np.linalg.norm(q))
    if qn > 1e-9:
        q = q / qn
    normalized = l2_normalize(np.asarray(matrix, dtype=np.float32))
    return (normalized @ q).astype(np.float32)


def top_k(scores: NDArray[np.float32], k: int) -> NDArray[np.int64]:
    """Indices of the ``k`` highest scores, ties broken by index for determinism."""
    if scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    k = min(k, scores.size)
    partial = np.argpartition(-scores, k - 1)[:k]
    return partial[np.lexsort((partial, -scores[partial]))].astype(np.int64)


class DenseRetriever:
    """Exact cosine retrieval over a materialised matrix."""

    def __init__(
        self,
        article_ids: Sequence[int],
        matrix: Matrix,
        *,
        method: str = "full",
        embedder: Embedder | None = None,
        projector: Any = None,
    ) -> None:
        self.article_ids = list(article_ids)
        # Normalised once, here. ``search`` scores with a plain matrix-vector
        # product rather than ``cosine_scores`` so that the whole index is not
        # re-normalised on every query.
        self.matrix = l2_normalize(np.asarray(matrix, dtype=np.float32))
        self.method = method
        self._embedder = embedder
        self.projector = projector
        self._id_array = np.asarray(self.article_ids, dtype=np.int64)

    def extend(self, article_ids: Sequence[int], matrix: Matrix) -> None:
        """Append new rows in place; used to grow a cached index after ingest."""
        if not len(article_ids):
            return
        block = l2_normalize(np.asarray(matrix, dtype=np.float32))
        if block.shape[1] != self.matrix.shape[1]:
            raise ValueError(
                f"cannot extend a {self.matrix.shape[1]}-d index with {block.shape[1]}-d vectors"
            )
        self.matrix = np.vstack([self.matrix, block])
        self.article_ids.extend(int(a) for a in article_ids)
        self._id_array = np.asarray(self.article_ids, dtype=np.int64)

    def _mask_for(self, allowed_ids: Collection[int]) -> NDArray[np.bool_]:
        """Boolean row mask for ``allowed_ids``, vectorised.

        ``np.isin`` sorts and searches in C. The obvious Python comprehension
        over ``article_ids`` costs milliseconds per query once the corpus is a
        few thousand articles, which is the same order as the search itself.
        """
        allowed = np.fromiter(allowed_ids, dtype=np.int64, count=len(allowed_ids))
        return np.isin(self._id_array, allowed)

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def embed_query(self, query: str) -> NDArray[np.float32]:
        vector = self.embedder.encode([query])[0]
        if self.projector is not None and getattr(self.projector, "method", "full") != "full":
            vector = self.projector.transform(vector.reshape(1, -1))[0]
        return np.asarray(vector, dtype=np.float32)

    def search(
        self,
        query: str | NDArray[np.float32],
        *,
        k: int = 10,
        allowed_ids: set[int] | None = None,
    ) -> list[ScoredHit]:
        vector = self.embed_query(query) if isinstance(query, str) else query
        scores = self.score(vector)
        if allowed_ids is not None:
            scores = np.where(self._mask_for(allowed_ids), scores, -np.inf).astype(np.float32)
        hits: list[ScoredHit] = []
        for rank, idx in enumerate(top_k(scores, k), start=1):
            score = float(scores[idx])
            if not np.isfinite(score):
                continue
            hits.append(
                ScoredHit(
                    article_id=self.article_ids[int(idx)],
                    score=score,
                    rank=rank,
                    method=f"cosine:{self.method}",
                    signals={"cosine": score},
                )
            )
        return hits

    def score(self, vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """Cosine scores against every row (the index is already normalised)."""
        if self.matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm > 1e-9:
            q = q / norm
        return (self.matrix @ q).astype(np.float32)

    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes)

    def benchmark(self, queries: Sequence[str], *, k: int = 10) -> dict[str, float]:
        """p50/p95 query latency in milliseconds over ``queries``."""
        vectors = [self.embed_query(q) for q in queries]
        timings: list[float] = []
        for vector in vectors:
            start = time.perf_counter()
            self.search(vector, k=k)
            timings.append((time.perf_counter() - start) * 1000.0)
        if not timings:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
        arr = np.array(timings)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
            "queries": float(len(timings)),
        }


class TfidfRetriever:
    """Lexical TF-IDF baseline; no external service, no embeddings."""

    method = "tfidf"

    def __init__(self, article_ids: Sequence[int], texts: Sequence[str]) -> None:
        self.article_ids = list(article_ids)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            sublinear_tf=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.matrix = (
            self.vectorizer.fit_transform(list(texts))
            if texts
            else self.vectorizer.fit_transform([""])
        )

    def search(
        self, query: str, *, k: int = 10, allowed_ids: set[int] | None = None
    ) -> list[ScoredHit]:
        if not self.article_ids:
            return []
        q = self.vectorizer.transform([query])
        scores = np.asarray((self.matrix @ q.T).todense()).reshape(-1).astype(np.float32)
        if allowed_ids is not None:
            allowed = np.fromiter(allowed_ids, dtype=np.int64, count=len(allowed_ids))
            mask = np.isin(np.asarray(self.article_ids, dtype=np.int64), allowed)
            scores = np.where(mask, scores, -np.inf).astype(np.float32)
        hits = []
        for rank, idx in enumerate(top_k(scores, k), start=1):
            score = float(scores[idx])
            if not np.isfinite(score) or score <= 0.0:
                continue
            hits.append(
                ScoredHit(
                    article_id=self.article_ids[int(idx)],
                    score=score,
                    rank=rank,
                    method="tfidf",
                    signals={"tfidf": score},
                )
            )
        return hits

    def memory_bytes(self) -> int:
        return int(self.matrix.data.nbytes + self.matrix.indices.nbytes + self.matrix.indptr.nbytes)


def recency_weight(published_at: datetime | None, now: datetime, half_life_hours: float) -> float:
    """Exponential recency decay used as an explicit, reportable ranking signal."""
    if published_at is None or half_life_hours <= 0:
        return 1.0
    age_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
    return float(0.5 ** (age_hours / half_life_hours))
