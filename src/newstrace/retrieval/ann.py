"""Approximate nearest-neighbour search: IVF, with optional product quantisation.

The exact baseline in :mod:`newstrace.retrieval.exact` is honest but linear: it
scores the query against every stored vector. This module is the other way to
buy the same memory and latency the dimensionality sweep buys by compressing
the vectors -- and the point of having both is that they interact. A 32-d SVD
index and an IVF-PQ index over 384-d vectors cost about the same; whether they
lose the same *documents* is an empirical question, and
``scripts/run_ann.py`` measures it.

Written against numpy and scikit-learn's k-means rather than FAISS, for the
same reason the SimHash and the PageRank here are written out: a dependency
with a compiled wheel per platform would make the offline quickstart a
conditional one, and the interesting part -- the recall/memory trade-off -- is
not in the library.

Everything is L2. Vectors are L2-normalised on the way in, and for unit
vectors ``||x - q||^2 = 2 - 2 x.q``, so ranking by squared distance and
ranking by cosine are the same ranking. That lets the coarse quantiser and the
PQ codebooks be plain k-means, which is what the theory is stated for.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from newstrace.logging import get_logger
from newstrace.representations.embedder import l2_normalize
from newstrace.retrieval.exact import ScoredHit

logger = get_logger(__name__)

Matrix = NDArray[np.float32]


@dataclass
class AnnConfig:
    """One index configuration.

    ``n_lists`` is the number of Voronoi cells, ``n_probe`` how many are
    scanned per query (the recall/latency dial), and ``n_subvectors`` the PQ
    split -- 0 keeps the full vectors in each list (IVF-Flat).
    """

    n_lists: int = 64
    n_probe: int = 8
    n_subvectors: int = 0
    n_bits: int = 8
    seed: int = 549

    @property
    def uses_pq(self) -> bool:
        return self.n_subvectors > 0

    @property
    def name(self) -> str:
        kind = f"pq{self.n_subvectors}x{self.n_bits}" if self.uses_pq else "flat"
        return f"ivf{self.n_lists}_{kind}_probe{self.n_probe}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_lists": self.n_lists,
            "n_probe": self.n_probe,
            "n_subvectors": self.n_subvectors,
            "n_bits": self.n_bits,
            "seed": self.seed,
        }


@dataclass
class AnnStats:
    build_seconds: float = 0.0
    train_vectors: int = 0
    lists_used: int = 0
    mean_list_size: float = 0.0
    max_list_size: int = 0
    codebook_bytes: int = 0
    payload_bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "build_seconds": round(self.build_seconds, 4),
            "train_vectors": self.train_vectors,
            "lists_used": self.lists_used,
            "mean_list_size": round(self.mean_list_size, 2),
            "max_list_size": self.max_list_size,
            "index_memory_mib": round((self.codebook_bytes + self.payload_bytes) / 1048576, 4),
            **self.extra,
        }


def _kmeans(matrix: Matrix, k: int, seed: int) -> Matrix:
    """k-means centroids, with k clamped to the number of distinct rows."""
    k = max(1, min(k, matrix.shape[0]))
    model = KMeans(n_clusters=k, n_init=4, random_state=seed, max_iter=100)
    model.fit(matrix)
    return np.asarray(model.cluster_centers_, dtype=np.float32)


class IvfIndex:
    """Inverted-file index with an optional product-quantised payload."""

    def __init__(self, config: AnnConfig | None = None) -> None:
        self.config = config or AnnConfig()
        self.centroids: Matrix = np.zeros((0, 0), dtype=np.float32)
        self.codebooks: Matrix = np.zeros((0, 0, 0), dtype=np.float32)
        self.lists: dict[int, NDArray[np.int64]] = {}
        self.payload: dict[int, NDArray[np.uint8] | Matrix] = {}
        self.article_ids: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self.stats = AnnStats()
        self._dimension = 0

    # -- build ---------------------------------------------------------------

    def fit(self, article_ids: Sequence[int], matrix: Matrix) -> IvfIndex:
        """Train the quantisers and add every vector. Returns ``self``."""
        started = time.perf_counter()
        data = l2_normalize(np.asarray(matrix, dtype=np.float32))
        if data.ndim != 2 or data.shape[0] == 0:
            raise ValueError("an IVF index needs at least one vector")
        self.article_ids = np.asarray(list(article_ids), dtype=np.int64)
        if self.article_ids.shape[0] != data.shape[0]:
            raise ValueError("article_ids and matrix disagree on the number of vectors")
        self._dimension = int(data.shape[1])

        self.centroids = _kmeans(data, self.config.n_lists, self.config.seed)
        assignments = self._assign(data)

        residuals = data - self.centroids[assignments]
        codes: NDArray[np.uint8] | Matrix
        if self.config.uses_pq:
            self._fit_codebooks(residuals)
            codes = self._encode(residuals)
        else:
            codes = residuals  # IVF-Flat keeps the residual itself.

        payload_bytes = 0
        for cell in range(self.centroids.shape[0]):
            rows = np.flatnonzero(assignments == cell).astype(np.int64)
            if rows.size == 0:
                continue
            self.lists[cell] = rows
            block = codes[rows]
            self.payload[cell] = block
            payload_bytes += int(block.nbytes)

        sizes = [int(rows.size) for rows in self.lists.values()]
        self.stats = AnnStats(
            build_seconds=time.perf_counter() - started,
            train_vectors=int(data.shape[0]),
            lists_used=len(self.lists),
            mean_list_size=float(np.mean(sizes)) if sizes else 0.0,
            max_list_size=max(sizes) if sizes else 0,
            codebook_bytes=int(self.centroids.nbytes + self.codebooks.nbytes),
            payload_bytes=payload_bytes,
        )
        logger.info(
            "Built %s over %s vectors: %s", self.config.name, data.shape[0], self.stats.as_dict()
        )
        return self

    def _assign(self, data: Matrix) -> NDArray[np.int64]:
        # ||x - c||^2 expanded, so the whole assignment is one matrix product.
        sq = (
            (data**2).sum(axis=1, keepdims=True)
            + (self.centroids**2).sum(axis=1)
            - 2 * (data @ self.centroids.T)
        )
        return np.asarray(np.argmin(sq, axis=1), dtype=np.int64)

    def _subvector_slices(self) -> list[tuple[int, int]]:
        m = self.config.n_subvectors
        if self._dimension % m:
            raise ValueError(
                f"n_subvectors={m} does not divide the embedding dimension {self._dimension}"
            )
        width = self._dimension // m
        return [(i * width, (i + 1) * width) for i in range(m)]

    def _fit_codebooks(self, residuals: Matrix) -> None:
        centroids_per_subspace = 2**self.config.n_bits
        books = []
        for start, stop in self._subvector_slices():
            books.append(
                _kmeans(residuals[:, start:stop], centroids_per_subspace, self.config.seed)
            )
        # (m, 2**bits, width); every subspace has the same width.
        self.codebooks = np.stack(books).astype(np.float32)

    def _encode(self, residuals: Matrix) -> NDArray[np.uint8]:
        codes = np.empty((residuals.shape[0], self.config.n_subvectors), dtype=np.uint8)
        for index, (start, stop) in enumerate(self._subvector_slices()):
            book = self.codebooks[index]
            block = residuals[:, start:stop]
            sq = (
                (block**2).sum(axis=1, keepdims=True) + (book**2).sum(axis=1) - 2 * (block @ book.T)
            )
            codes[:, index] = np.argmin(sq, axis=1).astype(np.uint8)
        return codes

    # -- search --------------------------------------------------------------

    def search(
        self,
        vector: NDArray[np.float32],
        *,
        k: int = 10,
        n_probe: int | None = None,
        allowed_ids: set[int] | None = None,
    ) -> list[ScoredHit]:
        """Top ``k`` by approximate cosine similarity."""
        if not self.lists:
            return []
        probe = max(1, min(n_probe or self.config.n_probe, self.centroids.shape[0]))
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(q))
        if norm > 1e-9:
            q = q / norm

        coarse = ((self.centroids - q) ** 2).sum(axis=1)
        cells = np.argsort(coarse, kind="stable")[:probe]

        ids: list[NDArray[np.int64]] = []
        distances: list[NDArray[np.float32]] = []
        for cell in cells:
            rows = self.lists.get(int(cell))
            if rows is None:
                continue
            residual_query = q - self.centroids[int(cell)]
            payload = self.payload[int(cell)]
            if self.config.uses_pq:
                table = self._distance_table(residual_query)
                codes = np.asarray(payload, dtype=np.uint8)
                # Asymmetric distance computation: sum the per-subspace
                # lookups, which is the whole point of PQ -- no vector in the
                # list is ever reconstructed.
                block = table[np.arange(self.config.n_subvectors), codes].sum(axis=1)
            else:
                diff = np.asarray(payload, dtype=np.float32) - residual_query
                block = (diff**2).sum(axis=1)
            ids.append(self.article_ids[rows])
            distances.append(np.asarray(block, dtype=np.float32))

        if not ids:
            return []
        candidate_ids = np.concatenate(ids)
        candidate_distances = np.concatenate(distances)
        if allowed_ids is not None:
            keep = np.isin(
                candidate_ids, np.fromiter(allowed_ids, dtype=np.int64, count=len(allowed_ids))
            )
            candidate_ids = candidate_ids[keep]
            candidate_distances = candidate_distances[keep]
        if candidate_ids.size == 0:
            return []

        take = min(k, candidate_ids.size)
        order = np.argpartition(candidate_distances, take - 1)[:take]
        order = order[np.lexsort((candidate_ids[order], candidate_distances[order]))]
        hits: list[ScoredHit] = []
        for rank, position in enumerate(order, start=1):
            # Back to cosine, so scores are comparable with the exact retriever.
            cosine = 1.0 - float(candidate_distances[position]) / 2.0
            hits.append(
                ScoredHit(
                    article_id=int(candidate_ids[position]),
                    score=cosine,
                    rank=rank,
                    method=f"ann:{self.config.name}",
                    signals={"cosine_estimate": cosine},
                )
            )
        return hits

    def _distance_table(self, residual_query: NDArray[np.float32]) -> Matrix:
        """(m, 2**bits) squared distances from each subquery to each centroid."""
        table = np.empty((self.config.n_subvectors, self.codebooks.shape[1]), dtype=np.float32)
        for index, (start, stop) in enumerate(self._subvector_slices()):
            book = self.codebooks[index]
            diff = book - residual_query[start:stop]
            table[index] = (diff**2).sum(axis=1)
        return table

    # -- accounting ----------------------------------------------------------

    def memory_bytes(self) -> int:
        """Bytes the index holds, excluding the article ids."""
        return int(self.stats.codebook_bytes + self.stats.payload_bytes)

    def benchmark(
        self, vectors: Sequence[NDArray[np.float32]], *, k: int = 10, repeats: int = 3
    ) -> dict[str, float]:
        timings: list[float] = []
        for _ in range(repeats):
            for vector in vectors:
                start = time.perf_counter()
                self.search(vector, k=k)
                timings.append((time.perf_counter() - start) * 1000.0)
        if not timings:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
        arr = np.asarray(timings)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean()),
            "queries": float(len(timings)),
        }
