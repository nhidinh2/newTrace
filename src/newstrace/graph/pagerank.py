"""Personalized PageRank over the news graph.

PageRank is the stationary distribution of a lazy random walk that restarts, with
probability ``1 - damping``, into the personalization vector.  We implement the
power iteration directly so we can report convergence behaviour (iterations, L1
residual per step) and handle dangling nodes explicitly, which the README asks
for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

from newstrace.graph.build import NODE_ARTICLE, article_node, parse_node


@dataclass
class PageRankResult:
    """Stationary distribution plus convergence diagnostics."""

    scores: dict[str, float]
    iterations: int
    converged: bool
    residuals: list[float] = field(default_factory=list)
    damping: float = 0.85
    dangling_nodes: int = 0

    def total_mass(self) -> float:
        return float(sum(self.scores.values()))

    def article_scores(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for node, value in self.scores.items():
            kind, raw = parse_node(node)
            if kind == NODE_ARTICLE:
                out[int(raw)] = value
        return out


def personalized_pagerank(
    graph: nx.DiGraph,
    *,
    personalization: Mapping[str, float] | None = None,
    damping: float = 0.85,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    weight: str = "weight",
) -> PageRankResult:
    """Power-iteration PageRank with explicit dangling-node redistribution.

    ``max_iterations`` is generous on purpose: convergence slows sharply as the
    damping factor approaches 1, and a cap that silently truncates the walk
    would report a distribution that is not stationary.
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return PageRankResult({}, 0, True, [], damping, 0)

    index = {node: i for i, node in enumerate(nodes)}

    # Personalization / restart distribution.
    if personalization:
        vec = np.zeros(n, dtype=np.float64)
        for node, value in personalization.items():
            if node in index and value > 0:
                vec[index[node]] = float(value)
        total = vec.sum()
        restart = vec / total if total > 0 else np.full(n, 1.0 / n)
    else:
        restart = np.full(n, 1.0 / n)

    # Row-stochastic transition matrix (dangling rows handled separately).
    out_weight = np.zeros(n, dtype=np.float64)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for u, v, data in graph.edges(data=True):
        w = float(data.get(weight, 1.0))
        if w <= 0:
            continue
        i, j = index[u], index[v]
        rows.append(i)
        cols.append(j)
        vals.append(w)
        out_weight[i] += w

    dangling = out_weight <= 0
    normalized = [w / out_weight[i] for i, w in zip(rows, vals, strict=True)]
    transition = csr_matrix((normalized, (rows, cols)), shape=(n, n), dtype=np.float64)

    rank = restart.copy()
    residuals: list[float] = []
    converged = False
    iterations = 0
    for step in range(1, max_iterations + 1):
        iterations = step
        dangling_mass = float(rank[dangling].sum())
        new_rank = damping * (rank @ transition + dangling_mass * restart)
        new_rank += (1.0 - damping) * restart
        total = new_rank.sum()
        if total > 0:
            new_rank /= total
        residual = float(np.abs(new_rank - rank).sum())
        residuals.append(residual)
        rank = new_rank
        if residual < tolerance:
            converged = True
            break

    return PageRankResult(
        scores={node: float(rank[i]) for node, i in index.items()},
        iterations=iterations,
        converged=converged,
        residuals=residuals,
        damping=damping,
        dangling_nodes=int(dangling.sum()),
    )


def seed_from_articles(
    article_ids: Sequence[int], weights: Sequence[float] | None = None
) -> dict[str, float]:
    """Build a personalization vector from retrieved candidate articles."""
    if weights is None:
        weights = [1.0] * len(article_ids)
    seeds: dict[str, float] = {}
    for article_id, weight in zip(article_ids, weights, strict=True):
        seeds[article_node(article_id)] = max(float(weight), 0.0)
    if not any(seeds.values()):
        seeds = dict.fromkeys(seeds, 1.0)
    return seeds


def damping_sensitivity(
    graph: nx.DiGraph,
    *,
    personalization: Mapping[str, float] | None = None,
    dampings: Sequence[float] = (0.5, 0.7, 0.85, 0.95),
) -> list[dict[str, float]]:
    """Report convergence and ranking stability as the damping factor varies."""
    baseline: dict[str, float] | None = None
    rows: list[dict[str, float]] = []
    for damping in dampings:
        result = personalized_pagerank(graph, personalization=personalization, damping=damping)
        scores = result.scores
        overlap = 0.0
        if baseline is not None:
            top_a = {k for k, _ in sorted(baseline.items(), key=lambda kv: -kv[1])[:10]}
            top_b = {k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:10]}
            overlap = len(top_a & top_b) / max(1, len(top_a))
        else:
            baseline = scores
        rows.append(
            {
                "damping": float(damping),
                "iterations": float(result.iterations),
                "converged": float(result.converged),
                "total_mass": result.total_mass(),
                "top10_overlap_with_first": overlap,
                "final_residual": result.residuals[-1] if result.residuals else 0.0,
            }
        )
    return rows
