"""Retrieval quality metrics and the chronological evaluation split.

Fitting anything (SVD, thresholds, calibration) on the test period would leak
the future into the past, so the split is enforced here and asserted by tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from newstrace.models import Article
from newstrace.utils import ensure_utc


@dataclass
class ChronologicalSplit:
    """Fit / validation / test partition of article ids, ordered by time."""

    fit_ids: list[int]
    validation_ids: list[int]
    test_ids: list[int]
    fit_cutoff: datetime | None
    validation_cutoff: datetime | None

    def as_dict(self) -> dict[str, object]:
        return {
            "fit_count": len(self.fit_ids),
            "validation_count": len(self.validation_ids),
            "test_count": len(self.test_ids),
            "fit_cutoff": self.fit_cutoff.isoformat() if self.fit_cutoff else None,
            "validation_cutoff": (
                self.validation_cutoff.isoformat() if self.validation_cutoff else None
            ),
        }

    def assert_no_leakage(self) -> None:
        overlap = (set(self.fit_ids) | set(self.validation_ids)) & set(self.test_ids)
        if overlap:
            raise AssertionError(f"chronological split leaks {len(overlap)} test articles")


def chronological_split(
    articles: Sequence[Article], *, fit_fraction: float = 0.6, validation_fraction: float = 0.2
) -> ChronologicalSplit:
    """Split articles by publication time into 60 / 20 / 20 by default."""
    ordered = sorted(
        articles,
        key=lambda a: (ensure_utc(a.published_at) or datetime.min.replace(tzinfo=None), a.id),
    )
    n = len(ordered)
    if n == 0:
        return ChronologicalSplit([], [], [], None, None)
    fit_end = max(1, int(n * fit_fraction)) if n > 2 else n
    val_end = min(n, max(fit_end, int(n * (fit_fraction + validation_fraction))))
    fit = ordered[:fit_end]
    validation = ordered[fit_end:val_end]
    test = ordered[val_end:]
    return ChronologicalSplit(
        fit_ids=[a.id for a in fit],
        validation_ids=[a.id for a in validation],
        test_ids=[a.id for a in test],
        fit_cutoff=ensure_utc(fit[-1].published_at) if fit else None,
        validation_cutoff=ensure_utc(validation[-1].published_at) if validation else None,
    )


# -- ranking metrics --------------------------------------------------------


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(ranked)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / float(k)


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(list(ranked)[:k])
    return len(top & relevant) / float(len(relevant))


def reciprocal_rank(ranked: Sequence[int], relevant: set[int]) -> float:
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def ndcg_at_k(ranked: Sequence[int], relevance: Mapping[int, float] | set[int], k: int) -> float:
    rel_map = dict.fromkeys(relevance, 1.0) if isinstance(relevance, set) else dict(relevance)
    gains = [rel_map.get(item, 0.0) for item in list(ranked)[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    denominator = dcg(ideal)
    return (dcg(gains) / denominator) if denominator > 0 else 0.0


def topk_overlap(baseline: Sequence[int], candidate: Sequence[int], k: int) -> float:
    """Top-k neighbour overlap with the full-dimensional baseline."""
    a = set(list(baseline)[:k])
    b = set(list(candidate)[:k])
    return (len(a & b) / float(len(a))) if a else 0.0


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval metrics for one method/dimension."""

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    topk_overlap_at_10: float = 0.0
    queries: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "precision_at_10": round(self.precision_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "topk_overlap_at_10": round(self.topk_overlap_at_10, 4),
            "queries": float(self.queries),
        }


def evaluate_rankings(
    rankings: Mapping[str, Sequence[int]],
    judgments: Mapping[str, set[int]],
    *,
    baseline: Mapping[str, Sequence[int]] | None = None,
) -> tuple[RetrievalMetrics, list[dict[str, object]]]:
    """Aggregate metrics plus a per-query breakdown."""
    rows: list[dict[str, object]] = []
    for query, ranked in rankings.items():
        relevant = judgments.get(query, set())
        if not relevant:
            continue
        row = {
            "query": query,
            "recall_at_5": recall_at_k(ranked, relevant, 5),
            "recall_at_10": recall_at_k(ranked, relevant, 10),
            "precision_at_5": precision_at_k(ranked, relevant, 5),
            "precision_at_10": precision_at_k(ranked, relevant, 10),
            "mrr": reciprocal_rank(ranked, relevant),
            "ndcg_at_10": ndcg_at_k(ranked, relevant, 10),
            "topk_overlap_at_10": (
                topk_overlap(baseline[query], ranked, 10) if baseline and query in baseline else 1.0
            ),
            "relevant_count": len(relevant),
            "returned_count": len(ranked),
        }
        rows.append(row)

    if not rows:
        return RetrievalMetrics(), rows

    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows]))  # type: ignore[arg-type]

    metrics = RetrievalMetrics(
        recall_at_5=mean("recall_at_5"),
        recall_at_10=mean("recall_at_10"),
        precision_at_5=mean("precision_at_5"),
        precision_at_10=mean("precision_at_10"),
        mrr=mean("mrr"),
        ndcg_at_10=mean("ndcg_at_10"),
        topk_overlap_at_10=mean("topk_overlap_at_10"),
        queries=len(rows),
    )
    return metrics, rows


# -- statistical uncertainty ------------------------------------------------

BOOTSTRAP_METRICS = ("recall_at_10", "ndcg_at_10", "mrr", "topk_overlap_at_10")
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_CONFIDENCE = 0.95


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 549,
) -> dict[str, float]:
    """Percentile-bootstrap interval for the mean of per-query values.

    Seeded, so the interval is reproducible for a fixed per-query file.  A
    single observation has no spread to resample, so its interval collapses to
    the point estimate rather than pretending to precision it does not have.
    """
    array = np.asarray([float(v) for v in values], dtype=float)
    if array.size == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0.0}
    mean = float(array.mean())
    if array.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "n": 1.0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(int(iterations), array.size))
    means = array[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    return {
        "mean": mean,
        "lo": float(np.percentile(means, tail)),
        "hi": float(np.percentile(means, 100.0 - tail)),
        "n": float(array.size),
    }


def paired_bootstrap_delta_ci(
    values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 549,
) -> dict[str, float]:
    """Paired bootstrap for ``mean(values - baseline_values)``.

    Pairing matters: the same queries are easy or hard for every method, so the
    paired interval is far tighter than comparing two independent intervals, and
    it is the interval that answers "is this method worse than full?".
    ``crosses_zero`` is 1.0 when the interval admits no difference.
    """
    a = np.asarray([float(v) for v in values], dtype=float)
    b = np.asarray([float(v) for v in baseline_values], dtype=float)
    if a.size == 0 or a.size != b.size:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0.0, "crosses_zero": 1.0}
    diff = a - b
    interval = bootstrap_mean_ci(diff, iterations=iterations, confidence=confidence, seed=seed)
    interval["crosses_zero"] = float(interval["lo"] <= 0.0 <= interval["hi"])
    return interval


def bootstrap_intervals(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_rows: Sequence[Mapping[str, object]] | None = None,
    metrics: Sequence[str] = BOOTSTRAP_METRICS,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = 549,
) -> dict[str, dict[str, float]]:
    """Bootstrap intervals per metric, plus paired deltas against a baseline.

    ``rows`` are the per-query dicts returned by :func:`evaluate_rankings`.
    Deltas are computed only over queries present in both row sets.
    """
    out: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = [float(r[metric]) for r in rows if metric in r]  # type: ignore[arg-type]
        if not values:
            continue
        out[metric] = bootstrap_mean_ci(
            values, iterations=iterations, confidence=confidence, seed=seed
        )

    if baseline_rows is None:
        return out

    baseline_by_query = {str(r.get("query")): r for r in baseline_rows}
    for metric in metrics:
        paired = [
            (float(r[metric]), float(baseline_by_query[str(r.get("query"))][metric]))  # type: ignore[arg-type]
            for r in rows
            if metric in r
            and str(r.get("query")) in baseline_by_query
            and metric in baseline_by_query[str(r.get("query"))]
        ]
        if not paired:
            continue
        out[f"{metric}_delta"] = paired_bootstrap_delta_ci(
            [p[0] for p in paired],
            [p[1] for p in paired],
            iterations=iterations,
            confidence=confidence,
            seed=seed,
        )
    return out
