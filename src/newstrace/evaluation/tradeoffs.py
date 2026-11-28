"""Shortlist saved measurements without rerunning or imputing an experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Real

TRADEOFF_NOTE = (
    "The frontier uses observed mean nDCG@10, index memory and p95 query latency within "
    "this saved run. A configuration is dominated when another has at least as much quality, "
    "no more memory and no more latency, with one strict improvement. Equal points do not "
    "dominate each other. This is not a statistical significance test or a deployment "
    "recommendation; "
    "inspect the paired intervals, query count and judgment source before choosing. "
    "Costs describe the measured index and query stage, not total application resources."
)


def finite_number(value: object) -> float | None:
    """Accept real stored measurements, excluding booleans, strings and nonfinite values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class TradeoffPoint:
    row: int
    method: str
    dimension: int
    quality: float
    memory_mib: float
    latency_ms: float
    frontier: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "result_row": self.row,
            "method": self.method,
            "dimension": self.dimension,
            "retrieval_ndcg_at_10": self.quality,
            "system_index_memory_mib": self.memory_mib,
            "system_p95_query_latency_ms": self.latency_ms,
            "frontier": self.frontier,
        }


@dataclass(frozen=True)
class ExcludedMeasurement:
    row: int
    reason: str


@dataclass(frozen=True)
class TradeoffAnalysis:
    points: tuple[TradeoffPoint, ...]
    excluded: tuple[ExcludedMeasurement, ...]

    def shortlist(
        self,
        *,
        min_quality: float = 0.0,
        max_memory_mib: float | None = None,
        max_latency_ms: float | None = None,
        frontier_only: bool = False,
    ) -> tuple[TradeoffPoint, ...]:
        """Inclusive budgets; frontier membership always refers to the full run."""
        quality = finite_number(min_quality)
        if quality is None or not 0 <= quality <= 1:
            raise ValueError("minimum nDCG@10 must be finite and between 0 and 1")
        for value in (max_memory_mib, max_latency_ms):
            if value is not None and (finite_number(value) is None or value < 0):
                raise ValueError("resource budgets must be finite and nonnegative")
        return tuple(
            p
            for p in self.points
            if p.quality >= quality
            and (max_memory_mib is None or p.memory_mib <= max_memory_mib)
            and (max_latency_ms is None or p.latency_ms <= max_latency_ms)
            and (not frontier_only or p.frontier)
        )


def analyze_tradeoffs(rows: Sequence[Mapping[str, object]]) -> TradeoffAnalysis:
    """Use only finite, valid quality and strictly positive measured costs.

    Row numbers are one-based positions in ``results``, preserving identity even
    for repeated method/dimension pairs. Missing/zero costs are not free resources.
    No measurements, confidence intervals or historical artifacts are rewritten.
    """
    points: list[TradeoffPoint] = []
    excluded: list[ExcludedMeasurement] = []
    for row_number, row in enumerate(rows, start=1):
        reasons: list[str] = []
        method = row.get("method")
        dimension = finite_number(row.get("dimension"))
        quality = finite_number(row.get("retrieval_ndcg_at_10"))
        memory = finite_number(row.get("system_index_memory_mib"))
        latency = finite_number(row.get("system_p95_query_latency_ms"))
        if not isinstance(method, str) or not method.strip():
            reasons.append("missing method")
        if dimension is None or dimension <= 0 or not dimension.is_integer():
            reasons.append("dimension must be a positive integer")
        if quality is None or not 0 <= quality <= 1:
            reasons.append("nDCG@10 must be finite and between 0 and 1")
        if memory is None or memory <= 0:
            reasons.append("index memory must be finite and positive")
        if latency is None or latency <= 0:
            reasons.append("p95 latency must be finite and positive")
        if reasons:
            excluded.append(ExcludedMeasurement(row_number, "; ".join(reasons)))
            continue
        assert isinstance(method, str)
        assert dimension is not None and quality is not None
        assert memory is not None and latency is not None
        points.append(TradeoffPoint(row_number, method, int(dimension), quality, memory, latency))

    def dominated(point: TradeoffPoint) -> bool:
        return any(
            other.quality >= point.quality
            and other.memory_mib <= point.memory_mib
            and other.latency_ms <= point.latency_ms
            and (
                other.quality > point.quality
                or other.memory_mib < point.memory_mib
                or other.latency_ms < point.latency_ms
            )
            for other in points
        )

    return TradeoffAnalysis(
        tuple(replace(point, frontier=not dominated(point)) for point in points), tuple(excluded)
    )
