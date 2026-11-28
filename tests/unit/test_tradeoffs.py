"""Tradeoff selection is a deterministic view of measured rows."""

from __future__ import annotations

from copy import deepcopy
from itertools import permutations

import pytest

from newstrace.evaluation.tradeoffs import analyze_tradeoffs


def measurement(method="full", quality=0.8, memory=10.0, latency=2.0, dimension=384):
    return {
        "method": method,
        "dimension": dimension,
        "retrieval_ndcg_at_10": quality,
        "system_index_memory_mib": memory,
        "system_p95_query_latency_ms": latency,
    }


def test_frontier_requires_all_three_objectives_and_one_strict_improvement():
    rows = [
        measurement(),
        measurement("svd", 0.8, 5.0, 2.0),  # dominates full
        measurement("svd", 0.8, 5.0, 2.0),  # equal points survive
        measurement("rp", 0.9, 6.0, 3.0),  # higher quality, higher costs
        measurement("fast", 0.7, 2.0, 1.0),
        measurement("slow", 0.7, 2.0, 4.0),
        measurement("zero", 0.0, 1.0, 0.5),  # zero quality remains valid
    ]
    saved = deepcopy(rows)
    analysis = analyze_tradeoffs(rows)
    assert [p.row for p in analysis.points if p.frontier] == [2, 3, 4, 5, 7]
    assert not analysis.excluded
    assert rows == saved
    # Inclusive bounds preserve repeated configurations' distinct row identities.
    chosen = analysis.shortlist(min_quality=0.8, max_memory_mib=5.0, max_latency_ms=2.0)
    assert [p.row for p in chosen] == [2, 3]
    assert [p.as_dict()["result_row"] for p in chosen] == [2, 3]
    assert len(analysis.shortlist()) == 7
    assert len(analysis.shortlist(frontier_only=True)) == 5
    assert analysis.shortlist(max_memory_mib=0.0) == ()
    assert analysis.shortlist(min_quality=1.0) == ()


def test_frontier_is_independent_of_input_order():
    rows = [measurement(), measurement("svd", 0.8, 4.0, 2.0), measurement("rp", 0.9, 5.0, 3.0)]
    for order in permutations(rows):
        assert {p.method for p in analyze_tradeoffs(order).points if p.frontier} == {"svd", "rp"}


def test_identical_points_can_both_be_dominated_by_another_configuration():
    analysis = analyze_tradeoffs([measurement(), measurement(), measurement("better", 0.9)])
    assert [p.row for p in analysis.points if p.frontier] == [3]


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("method", "", "missing method"),
        ("dimension", 2.5, "dimension"),
        ("dimension", 0, "dimension"),
        ("dimension", 10**1000, "dimension"),
        ("retrieval_ndcg_at_10", None, "nDCG"),
        ("retrieval_ndcg_at_10", -0.1, "nDCG"),
        ("retrieval_ndcg_at_10", 1.1, "nDCG"),
        ("retrieval_ndcg_at_10", float("nan"), "nDCG"),
        ("retrieval_ndcg_at_10", True, "nDCG"),
        ("system_index_memory_mib", 0.0, "memory"),
        ("system_index_memory_mib", "2.0", "memory"),
        ("system_index_memory_mib", float("inf"), "memory"),
        ("system_p95_query_latency_ms", None, "latency"),
        ("system_p95_query_latency_ms", -1, "latency"),
        ("system_p95_query_latency_ms", 0, "latency"),
    ],
)
def test_incomplete_or_invalid_measurements_are_explained(key, value, reason):
    row = measurement()
    row[key] = value
    analysis = analyze_tradeoffs([row, measurement("valid")])
    assert [p.row for p in analysis.points] == [2]
    assert analysis.excluded[0].row == 1
    assert reason in analysis.excluded[0].reason


@pytest.mark.parametrize(
    "budgets",
    [
        {"min_quality": float("nan")},
        {"min_quality": -1},
        {"min_quality": 1.1},
        {"max_memory_mib": float("inf")},
        {"max_memory_mib": -1},
        {"max_latency_ms": float("nan")},
        {"max_latency_ms": True},
    ],
)
def test_invalid_budgets_are_rejected_even_for_empty_runs(budgets):
    with pytest.raises(ValueError):
        analyze_tradeoffs([]).shortlist(**budgets)


def test_missing_fields_and_empty_runs():
    analysis = analyze_tradeoffs([{}])
    assert analysis.points == ()
    assert len(analysis.excluded[0].reason.split("; ")) == 5
    assert analyze_tradeoffs([]).shortlist() == ()
