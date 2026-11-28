"""Saved reports must not invent cost ratios from absent measurements."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from newstrace.evaluation import report


def rows():
    return [
        {
            "method": "full",
            "dimension": 384,
            "retrieval_ndcg_at_10": 0.8,
            "system_index_memory_mib": 10.0,
            "system_p95_query_latency_ms": 2.0,
        },
        {
            "method": "svd",
            "dimension": 32,
            "retrieval_ndcg_at_10": 0.8,
            "system_index_memory_mib": 2.0,
            "system_p95_query_latency_ms": 1.0,
        },
        {"method": "missing", "dimension": 16, "retrieval_ndcg_at_10": 0.6},
    ]


def test_report_explains_frontier_and_missing_costs_without_fabricated_ratios(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(report, "make_plots", lambda *_: [])
    metrics = {"results": rows()}
    source = tmp_path / "metrics.json"
    source.write_text(json.dumps(metrics))
    before = source.read_bytes()
    text = report.write_report(tmp_path, metrics).read_text()
    assert "0.00×" not in text
    assert "Pareto frontier" in text
    assert "| 1 | full | 384 | 0.8000 | 10.0000 | 2.0000 | False |" in text
    assert "| 2 | svd | 32 | 0.8000 | 2.0000 | 1.0000 | True |" in text
    assert "Result row 3: index memory" in text
    assert "not a statistical significance test" in text
    assert source.read_bytes() == before


@pytest.mark.parametrize("results", [[], [{"method": "full", "dimension": 384}]])
def test_report_handles_empty_and_legacy_incomplete_measurements(tmp_path, monkeypatch, results):
    monkeypatch.setattr(report, "make_plots", lambda *_: [])
    text = report.write_report(tmp_path, {"results": results}).read_text()
    assert "No configurations have complete, valid quality and positive cost measurements" in text
    assert "Only the full-dimensional baseline was run" not in text


def test_frontier_does_not_require_or_invent_a_full_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "make_plots", lambda *_: [])
    text = report.write_report(tmp_path, {"results": rows()[1:2]}).read_text()
    assert "`full` was omitted, so comparisons against it are unavailable" in text
    assert "| 1 | svd | 32 | 0.8000 | 2.0000 | 1.0000 | True |" in text


def test_plots_omit_invalid_coordinates_without_inventing_zeroes(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.axes

    plotted = []
    original = matplotlib.axes.Axes.plot

    def capture(self, x, y, **kwargs):
        plotted.append((list(x), list(y)))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture)
    invalid = {
        "method": "bad",
        "dimension": 4,
        "retrieval_ndcg_at_10": float("nan"),
        "system_index_memory_mib": None,
        "system_p95_query_latency_ms": float("inf"),
    }
    files = report.make_plots(tmp_path, {"results": [*rows(), invalid]})
    assert set(files) == {p[0] for p in report.PLOTS}
    assert all((tmp_path / filename).stat().st_size > 1000 for filename in files)
    assert ([16], [0.6]) in plotted  # quality/dimension remains measured without costs
    assert not any(0.0 in x for x, _ in plotted)


def test_committed_measurements_can_render_without_changing_metrics(tmp_path):
    root = Path(__file__).resolve().parents[2] / "docs" / "experiments" / "baseline"
    metrics = report.load_metrics(root / "metrics.json")
    before = json.dumps(metrics, sort_keys=True)
    text = report.write_report(tmp_path, metrics).read_text()
    assert "Pareto frontier" in text
    assert "0.00×" not in text
    assert json.dumps(metrics, sort_keys=True) == before
