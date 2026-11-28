"""Exercise saved-run exploration without executing an experiment."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from newstrace.config import get_settings
from tests.unit.test_tradeoff_reports import rows


@pytest.fixture
def saved_runs(session):
    root = get_settings().resolve(get_settings().artifacts_dir) / "experiments"
    for name, results in (
        (
            "2026-11-28",
            [
                *rows(),
                {
                    "method": "rp",
                    "dimension": 64,
                    "retrieval_ndcg_at_10": 0.9,
                    "system_index_memory_mib": 6.0,
                    "system_p95_query_latency_ms": 3.0,
                },
            ],
        ),
        (
            "2026-11-27",
            [
                {
                    "method": "full",
                    "dimension": 384,
                    "retrieval_ndcg_at_10": 0.7,
                    "system_index_memory_mib": 0.01,
                    "system_p95_query_latency_ms": 0.02,
                }
            ],
        ),
        ("2026-11-26", [{"method": "incomplete", "dimension": 16}]),
    ):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({"results": results}))
    return root


def app_for_experiments():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    path = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
    return AppTest.from_string(
        f"import runpy\nrunpy.run_path({str(path)!r})['page_experiments']()",
        default_timeout=30,
    ).run()


def widget(app, kind, label):
    return next(w for w in getattr(app, kind) if w.label == label)


def test_dashboard_budgets_frontier_empty_state_and_csv(saved_runs, monkeypatch):
    streamlit = pytest.importorskip("streamlit")
    captured = []
    original = streamlit.download_button

    def download(label, data, *args, **kwargs):
        captured.append(data)
        return original(label, data, *args, **kwargs)

    monkeypatch.setattr(streamlit, "download_button", download)
    before = {p: p.read_bytes() for p in saved_runs.glob("*/metrics.json")}
    app = app_for_experiments()
    assert not app.exception
    assert set(app.dataframe[0].value["result_row"]) == {2, 4}
    assert "2 matching configurations · 3 comparable · 1 excluded" in [c.value for c in app.caption]
    assert len(app.get("vega_lite_chart")) == 1
    assert any("Result row 3: index memory" in m.value for m in app.markdown)
    assert any("No uncertainty intervals" in i.value for i in app.info)
    assert list(pd.read_csv(io.StringIO(captured[-1]))["result_row"]) == [2, 4]

    widget(app, "checkbox", "Only Pareto frontier").uncheck().run()
    assert set(app.dataframe[0].value["result_row"]) == {1, 2, 4}
    widget(app, "number_input", "Maximum index memory (MiB)").set_value(2.0).run()
    assert list(app.dataframe[0].value["result_row"]) == [2]
    widget(app, "number_input", "Minimum nDCG at 10").set_value(0.85).run()
    assert not app.exception
    assert any("No configurations match these budgets" in i.value for i in app.info)
    assert not app.get("vega_lite_chart")
    assert not app.get("download_button")
    # The raw measured table remains inspectable when the shortlist is empty.
    assert len(app.dataframe[0].value) == 4
    widget(app, "number_input", "Maximum index memory (MiB)").set_value(10.0).run()
    widget(app, "selectbox", "Compare quality against").set_value("p95 latency (ms)").run()
    assert not app.exception
    chart = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    assert chart["encoding"]["x"]["field"] == "p95 latency (ms)"
    assert list(app.dataframe[0].value["result_row"]) == [4]
    assert {p: p.read_bytes() for p in before} == before


def test_switching_runs_resets_limits_and_explains_incomplete_measurements(saved_runs):
    app = app_for_experiments()
    widget(app, "number_input", "Minimum nDCG at 10").set_value(0.95).run()
    widget(app, "selectbox", "Experiment").set_value(saved_runs / "2026-11-27").run()
    assert not app.exception
    assert widget(app, "number_input", "Minimum nDCG at 10").value == 0.0
    assert widget(app, "number_input", "Maximum index memory (MiB)").value == 0.01
    assert len(app.dataframe[0].value) == 1
    widget(app, "selectbox", "Experiment").set_value(saved_runs / "2026-11-26").run()
    assert not app.exception
    assert any("No configurations have complete, valid" in i.value for i in app.info)
    assert not app.get("vega_lite_chart")
    assert len(app.dataframe[0].value) == 1


def test_saved_intervals_are_visible_without_recomputation(saved_runs):
    path = saved_runs / "2026-11-28" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["results"][0].update({"ci_ndcg_at_10_lo": 0.6, "ci_ndcg_at_10_hi": 0.95})
    path.write_text(json.dumps(metrics))
    before = path.read_bytes()
    app = app_for_experiments()
    assert not app.exception
    intervals = next(d.value for d in app.dataframe if "ci_ndcg_at_10_lo" in d.value.columns)
    assert intervals.iloc[0]["ci_ndcg_at_10_lo"] == 0.6
    assert intervals.iloc[0]["ci_ndcg_at_10_hi"] == 0.95
    assert path.read_bytes() == before
