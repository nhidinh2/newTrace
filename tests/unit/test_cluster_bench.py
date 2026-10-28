"""The clustering benchmark harness: loading the published splits and replaying them.

Exercised on a tiny synthetic stream written in the dataset's own layout -- the
point is the loader, the stream order and the metric plumbing, not the numbers.
No test here touches the network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from newstrace.config import get_settings
from newstrace.evaluation.cluster_bench import (
    ClusterBenchConfig,
    _embed,
    fit_weights,
    load_split,
    oracle_signals,
    render_table,
    run_cluster_bench,
)
from newstrace.representations.embedder import get_embedder

START = datetime(2015, 3, 1, 8, 0, tzinfo=UTC)
STORIES = [
    ("Volcano erupts near Reykjavik", "lava ash flights grounded iceland airport volcano"),
    ("Central bank raises interest rates", "inflation lending rates bank governor economy"),
    ("Champions league final tickets sold out", "football final stadium supporters tickets"),
]


def _write_split(root: Path, split: str, *, per_story: int = 4) -> None:
    records = []
    for story, (title, words) in enumerate(STORIES):
        for variant in range(per_story):
            records.append(
                {
                    "id": f"{split}-{story}-{variant}",
                    "title": f"{title} report {variant}",
                    "text": f"{title}. " + " ".join([words] * 5),
                    # Interleaved in time, and written out of order, so the loader
                    # has to restore stream order itself.
                    "date": (START + timedelta(hours=variant * 3 + story)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "source": f"outlet{variant}",
                    "lang": "eng",
                    "cluster": str(100 + story),
                    "duplicate": "False",
                }
            )
    records.append({**records[0], "id": f"{split}-de", "lang": "deu", "cluster": "999"})
    records.reverse()
    (root / f"dataset.{split}.json").write_text(json.dumps(records), encoding="utf-8")


@pytest.fixture
def bench_dir(tmp_path: Path) -> Path:
    for split in ("dev", "test"):
        _write_split(tmp_path, split)
    return tmp_path


def test_loader_filters_language_and_restores_stream_order(bench_dir: Path) -> None:
    documents = load_split(ClusterBenchConfig(data_dir=str(bench_dir)), "test")
    assert len(documents) == 12
    assert all(not d.doc_id.endswith("-de") for d in documents)
    times = [d.published_at for d in documents]
    assert times == sorted(times)
    assert documents[0].published_at.tzinfo is not None


def test_max_documents_takes_a_prefix_of_the_stream(bench_dir: Path) -> None:
    documents = load_split(ClusterBenchConfig(data_dir=str(bench_dir), max_documents=5), "test")
    full = load_split(ClusterBenchConfig(data_dir=str(bench_dir)), "test")
    assert [d.doc_id for d in documents] == [d.doc_id for d in full[:5]]


def test_loader_reports_a_missing_split(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--download"):
        load_split(ClusterBenchConfig(data_dir=str(tmp_path)), "test")


def test_run_tunes_on_dev_and_scores_test(bench_dir: Path) -> None:
    config = ClusterBenchConfig(data_dir=str(bench_dir), thresholds=[0.3, 0.62, 0.95])
    metrics = run_cluster_bench(config)

    assert [row["threshold"] for row in metrics["dev_sweep"]] == [0.3, 0.62, 0.95]
    tuned = metrics["clusterer"]["dev_tuned_threshold"]
    best_dev = max(row["pairwise_f1"] for row in metrics["dev_sweep"])
    assert (
        next(r for r in metrics["dev_sweep"] if r["threshold"] == tuned)["pairwise_f1"] == best_dev
    )

    settings = {row["setting"] for row in metrics["results"]}
    assert "shipped" in settings
    assert {"dev-learned", "dev-learned-time"} <= settings
    learned = metrics["clusterer"]["learned_weights"]
    row = next(r for r in metrics["results"] if r["setting"] == "dev-learned")
    assert row["title_weight"] == pytest.approx(learned["title_weight"], abs=1e-4)
    assert row["entity_weight"] == pytest.approx(learned["entity_weight"], abs=1e-4)
    assert row["time_weight"] == pytest.approx(learned["time_weight"], abs=1e-4)
    # The time-only variant keeps the shipped overlap weights.
    row = next(r for r in metrics["results"] if r["setting"] == "dev-learned-time")
    assert (row["title_weight"], row["entity_weight"]) == (0.2, 0.1)
    assert row["time_weight"] == pytest.approx(learned["time_weight"], abs=1e-4)
    for sweep in metrics["dev_sweep_learned"].values():
        assert [r["threshold"] for r in sweep] == [0.3, 0.62, 0.95]
    for row in metrics["results"]:
        assert row["items"] == 12
        assert row["true_clusters"] == 3
        assert 0.0 <= row["pairwise_f1"] <= 1.0

    table = render_table(metrics)
    assert "NewsTrace, shipped" in table
    assert "dev-learned-time" in table
    assert "Miranda et al.: + timestamps" in table


def test_no_tune_scores_only_the_shipped_threshold(bench_dir: Path) -> None:
    (bench_dir / "dataset.dev.json").unlink()
    metrics = run_cluster_bench(ClusterBenchConfig(data_dir=str(bench_dir), tune=False))
    assert [row["setting"] for row in metrics["results"]] == ["shipped"]
    assert metrics["dev_sweep"] == []
    assert metrics["dataset"]["dev"] is None


def test_benchmark_rejects_incomplete_cluster_output(
    bench_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "newstrace.evaluation.cluster_bench.cluster_stream",
        lambda *args, **kwargs: (dict.fromkeys(range(1, 12), 0), 1.0),
    )
    with pytest.raises(ValueError, match="Missing predictions for 1 of 12 labelled items"):
        run_cluster_bench(ClusterBenchConfig(data_dir=str(bench_dir), tune=False))


def test_oracle_signals_label_each_article_against_its_own_story(bench_dir: Path) -> None:
    settings = get_settings()
    documents = load_split(ClusterBenchConfig(data_dir=str(bench_dir)), "dev")
    vectors = _embed(documents, get_embedder(settings))
    signals, labels = oracle_signals(documents, vectors, settings=settings)

    assert signals.shape == (labels.size, 4)
    # Every article after the first of its story matches exactly one candidate:
    # 12 articles in 3 stories, all inside one window.
    assert int(labels.sum()) == 12 - 3
    assert signals[labels, 0].mean() > signals[~labels, 0].mean()


def test_fit_weights_recovers_the_blend_that_generated_the_labels() -> None:
    rng = np.random.default_rng(0)
    signals = rng.uniform(0.0, 1.0, size=(20_000, 4))
    signals[:, 3] *= 3.0  # days
    blend = signals[:, 0] + 0.5 * signals[:, 1] + 0.25 * signals[:, 2] - 0.1 * signals[:, 3]
    labels = rng.uniform(size=blend.size) < 1.0 / (1.0 + np.exp(-12.0 * (blend - 0.9)))

    fitted = fit_weights(signals, labels)
    assert fitted.title_weight == pytest.approx(0.5, abs=0.05)
    assert fitted.entity_weight == pytest.approx(0.25, abs=0.05)
    assert fitted.time_weight == pytest.approx(0.1, abs=0.03)
    assert fitted.implied_threshold == pytest.approx(0.9, abs=0.05)


def test_fit_weights_needs_both_classes() -> None:
    signals = np.ones((4, 4))
    with pytest.raises(ValueError, match="both"):
        fit_weights(signals, np.zeros(4, dtype=bool))
