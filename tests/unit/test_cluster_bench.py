"""The clustering benchmark harness: loading the published splits and replaying them.

Exercised on a tiny synthetic stream written in the dataset's own layout -- the
point is the loader, the stream order and the metric plumbing, not the numbers.
No test here touches the network.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from newstrace.evaluation.cluster_bench import (
    ClusterBenchConfig,
    load_split,
    render_table,
    run_cluster_bench,
)

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
    for row in metrics["results"]:
        assert row["items"] == 12
        assert row["true_clusters"] == 3
        assert 0.0 <= row["pairwise_f1"] <= 1.0

    table = render_table(metrics)
    assert "NewsTrace, shipped" in table
    assert "Miranda et al.: + timestamps" in table


def test_no_tune_scores_only_the_shipped_threshold(bench_dir: Path) -> None:
    (bench_dir / "dataset.dev.json").unlink()
    metrics = run_cluster_bench(ClusterBenchConfig(data_dir=str(bench_dir), tune=False))
    assert [row["setting"] for row in metrics["results"]] == ["shipped"]
    assert metrics["dev_sweep"] == []
    assert metrics["dataset"]["dev"] is None
