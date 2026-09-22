"""The BEIR harness: loading a published dataset layout and sweeping over it.

The sweep itself is exercised on a tiny synthetic dataset written to a temp
directory -- the point is that the loader, the qrels parsing and the metric
plumbing are right. Numbers from six documents mean nothing, and no test here
touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from newstrace.evaluation.beir import BeirConfig, load_dataset, render_table, run_beir


def _write_dataset(root: Path, *, name: str = "mini", queries: int = 4) -> Path:
    directory = root / name
    (directory / "qrels").mkdir(parents=True, exist_ok=True)
    topics = [
        "quantum error correction",
        "grid demand forecasting",
        "agent benchmark scores",
        "protein folding accuracy",
    ][:queries]

    documents = []
    judgments = []
    doc_id = 0
    for index, topic in enumerate(topics):
        for variant in range(8):
            doc_id += 1
            documents.append(
                {
                    "_id": f"d{doc_id}",
                    "title": f"{topic} study {variant}",
                    "text": f"A report on {topic}. Finding {variant} concerns {topic} directly.",
                }
            )
            if variant < 3:
                judgments.append((f"q{index}", f"d{doc_id}", 1 if variant else 2))

    with (directory / "corpus.jsonl").open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document) + "\n")
    with (directory / "queries.jsonl").open("w", encoding="utf-8") as handle:
        for index, topic in enumerate(topics):
            handle.write(json.dumps({"_id": f"q{index}", "text": f"evidence about {topic}"}) + "\n")
        # An unjudged query must be dropped rather than counted as a miss.
        handle.write(json.dumps({"_id": "q999", "text": "unjudged query"}) + "\n")
    with (directory / "qrels" / "test.tsv").open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for query_id, document_id, score in judgments:
            handle.write(f"{query_id}\t{document_id}\t{score}\n")
    return directory


def test_loader_reads_the_published_layout(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    dataset = load_dataset(BeirConfig(dataset="mini", data_dir=str(tmp_path)))
    assert len(dataset) == 32
    assert dataset.query_ids == ["q0", "q1", "q2", "q3"]
    assert "q999" not in dataset.qrels
    assert all(len(v) == 3 for v in dataset.qrels.values())
    assert dataset.documents[0].startswith("quantum error correction study 0.")


def test_loader_reports_a_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="--download"):
        load_dataset(BeirConfig(dataset="absent", data_dir=str(tmp_path)))


def test_qrels_without_a_header_are_read(tmp_path: Path) -> None:
    directory = _write_dataset(tmp_path)
    path = directory / "qrels" / "test.tsv"
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = load_dataset(BeirConfig(dataset="mini", data_dir=str(tmp_path)))
    assert dataset.qrels


def test_trimming_the_corpus_keeps_every_judged_document(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    config = BeirConfig(dataset="mini", data_dir=str(tmp_path), max_documents=5)
    dataset = load_dataset(config)
    judged = {doc for query in dataset.qrels.values() for doc in query}
    assert judged <= set(dataset.doc_ids)
    assert len(dataset) < 32


def test_sweep_produces_one_row_per_configuration(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    config = BeirConfig(
        dataset="mini",
        data_dir=str(tmp_path),
        methods=["full", "svd"],
        dimensions=[8],
        top_k=5,
    )
    metrics = run_beir(config)
    methods = [(row["method"], row["dimension"]) for row in metrics["results"]]
    assert ("full", metrics["results"][0]["dimension"]) in methods
    assert any(method == "svd" for method, _ in methods)
    assert any(method == "tfidf" for method, _ in methods)
    assert metrics["dataset"]["queries"] == 4
    assert metrics["dataset"]["judgment_source"].startswith("human")

    for row in metrics["results"]:
        assert 0.0 <= row["ndcg_at_10"] <= 1.0
        assert row["index_memory_mib"] >= 0.0
    # Every non-baseline row carries a paired interval against `full`.
    for row in metrics["results"][1:]:
        assert "ndcg_at_10_delta" in row["intervals"]


def test_render_table_includes_the_intervals(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    metrics = run_beir(
        BeirConfig(
            dataset="mini",
            data_dir=str(tmp_path),
            methods=["full", "svd"],
            dimensions=[8],
            include_tfidf=False,
        )
    )
    table = render_table(metrics)
    assert "BEIR: mini" in table
    assert "nDCG@10" in table
    assert "baseline" in table
