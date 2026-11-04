"""The BEIR harness: loading a published dataset layout and sweeping over it.

The sweep itself is exercised on a tiny synthetic dataset written to a temp
directory -- the point is that the loader, the qrels parsing and the metric
plumbing are right. Numbers from six documents mean nothing, and no test here
touches the network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from newstrace.evaluation.beir import BeirConfig, load_dataset, render_table, run_beir
from newstrace.evaluation.retrieval import bootstrap_intervals, evaluate_rankings


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
    assert len(metrics["per_query"]) == len(metrics["results"]) * 4


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


@pytest.mark.parametrize("shared_text", [True, False])
def test_query_ids_survive_scoring_and_paired_intervals(tmp_path: Path, shared_text: bool) -> None:
    directory = _write_dataset(tmp_path, queries=2)
    texts = ["quantum error correction", "grid demand forecasting"]
    if shared_text:
        texts[1] = texts[0]
    query_ids = ["research/研究", "research/β"]
    (directory / "queries.jsonl").write_text(
        "".join(
            json.dumps({"_id": qid, "text": text}) + "\n"
            for qid, text in zip(query_ids, texts, strict=True)
        ),
        encoding="utf-8",
    )
    # Identical wording can have distinct relevance judgments. Neither ID may
    # overwrite the other, and every method must evaluate the same population.
    (directory / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        + "".join(f"{query_ids[0]}\td{i}\t1\n" for i in range(1, 9))
        + "".join(f"{query_ids[1]}\td{i}\t1\n" for i in range(9, 17)),
        encoding="utf-8",
    )
    metrics = run_beir(
        BeirConfig(
            dataset="mini",
            data_dir=str(tmp_path),
            methods=["full", "gaussian_rp"],
            dimensions=[8],
            top_k=1,
        )
    )
    assert metrics["dataset"]["queries"] == 2
    assert all(row["queries"] == 2 for row in metrics["results"])
    records = metrics["per_query"]
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)
    assert set(grouped) == {"full", "gaussian_rp", "tfidf"}
    baseline = None
    for result in metrics["results"]:
        rows = grouped[result["method"]]
        assert [r["query_id"] for r in rows] == query_ids
        assert [r["query_text"] for r in rows] == texts
        assert all(r["dimension"] == result["dimension"] for r in rows)
        assert all(r["relevant_count"] == 8 and r["returned_count"] == 1 for r in rows)
        assert result["ndcg_at_10"] == pytest.approx(
            round(sum(r["ndcg_at_10"] for r in rows) / 2, 4)
        )
        # The generic bootstrap joins on `query`; artifacts expose the same
        # identity explicitly as `query_id`, alongside the original text.
        bootstrap_rows = [{**r, "query": r["query_id"]} for r in rows]
        if result["method"] == "full":
            baseline = bootstrap_rows
        assert result["intervals"] == bootstrap_intervals(
            bootstrap_rows, baseline_rows=None if result["method"] == "full" else baseline
        )
        assert result["intervals"]["ndcg_at_10"]["n"] == 2
    if shared_text:
        # One relevant hit and one miss, rather than whichever qrel was last.
        assert sorted(r["recall_at_10"] for r in grouped["tfidf"]) == [0.0, 0.125]


def test_duplicate_judged_query_ids_fail_before_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from newstrace.evaluation import beir

    directory = _write_dataset(tmp_path)
    with (directory / "queries.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"_id": "q0", "text": "different text"}) + "\n")

    def unexpected_embedder(*args, **kwargs):
        pytest.fail("ambiguous query IDs must be rejected before obtaining an embedder")

    monkeypatch.setattr(beir, "get_embedder", unexpected_embedder)
    with pytest.raises(ValueError, match="duplicate judged query ID: 'q0'"):
        run_beir(BeirConfig(dataset="mini", data_dir=str(tmp_path)))


def test_cli_rejects_duplicate_query_ids_without_success_artifacts(tmp_path: Path) -> None:
    directory = _write_dataset(tmp_path)
    with (directory / "queries.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"_id": "q0", "text": "ambiguous query"}) + "\n")
    artifacts = tmp_path / "exported"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_beir.py",
            "--dataset",
            "mini",
            "--data-dir",
            str(tmp_path),
            "--artifacts-dir",
            str(artifacts),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "duplicate judged query ID" in completed.stdout
    assert "Traceback" not in completed.stdout + completed.stderr
    assert not artifacts.exists()


@pytest.mark.parametrize("max_queries", [1, 2])
def test_cli_exports_recomputable_query_records(tmp_path: Path, max_queries: int) -> None:
    directory = _write_dataset(tmp_path, queries=2)
    # Exercise the public CLI with colliding text and a selected query prefix.
    (directory / "queries.jsonl").write_text(
        "".join(
            json.dumps({"_id": qid, "text": "quantum error correction"}) + "\n"
            for qid in ("q0", "q1")
        ),
        encoding="utf-8",
    )
    artifacts = tmp_path / "exported"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_beir.py",
            "--dataset",
            "mini",
            "--data-dir",
            str(tmp_path),
            "--methods",
            "full,gaussian_rp",
            "--dimensions",
            "8",
            "--max-queries",
            str(max_queries),
            "--top-k",
            "5",
            "--artifacts-dir",
            str(artifacts),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    outdir = next((artifacts / "beir").iterdir())
    metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (outdir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "per_query" not in metrics  # detailed records live in the JSONL file
    assert metrics["dataset"]["queries"] == max_queries
    assert len(records) == len(metrics["results"]) * max_queries
    baseline_rankings = None
    baseline_rows = None
    for result in metrics["results"]:
        rows = [
            r
            for r in records
            if (r["method"], r["dimension"]) == (result["method"], result["dimension"])
        ]
        assert [r["query_id"] for r in rows] == ["q0", "q1"][:max_queries]
        # Restore integer rows from the original document IDs in the artifact.
        rankings = {r["query_id"]: [int(doc[1:]) for doc in r["ranked_document_ids"]] for r in rows}
        judgments = {
            r["query_id"]: {int(doc[1:]) for doc in r["relevant_document_ids"]} for r in rows
        }
        rescored, rescored_rows = evaluate_rankings(rankings, judgments, baseline=baseline_rankings)
        assert all(result[key] == value for key, value in rescored.as_dict().items())
        for saved, rescored_row in zip(rows, rescored_rows, strict=True):
            assert saved["query_text"] == "quantum error correction"
            assert all(saved[key] == value for key, value in rescored_row.items() if key != "query")
        assert result["intervals"] == bootstrap_intervals(
            rescored_rows,
            baseline_rows=baseline_rows,
        )
        if result["method"] == "full":
            baseline_rankings, baseline_rows = rankings, rescored_rows
    table = (outdir / "results.md").read_text(encoding="utf-8")
    assert "| queries |" in table
    assert "per_query.jsonl" in table
