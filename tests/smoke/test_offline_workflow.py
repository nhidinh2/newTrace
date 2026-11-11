"""Offline smoke test: the complete fixture-backed workflow, no network, no keys."""

from __future__ import annotations

import csv
import json
import socket
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy.orm import Session

from newstrace.db import session_scope
from newstrace.evaluation.systems import ExperimentConfig, run_experiment
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.models import Article, Claim, EvaluationRun, StoryCluster
from newstrace.search import SearchFilters, SearchRequest, search
from newstrace.stories import build_timeline, list_stories, load_story
from newstrace.summarization.extractive import ExtractiveSummarizer


@pytest.fixture
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound socket attempt fail loudly."""

    def guard(*args: object, **kwargs: object) -> None:
        raise AssertionError("the offline workflow must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


def test_offline_fixture_workflow(session: Session, _no_network: None) -> None:
    run, result = run_ingestion(session, IngestOptions(source="fixtures", index=True))
    assert run.status.startswith("completed")
    assert result.inserted > 50
    assert result.duplicates > 0

    articles = session.query(Article).count()
    stories = list_stories(session, limit=200)
    assert articles == result.inserted + result.duplicates
    assert stories

    # Duplicates are stored but never counted as independent sources.
    multi = [s for s in stories if s.article_count > 1]
    assert multi
    for story in multi:
        bundle = load_story(session, story.id)
        assert bundle is not None
        assert story.distinct_domain_count == len(bundle.independent_domains)
        assert len(bundle.articles) >= len(bundle.independent)

    # Timelines are ordered and mark exactly one earliest available report.
    bundle = load_story(session, multi[0].id)
    assert bundle is not None
    entries = build_timeline(bundle)
    stamps = [e.published_at for e in entries if e.published_at]
    assert stamps == sorted(stamps)
    assert sum(1 for e in entries if e.is_first_report) == 1

    # Every summary statement is grounded in retrieved evidence.
    summary = ExtractiveSummarizer(session).summarize_bundle(bundle)
    assert summary.statements
    assert summary.validate_citations() == []
    assert summary.citation_coverage() == 1.0

    # Search returns provenance and an explanation for each hit.
    response = search(
        session,
        SearchRequest(query="agent benchmark score", filters=SearchFilters(), top_k=5),
    )
    assert response.items
    for item in response.items:
        assert item.article.url
        assert item.explanation
        assert item.excerpt

    assert session.query(Claim).count() > 0
    assert session.query(StoryCluster).count() > 1


@pytest.mark.parametrize("full_position", [0, 2, None], ids=["full-first", "full-last", "no-full"])
def test_offline_experiment_writes_reproducible_artifacts(
    session: Session, tmp_path: Path, _no_network: None, full_position: int | None
) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=True))
    methods = ["gaussian_rp", "sparse_rp"]
    expected_rows = ["gaussian_rp", "gaussian_rp", "sparse_rp", "sparse_rp"]
    if full_position is not None:
        methods.insert(full_position, "full")
        expected_rows.insert(full_position * 2, "full")
    config = ExperimentConfig(
        methods=methods,
        dimensions=[16, 32],
        seed=549,
        max_queries=8,
        latency_repeats=1,
        include_pagerank=True,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    run = run_experiment(session, config)
    assert run.status == "completed", run.errors
    assert isinstance(run, EvaluationRun)

    outdir = Path(run.artifact_path or "")
    assert (outdir / "metrics.json").exists()
    assert (outdir / "config.json").exists()
    assert (outdir / "environment.json").exists()
    assert (outdir / "report.md").exists()

    metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert [row["method"] for row in metrics["results"]] == expected_rows
    assert json.loads((outdir / "config.json").read_text(encoding="utf-8"))["methods"] == methods
    assert metrics["split"]["test_count"] > 0
    assert metrics["judgment_source"] == "synthetic_fixture_labels"

    # The reported query count must be the number of queries actually scored:
    # duplicate query texts collapse when rankings are keyed by text.
    assert metrics["query_count"] == metrics["results"][0]["retrieval_queries"]

    parquet = outdir / "per_query_metrics.parquet"
    if parquet.exists():
        per_query = pd.read_parquet(parquet).to_dict(orient="records")
    else:
        with (outdir / "per_query_metrics.csv").open(encoding="utf-8", newline="") as handle:
            per_query = list(csv.DictReader(handle))
    assert [row["method"] for row in per_query] == [
        method for method in expected_rows for _ in range(metrics["query_count"])
    ]
    assert all(("topk_overlap_at_10" in row) == (full_position is not None) for row in per_query)

    # Every reported metric carries a bootstrap interval that brackets it.
    for row in metrics["results"]:
        lo, hi = row["ci_ndcg_at_10_lo"], row["ci_ndcg_at_10_hi"]
        assert lo <= row["retrieval_ndcg_at_10"] <= hi
        assert ("retrieval_topk_overlap_at_10" in row) == (full_position is not None)
        if full_position is not None and row["method"] != "full":
            assert "ci_ndcg_at_10_delta_lo" in row, "non-baseline rows need a paired delta"
        else:
            assert "ci_ndcg_at_10_delta_lo" not in row
    if full_position is not None:
        assert any(row["retrieval_topk_overlap_at_10"] < 1.0 for row in metrics["results"])

    environment = json.loads((outdir / "environment.json").read_text(encoding="utf-8"))
    assert environment["random_seed"] == 549
    assert environment["packages"]["numpy"]

    report = (outdir / "report.md").read_text(encoding="utf-8")
    assert "not a fact checker" in report
    assert "Chronological split" in report
    assert "separable from full?" in report, "the report must state what it cannot separate"
    assert ("`full` was omitted" in report) == (full_position is None)

    pagerank = metrics["pagerank"]
    assert pagerank["max_abs_mass_error"] < 1e-6
    assert pagerank["graph"]["nodes"] > 0
    for section in [metrics["tfidf_baseline"], pagerank]:
        assert ("topk_overlap_at_10" in section["retrieval"]) == (full_position is not None)
        assert ("ndcg_at_10_delta" in section["intervals"]) == (full_position is not None)


def test_experiment_is_deterministic_across_runs(
    session: Session, tmp_path: Path, _no_network: None
) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=True))
    config = ExperimentConfig(
        methods=["gaussian_rp"],
        dimensions=[16],
        seed=549,
        max_queries=6,
        latency_repeats=1,
        include_pagerank=False,
        include_tfidf=False,
        artifacts_dir=str(tmp_path / "a1"),
    )
    first = run_experiment(session, config).metrics["results"][0]
    config.artifacts_dir = str(tmp_path / "a2")
    second = run_experiment(session, config).metrics["results"][0]

    quality_keys = [k for k in first if k.startswith(("retrieval_", "distortion_"))]
    assert quality_keys
    for key in quality_keys:
        assert first[key] == second[key], f"{key} differs between identical seeded runs"


def test_demo_script_runs_offline(session: Session, _no_network: None, capsys) -> None:
    from scripts.ingest_demo import main

    with session_scope():
        pass
    assert main(["--source", "fixtures", "--limit", "3"]) == 0
    assert "Ingestion run" in capsys.readouterr().out
