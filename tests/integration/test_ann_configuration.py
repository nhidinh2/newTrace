"""CLI validation and persisted ANN artifacts, using offline synthetic articles."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from newstrace.config import get_settings
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.models import EvaluationRun

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--lists", "one"], "--lists"),
        (["--probes", "1,,4"], "--probes"),
        (["--subvectors", "4,"], "--subvectors"),
        (["--compare-dimensions", "32,NaN"], "--compare-dimensions"),
        (["--lists", ""], "n_lists"),
        (["--probes", ""], "n_probes"),
        (["--subvectors", ""], "n_subvectors"),
        (["--probes", "0"], "n_probes"),
        (["--bits", "9"], "n_bits"),
        (["--top-k", "0"], "top_k"),
        (["--max-queries", "-2"], "max_queries"),
        (["--since-days", "0"], "since_days"),
        (["--seed", "4294967296"], "seed"),
    ],
)
def test_cli_rejects_invalid_options_without_side_effects(
    tmp_path: Path, args: list[str], message: str
) -> None:
    database = tmp_path / "must-not-exist.db"
    artifacts = tmp_path / "must-not-exist"
    env = dict(os.environ, NEWSTRACE_DATABASE_URL=f"sqlite:///{database}")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ann.py"),
            *args,
            "--artifacts-dir",
            str(artifacts),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stderr
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not database.exists()
    assert not artifacts.exists()


def test_cli_runs_valid_grid_and_explains_skipped_layout(session: Session, tmp_path: Path) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=True))
    artifacts = tmp_path / "ann-artifacts"
    env = dict(os.environ, NEWSTRACE_DATABASE_URL=get_settings().database_url)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ann.py"),
            "--lists",
            "2",
            "--probes",
            "1,999",
            "--subvectors",
            "0,1,7",
            "--bits",
            "1",
            "--compare-dimensions",
            "",
            "--max-queries",
            "2",
            "--artifacts-dir",
            str(artifacts),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Skipped ivf2_pq7x1_probe999" in result.stdout
    assert "does not divide" in result.stdout
    session.expire_all()
    run = session.query(EvaluationRun).filter_by(kind="ann_sweep").one()
    assert run.status == "completed", run.errors
    outdir = Path(run.artifact_path or "")
    config = json.loads((outdir / "config.json").read_text())
    assert config["n_bits"] == 1
    assert config["compare_dimensions"] == []
    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics == run.metrics
    assert [row["index"] for row in metrics["results"]] == [
        "exact",
        "ivf2_flat",
        "ivf2_flat",
        "ivf2_pq1x1",
        "ivf2_pq1x1",
    ]
    assert [row["n_probe"] for row in metrics["results"]] == [0, 1, 999, 1, 999]
    skipped = [note for note in metrics["notes"] if note.startswith("Skipped")]
    assert len(skipped) == 1
    assert "does not divide the embedding dimension 384" in skipped[0]
    assert skipped[0] in (outdir / "results.md").read_text()
    assert (outdir / "environment.json").exists()
