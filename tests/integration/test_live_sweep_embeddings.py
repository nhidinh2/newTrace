"""The live sweep must compare representations in one embedding space."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.config import get_settings
from newstrace.evaluation.retrieval import chronological_split
from newstrace.evaluation.systems import ExperimentConfig, run_experiment
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.models import Article, EmbeddingRecord, ProjectionArtifact
from newstrace.representations.embedder import HashingEmbedder, Matrix, get_embedder
from newstrace.representations.registry import ensure_embeddings
from newstrace.retrieval import exact


def _sweep(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        methods=["full", "gaussian_rp"],
        dimensions=[8],
        max_queries=12,
        latency_repeats=1,
        artifacts_dir=str(root),
    )


@pytest.mark.parametrize("same_width", [True, False])
def test_sweep_uses_only_configured_model_vectors(
    session: Session, tmp_path: Path, same_width: bool
) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=False))
    articles = list(session.scalars(select(Article).order_by(Article.id)))
    first_fit_id = chronological_split(articles).fit_ids[0]
    selected = [article for article in articles if article.id != first_fit_id]
    embedder = get_embedder()
    legacy_width = embedder.dimension if same_width else embedder.dimension // 2
    ensure_embeddings(session, articles, embedder=HashingEmbedder(legacy_width, name="legacy"))
    ensure_embeddings(session, selected, embedder=embedder)
    session.commit()

    run = run_experiment(session, _sweep(tmp_path))
    assert run.status == "completed", run.errors
    metrics = run.metrics
    assert metrics["embedding_model"] == embedder.name
    assert metrics["embedding_dimension"] == embedder.dimension
    assert metrics["retrieval_pool_size"] == len(selected)
    assert metrics["embedding_coverage"]["fit"] == metrics["split"]["fit_count"] - 1
    assert all(row["system_vectors"] == len(selected) for row in metrics["results"])
    assert metrics["query_count"] > 0

    projected_ids = set(
        session.scalars(
            select(EmbeddingRecord.article_id).where(EmbeddingRecord.method == "gaussian_rp")
        )
    )
    assert projected_ids == {article.id for article in selected}
    projector = session.scalars(select(ProjectionArtifact)).one()
    assert projector.source_model == metrics["embedding_model"]
    assert projector.fit_article_count == metrics["embedding_coverage"]["fit"]
    outdir = Path(run.artifact_path)
    environment = json.loads((outdir / "environment.json").read_text(encoding="utf-8"))
    assert environment["embedding_model"] == metrics["embedding_model"]
    assert "1 article(s) in the evaluation window lack" in (outdir / "report.md").read_text(
        encoding="utf-8"
    )


def test_sweep_reuses_one_batch_for_dense_methods_and_pagerank(
    session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=True))
    settings = get_settings()
    batches: list[list[str]] = []

    class RecordingEmbedder(HashingEmbedder):
        def encode(self, texts: Sequence[str]) -> Matrix:
            batches.append(list(texts))
            return super().encode(texts)

    from newstrace.evaluation import systems

    embedder = RecordingEmbedder(settings.embedding_dimension_fallback)
    monkeypatch.setattr(systems, "get_embedder", lambda settings: embedder)

    def unconfigured_embedder() -> None:
        pytest.fail("the sweep tried to encode a query with the process default")

    monkeypatch.setattr(exact, "get_embedder", unconfigured_embedder)
    config = _sweep(tmp_path)
    config.dimensions = [8, 16]
    run = run_experiment(session, config, settings=settings)
    assert run.status == "completed", run.errors
    assert len(batches) == 1
    assert len(batches[0]) == run.metrics["query_count"] > 0
    assert run.metrics["pagerank"]["retrieval"]["queries"] == len(batches[0])
    assert len(run.metrics["results"]) == 3


@pytest.mark.parametrize("coverage", ["none", "test_only"])
def test_sweep_fails_before_saving_projections_without_compatible_fit_vectors(
    session: Session, tmp_path: Path, coverage: str
) -> None:
    run_ingestion(session, IngestOptions(source="fixtures", index=True))
    articles = list(session.scalars(select(Article).order_by(Article.id)))
    settings = get_settings().model_copy(update={"embedding_dimension_fallback": 24})
    if coverage == "test_only":
        test_ids = set(chronological_split(articles).test_ids)
        ensure_embeddings(
            session,
            [article for article in articles if article.id in test_ids],
            embedder=HashingEmbedder(24),
        )
        session.commit()

    run = run_experiment(session, _sweep(tmp_path), settings=settings)
    assert run.status == "failed"
    assert "hashing-ngram-v1-24" in run.errors["error"]
    assert ("fit period" if coverage == "test_only" else "evaluation window") in run.errors["error"]
    assert session.scalars(select(ProjectionArtifact)).first() is None
    assert not (Path(run.artifact_path) / "metrics.json").exists()
