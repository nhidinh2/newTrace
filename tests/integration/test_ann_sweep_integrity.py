"""ANN comparisons must share an encoder and a declared document pool."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from newstrace.config import get_settings
from newstrace.evaluation import ann_sweep
from newstrace.evaluation.ann_sweep import AnnSweepConfig, run_ann_sweep
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.models import (
    Article,
    ClusterMembership,
    EmbeddingRecord,
    ProjectionArtifact,
    StoryCluster,
)
from newstrace.representations.embedder import HashingEmbedder, Matrix, get_embedder
from newstrace.representations.projection import make_projector
from newstrace.representations.registry import ensure_embeddings, load_vectors, store_vectors
from newstrace.retrieval import exact
from newstrace.utils import utcnow


def _config(root: Path, **kwargs) -> AnnSweepConfig:
    return AnnSweepConfig(
        n_lists=[1],
        n_probes=[1],
        n_subvectors=[0],
        latency_repeats=1,
        compare_dimensions=[8],
        artifacts_dir=str(root),
        **kwargs,
    )


def _articles(session: Session) -> list[Article]:
    run_ingestion(session, IngestOptions(source="fixtures", index=False))
    return list(session.scalars(select(Article).order_by(Article.id)))


def _projection(
    session: Session,
    root: Path,
    *,
    embedder=None,
    selected=None,
    source_model=None,
    with_artifact=True,
) -> ProjectionArtifact:
    embedder = embedder or get_embedder()
    full = load_vectors(
        session,
        method="full",
        model_name=embedder.name,
        dimension=embedder.dimension,
        fit_version="identity",
    )
    projector = make_projector("gaussian_rp", 8).fit(full.matrix[:4])
    path = root / "projection.joblib"
    projector.save(path)
    rows = [i for i, aid in enumerate(full.article_ids) if selected is None or aid in selected]
    store_vectors(
        session,
        [full.article_ids[i] for i in rows],
        projector.transform(full.matrix[rows]),
        model_name=embedder.name,
        method="gaussian_rp",
        fit_version="test-fit",
    )
    artifact = ProjectionArtifact(
        method="gaussian_rp",
        dimension=8,
        fit_version="test-fit",
        source_model=source_model or embedder.name,
        path=str(path),
    )
    if with_artifact:
        session.add(artifact)
    session.commit()
    return artifact


@pytest.mark.parametrize("same_width", [True, False])
def test_ann_selects_one_encoder_and_reports_missing_vectors(session, tmp_path, same_width):
    articles = _articles(session)
    embedder = get_embedder()
    legacy = HashingEmbedder(embedder.dimension if same_width else 24, name="legacy")
    ensure_embeddings(session, articles, embedder=legacy)
    ensure_embeddings(session, articles[1:], embedder=embedder)
    session.commit()

    run = run_ann_sweep(session, _config(tmp_path))
    assert run.status == "completed", run.errors
    assert run.metrics["corpus"]["vectors"] == len(articles) - 1
    assert run.metrics["corpus"]["articles"] == len(articles)
    assert run.metrics["corpus"]["missing_vectors"] == 1
    assert run.metrics["corpus"]["embedding_model"] == embedder.name
    assert len(run.metrics["results"]) == 2
    assert run.metrics["results"][1]["recall_at_k_vs_exact"] == 1.0
    report = (Path(run.artifact_path) / "results.md").read_text()
    assert "1 article(s)" in report
    assert embedder.name in report
    saved = json.loads((Path(run.artifact_path) / "metrics.json").read_text())
    assert saved == run.metrics


def test_ann_encodes_queries_once_with_configured_encoder(session, tmp_path, monkeypatch):
    articles = _articles(session)
    batches = []

    class RecordingEmbedder(HashingEmbedder):
        def encode(self, texts: Sequence[str]) -> Matrix:
            batches.append(list(texts))
            return super().encode(texts)

    embedder = RecordingEmbedder(32, name="configured")
    ensure_embeddings(session, articles, embedder=embedder)
    _projection(session, tmp_path, embedder=embedder)
    batches.clear()
    monkeypatch.setattr(ann_sweep, "get_embedder", lambda settings: embedder)

    def unexpected_default():
        pytest.fail("ANN queries used the process-default encoder")

    monkeypatch.setattr(exact, "get_embedder", unexpected_default)
    run = run_ann_sweep(session, _config(tmp_path), settings=get_settings())
    assert run.status == "completed", run.errors
    assert len(batches) == 1
    assert len(batches[0]) == run.metrics["corpus"]["queries"] > 0
    assert len(run.metrics["results"]) == 3
    assert run.metrics["results"][1]["fit_version"] == "test-fit"


@pytest.mark.parametrize("mismatch", ["model", "width", "missing", "file"])
def test_ann_skips_incompatible_compression_with_visible_reason(session, tmp_path, mismatch):
    articles = _articles(session)
    embedder = get_embedder()
    ensure_embeddings(session, articles, embedder=embedder)
    projected_embedder = HashingEmbedder(24, name="legacy") if mismatch == "width" else embedder
    if mismatch == "width":
        ensure_embeddings(session, articles, embedder=projected_embedder)
    artifact = _projection(
        session,
        tmp_path,
        embedder=projected_embedder,
        source_model="legacy" if mismatch == "model" else embedder.name,
        with_artifact=mismatch != "missing",
    )
    if mismatch == "file":
        Path(artifact.path).unlink()
    run = run_ann_sweep(session, _config(tmp_path))
    assert run.status == "completed", run.errors
    assert len(run.metrics["results"]) == 2
    notes = " ".join(run.metrics["notes"])
    assert "Skipping gaussian_rp@8" in notes
    assert "projector" in notes
    assert "Skipping gaussian_rp@8" in (Path(run.artifact_path) / "results.md").read_text()


def test_ann_window_and_compression_coverage_share_the_exact_pool(session, tmp_path, monkeypatch):
    articles = _articles(session)
    embedder = get_embedder()
    ensure_embeddings(session, articles, embedder=embedder)
    now = utcnow()
    # Preserve publication order for query generation; leave two old documents.
    for i, article in enumerate(articles):
        article.published_at = now - timedelta(days=40 if i < 2 else 1, seconds=len(articles) - i)
    session.commit()
    window_ids = {article.id for article in articles[2:]}
    selected = {article.id for article in articles[:-1]}
    _projection(session, tmp_path, selected=selected)

    pools = []
    original_search = exact.DenseRetriever.search

    def recording_search(self, query, **kwargs):
        assert set(self.article_ids) <= window_ids
        assert kwargs["allowed_ids"] <= set(self.article_ids)
        pools.append(set(self.article_ids))
        return original_search(self, query, **kwargs)

    monkeypatch.setattr(exact.DenseRetriever, "search", recording_search)
    original_fit = ann_sweep.IvfIndex.fit

    def recording_fit(self, article_ids, matrix):
        assert set(article_ids) == window_ids
        return original_fit(self, article_ids, matrix)

    monkeypatch.setattr(ann_sweep.IvfIndex, "fit", recording_fit)
    run = run_ann_sweep(session, _config(tmp_path, since_days=30))
    assert run.status == "completed", run.errors
    assert pools
    assert run.metrics["corpus"]["vectors"] == len(window_ids)
    assert run.metrics["corpus"]["articles"] == len(window_ids)
    compressed = run.metrics["results"][1]
    assert compressed["coverage"] == round(len(window_ids & selected) / len(window_ids), 4)
    assert compressed["vectors"] == len(window_ids & selected)
    assert "covers" in " ".join(run.metrics["notes"])
    assert "covers" in (Path(run.artifact_path) / "results.md").read_text()


def test_ann_fails_without_compatible_vectors_and_writes_no_success_metrics(session, tmp_path):
    articles = _articles(session)
    ensure_embeddings(session, articles, embedder=HashingEmbedder(24, name="legacy"))
    session.commit()
    run = run_ann_sweep(session, _config(tmp_path))
    assert run.status == "failed"
    assert get_embedder().name in run.errors["error"]
    assert "evaluation window" in run.errors["error"]
    assert not (Path(run.artifact_path) / "metrics.json").exists()


def test_ann_window_excludes_out_of_window_relevance_judgments(session, tmp_path, monkeypatch):
    articles = _articles(session)
    ensure_embeddings(session, articles)
    now = utcnow()
    for i, article in enumerate(articles):
        article.published_at = now - timedelta(days=40 if i < 2 else 1, seconds=len(articles) - i)
    story = StoryCluster(display_title="Synthetic coverage across a window boundary")
    session.add(story)
    session.flush()
    for article in articles:
        session.add(
            ClusterMembership(
                article_id=article.id,
                story_cluster_id=story.id,
                similarity=1.0,
                assignment_method="test",
            )
        )
    session.commit()
    # Exercise the real membership-based silver judgments, which see all dates.
    monkeypatch.setattr(ann_sweep, "load_story_labels", lambda articles: {})
    captured = []
    evaluate = ann_sweep.evaluate_rankings
    window_ids = {a.id for a in articles[2:]}
    missing_id = next(a.id for a in articles[2:] if not a.is_near_duplicate)

    session.execute(delete(EmbeddingRecord).where(EmbeddingRecord.article_id == missing_id))
    session.commit()

    def inspect_judgments(rankings, judgments):
        assert judgments
        assert all(relevant <= window_ids for relevant in judgments.values())
        assert any(missing_id in relevant for relevant in judgments.values())
        captured.append(judgments)
        return evaluate(rankings, judgments)

    monkeypatch.setattr(ann_sweep, "evaluate_rankings", inspect_judgments)
    run = run_ann_sweep(session, _config(tmp_path, since_days=30))
    assert run.status == "completed", run.errors
    assert (
        run.metrics["corpus"]["judgment_source"]
        == "silver_from_full_dim_clustering_excluding_near_duplicates"
    )
    assert captured


@pytest.mark.parametrize("compatible", [True, False])
def test_ann_cli_reports_comparison_notes_and_failure(session, tmp_path, capsys, compatible):
    import runpy

    from rich.console import Console

    articles = _articles(session)
    ensure_embeddings(
        session,
        articles,
        embedder=get_embedder() if compatible else HashingEmbedder(24, name="legacy"),
    )
    session.commit()
    if compatible:
        _projection(session, tmp_path, source_model="wrong-encoder")
    main = runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "run_ann.py"))["main"]
    main.__globals__["console"] = Console(width=200, color_system=None)
    status = main(
        [
            "--lists",
            "1",
            "--probes",
            "1",
            "--subvectors",
            "0",
            "--compare-dimensions",
            "8",
            "--top-k",
            "3",
            "--max-queries",
            "2",
            "--artifacts-dir",
            str(tmp_path / "cli-artifacts"),
        ]
    )
    output = " ".join(capsys.readouterr().out.split())
    assert status == (0 if compatible else 1)
    if compatible:
        assert "Skipping gaussian_rp@8" in output
        metrics_path = next((tmp_path / "cli-artifacts").rglob("metrics.json"))
        metrics = json.loads(metrics_path.read_text())
        assert metrics["corpus"]["queries"] == 2
        assert "recall@3 vs exact" in metrics_path.with_name("results.md").read_text()
    else:
        assert "no compatible full embeddings" in output
        assert not list((tmp_path / "cli-artifacts").rglob("metrics.json"))
