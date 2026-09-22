"""Approximate retrieval sweep: what recall does an ANN index actually cost?

The dimensionality sweep asks what happens when the *vectors* get smaller.
This asks what happens when the *scan* stops being exhaustive, and it reports
both against the same exact baseline, so the two ways of buying memory and
latency can be read off one table.

Two things are measured, and they are not the same thing:

* **recall against exact retrieval** -- of the ten documents the exhaustive
  scan returns, how many does the approximate index return? This is a property
  of the index alone and needs no relevance judgments.
* **nDCG against the silver judgments** -- whether the substitution is visible
  in ranking quality at all. On this corpus the judgments are weak enough that
  it usually is not, which is a statement about the judgments, and it is why
  recall-against-exact is reported beside it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from newstrace.config import Settings, get_settings
from newstrace.evaluation.labels import load_story_labels
from newstrace.evaluation.retrieval import chronological_split, evaluate_rankings, ndcg_at_k
from newstrace.evaluation.systems import build_queries, environment_snapshot, silver_judgments
from newstrace.logging import get_logger
from newstrace.models import Article, EvaluationRun
from newstrace.representations.embedder import get_embedder
from newstrace.representations.registry import load_vectors
from newstrace.retrieval.ann import AnnConfig, IvfIndex
from newstrace.retrieval.exact import DenseRetriever
from newstrace.retrieval.index import get_index
from newstrace.utils import ensure_utc, git_commit, set_global_seed, utcnow

logger = get_logger(__name__)

DEFAULT_LISTS = (64,)
DEFAULT_PROBES = (1, 4, 8, 16)
DEFAULT_SUBVECTORS = (0, 48, 96)


@dataclass
class AnnSweepConfig:
    n_lists: list[int] = field(default_factory=lambda: list(DEFAULT_LISTS))
    n_probes: list[int] = field(default_factory=lambda: list(DEFAULT_PROBES))
    n_subvectors: list[int] = field(default_factory=lambda: list(DEFAULT_SUBVECTORS))
    n_bits: int = 8
    top_k: int = 10
    max_queries: int = 60
    latency_repeats: int = 3
    seed: int = 549
    since_days: int | None = None
    compare_dimensions: list[int] = field(default_factory=lambda: [32])
    artifacts_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _recall_against_exact(exact: list[int], approximate: list[int], k: int) -> float:
    reference = set(exact[:k])
    if not reference:
        return 1.0
    return len(reference & set(approximate[:k])) / len(reference)


def run_ann_sweep(
    session: Any, config: AnnSweepConfig | None = None, *, settings: Settings | None = None
) -> EvaluationRun:
    """Run the sweep, persist an :class:`EvaluationRun`, write artifacts."""
    settings = settings or get_settings()
    config = config or AnnSweepConfig(seed=settings.random_seed)
    set_global_seed(config.seed)

    run = EvaluationRun(
        kind="ann_sweep",
        status="running",
        started_at=utcnow(),
        git_commit=git_commit(),
        random_seed=config.seed,
        config=config.as_dict(),
    )
    session.add(run)
    session.commit()

    artifacts_root = Path(config.artifacts_dir or settings.resolve(settings.artifacts_dir))
    outdir = artifacts_root / "ann" / utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = _execute(session, config, settings)
    except Exception as exc:
        logger.exception("ANN sweep failed")
        run.status = "failed"
        run.finished_at = utcnow()
        run.errors = {"error": f"{exc.__class__.__name__}: {exc}"}
        run.artifact_path = str(outdir)
        session.commit()
        return run

    (outdir / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    (outdir / "environment.json").write_text(
        json.dumps(environment_snapshot(settings), indent=2), encoding="utf-8"
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "results.md").write_text(render_table(metrics), encoding="utf-8")

    run.status = "completed"
    run.finished_at = utcnow()
    run.metrics = metrics
    run.artifact_path = str(outdir)
    session.commit()
    return run


def _execute(session: Any, config: AnnSweepConfig, settings: Settings) -> dict[str, Any]:
    from sqlalchemy import select

    articles = list(session.execute(select(Article).order_by(Article.id)).scalars())
    if config.since_days:
        cutoff = utcnow() - timedelta(days=config.since_days)
        articles = [a for a in articles if (ensure_utc(a.published_at) or utcnow()) >= cutoff]
    if len(articles) < 4:
        raise ValueError("not enough articles to run a sweep; ingest first")

    full = load_vectors(session, method="full")
    if len(full) == 0:
        raise ValueError("no full-dimensional embeddings found; run indexing first")

    split = chronological_split(articles)
    split.assert_no_leakage()
    story_labels = load_story_labels(articles)
    test_ids = set(split.test_ids)
    # Most articles are the only report of their event, so a query built from
    # one has nothing to retrieve and no judgment. Build generously from the
    # held-out period, then keep the first ``max_queries`` that a judgment
    # actually covers -- truncating first would usually leave nothing.
    candidates = build_queries(
        [a for a in articles if a.id in test_ids], limit=len(articles)
    ) or build_queries(articles, limit=len(articles))
    judgments, judgment_source = silver_judgments(session, candidates, story_labels)
    collapsed: dict[str, int] = {}
    for query, article_id in candidates:
        if judgments.get(query):
            collapsed.setdefault(query, article_id)
        if len(collapsed) >= config.max_queries:
            break
    queries = list(collapsed.items())
    if not queries:
        raise ValueError("no evaluable queries; the corpus has no multi-article stories")
    query_texts = [q for q, _ in queries]
    exclude = dict(queries)

    embedder = get_embedder(settings)
    exact = DenseRetriever(full.article_ids, full.matrix, method="full")
    query_vectors = [exact.embed_query(q) for q in query_texts]
    pool = set(full.article_ids)

    # Ground truth: the exhaustive scan, with the query's own article removed
    # exactly as the dimensionality sweep removes it.
    exact_rankings: dict[str, list[int]] = {}
    exact_latency: list[float] = []
    for _ in range(config.latency_repeats):
        for query, vector in zip(query_texts, query_vectors, strict=True):
            allowed = pool - {exclude.get(query, -1)}
            start = time.perf_counter()
            hits = exact.search(vector, k=config.top_k, allowed_ids=allowed)
            exact_latency.append((time.perf_counter() - start) * 1000.0)
            exact_rankings[query] = [h.article_id for h in hits]

    exact_metrics, _ = evaluate_rankings(exact_rankings, judgments)
    notes: list[str] = []
    rows: list[dict[str, Any]] = [
        {
            "index": "exact",
            "dimension": int(full.matrix.shape[1]),
            "n_lists": 0,
            "n_probe": 0,
            "n_subvectors": 0,
            "recall_at_k_vs_exact": 1.0,
            "coverage": 1.0,
            "ndcg_at_10": round(exact_metrics.ndcg_at_10, 4),
            "index_memory_mib": round(exact.memory_bytes() / 1048576, 4),
            "p95_ms": round(float(np.percentile(exact_latency, 95)), 4),
            "build_seconds": 0.0,
        }
    ]

    # The compression arm, for the comparison that makes this interesting: a
    # 32-d exact index and an IVF-PQ index over 384-d vectors cost about the
    # same, and this is where they can be read side by side.
    for dimension in config.compare_dimensions:
        for method in ("svd", "gaussian_rp", "sparse_rp"):
            vectors = load_vectors(session, method=method, dimension=dimension)
            if len(vectors) == 0:
                continue
            # Through the cache, so the query is projected with the *saved*
            # projector for this fit -- a query embedded at 384-d cannot be
            # scored against a 32-d index.
            retriever = get_index(session, method=method, dimension=dimension, settings=settings)
            if retriever is None or retriever.projector is None:
                logger.warning(
                    "Skipping %s@%s: no saved projector for its fit_version", method, dimension
                )
                continue
            projected = [retriever.embed_query(q) for q in query_texts]
            rankings: dict[str, list[int]] = {}
            latency: list[float] = []
            for _ in range(config.latency_repeats):
                for query, vector in zip(query_texts, projected, strict=True):
                    allowed = set(vectors.article_ids) - {exclude.get(query, -1)}
                    start = time.perf_counter()
                    hits = retriever.search(vector, k=config.top_k, allowed_ids=allowed)
                    latency.append((time.perf_counter() - start) * 1000.0)
                    rankings[query] = [h.article_id for h in hits]
            metrics, _ = evaluate_rankings(rankings, judgments)
            # A compressed representation stored by an earlier sweep covers
            # only the articles that existed then. Scoring it against an exact
            # baseline over the whole corpus would read as a quality collapse
            # when it is really a stale index, so the coverage is reported
            # beside every number it explains.
            coverage = len(vectors) / max(1, len(full))
            if coverage < 0.999:
                notes.append(
                    f"{method}@{vectors.dimension} covers {len(vectors)}/{len(full)} vectors "
                    "(stored by an earlier sweep); re-run scripts/run_experiments.py to refresh "
                    "it before comparing"
                )
            rows.append(
                {
                    "index": f"exact:{method}@{vectors.dimension}",
                    "dimension": vectors.dimension,
                    "n_lists": 0,
                    "n_probe": 0,
                    "n_subvectors": 0,
                    "recall_at_k_vs_exact": round(
                        float(
                            np.mean(
                                [
                                    _recall_against_exact(
                                        exact_rankings[q], rankings.get(q, []), config.top_k
                                    )
                                    for q in query_texts
                                ]
                            )
                        ),
                        4,
                    ),
                    "coverage": round(coverage, 4),
                    "ndcg_at_10": round(metrics.ndcg_at_10, 4),
                    "index_memory_mib": round(retriever.memory_bytes() / 1048576, 4),
                    "p95_ms": round(float(np.percentile(latency, 95)), 4),
                    "build_seconds": 0.0,
                }
            )

    # The approximate arm.
    for n_lists in config.n_lists:
        for subvectors in config.n_subvectors:
            index_config = AnnConfig(
                n_lists=n_lists,
                n_probe=max(config.n_probes),
                n_subvectors=subvectors,
                n_bits=config.n_bits,
                seed=config.seed,
            )
            try:
                index = IvfIndex(index_config).fit(full.article_ids, full.matrix)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", index_config.name, exc)
                continue

            for probe in config.n_probes:
                rankings = {}
                latency = []
                for _ in range(config.latency_repeats):
                    for query, vector in zip(query_texts, query_vectors, strict=True):
                        allowed = pool - {exclude.get(query, -1)}
                        start = time.perf_counter()
                        hits = index.search(
                            vector, k=config.top_k, n_probe=probe, allowed_ids=allowed
                        )
                        latency.append((time.perf_counter() - start) * 1000.0)
                        rankings[query] = [h.article_id for h in hits]
                metrics, _ = evaluate_rankings(rankings, judgments)
                recall = float(
                    np.mean(
                        [
                            _recall_against_exact(
                                exact_rankings[q], rankings.get(q, []), config.top_k
                            )
                            for q in query_texts
                        ]
                    )
                )
                rows.append(
                    {
                        "index": f"ivf{n_lists}_"
                        + (f"pq{subvectors}x{config.n_bits}" if subvectors else "flat"),
                        "dimension": int(full.matrix.shape[1]),
                        "n_lists": n_lists,
                        "n_probe": probe,
                        "n_subvectors": subvectors,
                        "recall_at_k_vs_exact": round(recall, 4),
                        "coverage": 1.0,
                        "ndcg_at_10": round(metrics.ndcg_at_10, 4),
                        "index_memory_mib": round(index.memory_bytes() / 1048576, 4),
                        "p95_ms": round(float(np.percentile(latency, 95)), 4),
                        "build_seconds": round(index.stats.build_seconds, 3),
                    }
                )

    return {
        "corpus": {
            "articles": len(articles),
            "vectors": len(full),
            "dimension": int(full.matrix.shape[1]),
            "queries": len(queries),
            "judgment_source": judgment_source,
            "embedding_model": embedder.name,
        },
        "top_k": config.top_k,
        "results": rows,
        "notes": [
            "recall_at_k_vs_exact is measured against the exhaustive scan over the same "
            "vectors, not against human judgments.",
            "ndcg_at_10 uses the silver judgments and inherits every caveat in "
            "docs/limitations.md.",
            *notes,
        ],
    }


def render_table(metrics: dict[str, Any]) -> str:
    """A markdown table of the sweep, for the artifact directory."""
    header = (
        "| Index | dim | probe | recall@10 vs exact | nDCG@10 | Index | p95 | corpus covered |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = []
    for row in metrics.get("results", []):
        lines.append(
            f"| {row['index']} | {row['dimension']} | {row['n_probe'] or '-'} | "
            f"{row['recall_at_k_vs_exact']:.2f} | {row['ndcg_at_10']:.3f} | "
            f"{row['index_memory_mib']:.2f} MiB | {row['p95_ms']:.2f} ms | "
            f"{row.get('coverage', 1.0) * 100:.0f}% |"
        )
    corpus = metrics.get("corpus", {})
    preamble = (
        f"# Approximate retrieval sweep\n\n"
        f"{corpus.get('vectors', 0)} vectors, {corpus.get('dimension', 0)}-d, "
        f"{corpus.get('queries', 0)} queries "
        f"(judgments: {corpus.get('judgment_source', 'unknown')}).\n\n"
    )
    return preamble + header + "\n".join(lines) + "\n"


def best_ndcg(metrics: dict[str, Any]) -> float:
    return max((row["ndcg_at_10"] for row in metrics.get("results", [])), default=0.0)


__all__ = [
    "AnnSweepConfig",
    "best_ndcg",
    "ndcg_at_k",
    "render_table",
    "run_ann_sweep",
]
