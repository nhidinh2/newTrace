"""The experiment runner: dimension sweeps, system metrics and PageRank analysis.

Design rules enforced here:

* Projectors are fitted **only** on the chronological fit period.
* The test period is scored once, at the end.
* Every number written to ``metrics.json`` comes from a measurement in this file.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.config import Settings, get_settings
from newstrace.evaluation.clustering import evaluate_clustering
from newstrace.evaluation.labels import load_retrieval_judgments, load_story_labels
from newstrace.evaluation.retrieval import (
    ChronologicalSplit,
    bootstrap_intervals,
    chronological_split,
    evaluate_rankings,
)
from newstrace.evaluation.summaries import evaluate_summary
from newstrace.logging import get_logger
from newstrace.models import Article, EvaluationRun, ProjectionArtifact, StoryCluster
from newstrace.representations.embedder import get_embedder
from newstrace.representations.projection import (
    cosine_distortion,
    make_projector,
    pairwise_distortion,
)
from newstrace.representations.registry import load_vectors, store_vectors
from newstrace.retrieval.exact import DenseRetriever, TfidfRetriever
from newstrace.stories import load_story
from newstrace.summarization.extractive import ExtractiveSummarizer
from newstrace.utils import ensure_utc, git_commit, set_global_seed, utcnow

logger = get_logger(__name__)

DEFAULT_METHODS = ("full", "svd", "gaussian_rp", "sparse_rp")
DEFAULT_DIMENSIONS = (32, 64, 128, 256)


@dataclass
class ExperimentConfig:
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    dimensions: list[int] = field(default_factory=lambda: list(DEFAULT_DIMENSIONS))
    seed: int = 549
    top_k: int = 10
    include_pagerank: bool = True
    include_tfidf: bool = True
    max_queries: int = 60
    latency_repeats: int = 5
    since_days: int | None = None
    artifacts_dir: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_snapshot(settings: Settings) -> dict[str, Any]:
    """Everything needed to reproduce a run."""
    import sklearn

    packages = {"numpy": np.__version__, "scikit-learn": sklearn.__version__}
    for name in ("scipy", "networkx", "pandas", "sentence_transformers", "torch"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            packages[name] = "not installed"
    embedder = get_embedder(settings)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "embedding_model": embedder.name,
        "embedding_dimension": embedder.dimension,
        "random_seed": settings.random_seed,
        "git_commit": git_commit(),
        "generated_at": utcnow().isoformat(),
    }


def peak_memory_mib() -> float:
    """Peak resident memory of this process, in MiB (0 when psutil is absent)."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KiB, macOS reports bytes.
        return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024
    except (ImportError, OSError):  # pragma: no cover
        return 0.0


QUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "be",
    "it",
    "its",
    "this",
    "that",
    "after",
    "over",
    "says",
    "said",
    "new",
    "will",
    "has",
    "have",
    "would",
    "could",
}


def build_queries(articles: list[Article], *, limit: int) -> list[tuple[str, int]]:
    """Build short keyword queries from held-out articles.

    Using the full headline as the query makes the task trivial -- the source
    article matches itself exactly. Instead we keep the salient content words,
    which is closer to what a user types, and at scoring time the source article
    is removed from both the candidate pool and the relevant set. The task is
    therefore "find the *other* reporting on this event", which is what the
    system is actually for.
    """
    from newstrace.utils import tokenize

    out: list[tuple[str, int]] = []
    for article in articles:
        tokens = [t for t in tokenize(article.title or "") if t not in QUERY_STOPWORDS]
        if len(tokens) < 4:
            continue
        out.append((" ".join(tokens[:7]), article.id))
        if len(out) >= limit:
            break
    return out


def silver_judgments(
    session: Session,
    queries: list[tuple[str, int]],
    story_labels: dict[int, int],
) -> tuple[dict[str, set[int]], str]:
    """Relevance judgments for the query set.

    Prefers reviewer judgments, then synthetic fixture labels, then the system's
    own full-dimensional clustering (a *silver* proxy, reported as such).
    """
    human = load_retrieval_judgments()
    if human:
        return ({q: set(v) for q, v in human.items()}, "human_reviewed")

    if story_labels:
        by_label: dict[int, set[int]] = {}
        for article_id, label in story_labels.items():
            by_label.setdefault(label, set()).add(article_id)
        judgments = {
            query: by_label.get(story_labels.get(article_id, -1), set()) - {article_id}
            for query, article_id in queries
        }
        return ({q: v for q, v in judgments.items() if v}, "synthetic_fixture_labels")

    from newstrace.models import ClusterMembership

    membership: dict[int, int] = {
        int(article_id): int(story_id)
        for article_id, story_id in session.execute(
            select(ClusterMembership.article_id, ClusterMembership.story_cluster_id)
        ).all()
    }
    # A syndicated copy is not independent coverage, so retrieving one earns no
    # credit here.  Leaving copies in the relevant set makes the task trivial:
    # every method finds a near-identical article, and the sweep can no longer
    # tell the representations apart.
    duplicates = {
        int(article_id)
        for (article_id,) in session.execute(
            select(Article.id).where(Article.is_near_duplicate.is_(True))
        ).all()
    }
    by_story: dict[int, set[int]] = {}
    for article_id, story_id in membership.items():
        if article_id in duplicates:
            continue
        by_story.setdefault(story_id, set()).add(article_id)
    judgments = {
        query: by_story.get(membership.get(article_id, -1), set()) - {article_id}
        for query, article_id in queries
    }
    return (
        {q: v for q, v in judgments.items() if v},
        "silver_from_full_dim_clustering_excluding_near_duplicates",
    )


@dataclass
class MethodResult:
    method: str
    dimension: int
    fit_version: str
    retrieval: dict[str, float]
    system: dict[str, float]
    distortion: dict[str, float]
    clustering: dict[str, float] | None = None
    intervals: dict[str, dict[str, float]] | None = None

    def flat(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": self.method,
            "dimension": self.dimension,
            "fit_version": self.fit_version,
        }
        row.update({f"retrieval_{k}": v for k, v in self.retrieval.items()})
        row.update({f"system_{k}": v for k, v in self.system.items()})
        row.update({f"distortion_{k}": v for k, v in self.distortion.items()})
        if self.clustering:
            row.update({f"clustering_{k}": v for k, v in self.clustering.items()})
        for metric, interval in (self.intervals or {}).items():
            row[f"ci_{metric}_mean"] = round(interval["mean"], 4)
            row[f"ci_{metric}_lo"] = round(interval["lo"], 4)
            row[f"ci_{metric}_hi"] = round(interval["hi"], 4)
        return row


def _cluster_predictions_for(
    session: Session,
    article_ids: list[int],
    matrix: np.ndarray,
    *,
    settings: Settings,
    method: str,
) -> dict[int, int]:
    """Cluster a representation in memory (no writes) to score clustering quality."""
    from newstrace.clustering.online import combined_score, content_tokens, default_threshold
    from newstrace.utils import jaccard

    articles = {
        a.id: a
        for a in session.execute(select(Article).where(Article.id.in_(article_ids))).scalars()
    }
    order = sorted(
        article_ids,
        key=lambda aid: (
            ensure_utc(articles[aid].published_at) or datetime.max.replace(tzinfo=None),
            aid,
        ),
    )
    index = {aid: i for i, aid in enumerate(article_ids)}
    threshold = default_threshold(settings)
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    tokens: list[set[str]] = []
    assignment: dict[int, int] = {}

    for aid in order:
        vector = matrix[index[aid]]
        article = articles[aid]
        title_tokens = content_tokens(article.title)
        best_i, best_score = -1, -1.0
        for i, centroid in enumerate(centroids):
            denom = float(np.linalg.norm(vector) * np.linalg.norm(centroid))
            cosine = float(np.dot(vector, centroid) / denom) if denom > 1e-9 else 0.0
            score = combined_score(
                cosine,
                jaccard(title_tokens, tokens[i]),
                0.0,
                title_weight=settings.cluster_title_overlap_weight,
                entity_weight=settings.cluster_entity_overlap_weight,
            )
            if score > best_score:
                best_i, best_score = i, score
        if best_i >= 0 and best_score >= threshold:
            n = counts[best_i]
            centroids[best_i] = (centroids[best_i] * n + vector) / (n + 1)
            counts[best_i] = n + 1
            tokens[best_i] |= title_tokens
            assignment[aid] = best_i
        else:
            centroids.append(vector.astype(np.float64).copy())
            counts.append(1)
            tokens.append(set(title_tokens))
            assignment[aid] = len(centroids) - 1
    logger.debug("Clustered %s articles into %s clusters (%s)", len(order), len(centroids), method)
    return assignment


def run_experiment(
    session: Session,
    config: ExperimentConfig | None = None,
    *,
    settings: Settings | None = None,
) -> EvaluationRun:
    """Run the full sweep and persist an :class:`EvaluationRun` plus artifacts."""
    settings = settings or get_settings()
    config = config or ExperimentConfig(seed=settings.random_seed)
    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    run = EvaluationRun(
        kind="dimension_sweep",
        status="running",
        started_at=utcnow(),
        git_commit=git_commit(),
        random_seed=config.seed,
        config=config.as_dict(),
    )
    session.add(run)
    session.commit()

    artifacts_root = Path(config.artifacts_dir or settings.resolve(settings.artifacts_dir))
    stamp = utcnow().strftime("%Y-%m-%dT%H%M%SZ")
    outdir = artifacts_root / "experiments" / stamp
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = _execute(session, config, settings, rng, outdir)
    except Exception as exc:
        logger.exception("Experiment failed")
        run.status = "failed"
        run.finished_at = utcnow()
        run.errors = {"error": f"{exc.__class__.__name__}: {exc}"}
        run.artifact_path = str(outdir)
        session.commit()
        return run

    run.status = "completed"
    run.finished_at = utcnow()
    run.metrics = metrics
    run.artifact_path = str(outdir)
    session.commit()
    return run


def _execute(
    session: Session,
    config: ExperimentConfig,
    settings: Settings,
    rng: np.random.Generator,
    outdir: Path,
) -> dict[str, Any]:
    articles = list(session.execute(select(Article).order_by(Article.id)).scalars())
    corpus_size = len(articles)
    window_cutoff: datetime | None = None
    if config.since_days:
        # A live index serves a recent window. Archive feeds can stretch a corpus
        # over years, which would put a decade-old vendor post in the fit period
        # and this morning's wire story in the test period.
        window_cutoff = utcnow() - timedelta(days=config.since_days)
        articles = [
            a for a in articles if (ensure_utc(a.published_at) or utcnow()) >= window_cutoff
        ]
        logger.info(
            "Restricted the corpus to the last %s days: %s of %s articles",
            config.since_days,
            len(articles),
            corpus_size,
        )
    if len(articles) < 4:
        raise ValueError("not enough articles to run an experiment; ingest fixtures first")

    split = chronological_split(articles)
    split.assert_no_leakage()
    story_labels = load_story_labels(articles)

    full = load_vectors(session, method="full")
    if len(full) == 0:
        raise ValueError("no full-dimensional embeddings found; run indexing first")
    embedder = get_embedder(settings)
    index_by_article = full.index_by_article

    fit_ids = [aid for aid in split.fit_ids if aid in index_by_article]
    test_ids = [aid for aid in split.test_ids if aid in index_by_article]
    fit_matrix = full.matrix[[index_by_article[a] for a in fit_ids]] if fit_ids else full.matrix

    # The retrieval pool is the whole corpus -- at query time a live index holds
    # everything ingested so far. Only the *queries* come from the held-out
    # period, so no fitted artefact ever saw the articles being scored.
    in_window = {a.id for a in articles}
    eval_ids = [aid for aid in full.article_ids if aid in in_window]
    test_articles = [a for a in articles if a.id in set(test_ids)]
    queries = build_queries(test_articles, limit=config.max_queries)
    if not queries:
        queries = build_queries(articles, limit=config.max_queries)
    judgments, judgment_source = silver_judgments(session, queries, story_labels)
    queries = [(q, aid) for q, aid in queries if q in judgments]
    # Queries are built from article keywords, so two near-identical articles can
    # produce the same text. Every downstream stage keys rankings by that text,
    # which silently collapses the duplicates -- while the reported query count
    # went on counting them. Collapse them here instead, once and visibly.
    collapsed: dict[str, int] = {}
    for query, article_id in queries:
        collapsed[query] = article_id
    queries = list(collapsed.items())
    query_texts = [q for q, _ in queries]
    exclude_by_query = dict(queries)

    baseline_rankings: dict[str, list[int]] | None = None

    baseline_query_rows: list[dict[str, object]] | None = None
    results: list[MethodResult] = []
    per_query_rows: list[dict[str, Any]] = []
    sweep_notes: list[str] = []

    for method in config.methods:
        dims = (
            [0] if method == "full" else [d for d in config.dimensions if d < full.matrix.shape[1]]
        )
        if method != "full" and not dims:
            logger.warning(
                "All requested dimensions exceed the embedding dimension; skipping %s", method
            )
            continue
        seen_dims: set[int] = set()
        for dimension in dims:
            projector = make_projector(method, dimension, random_seed=config.seed)

            fit_start = time.perf_counter()
            projector.fit(fit_matrix)  # fit period only -- no look-ahead
            fit_seconds = time.perf_counter() - fit_start

            transform_start = time.perf_counter()
            projected_all = projector.transform(full.matrix)
            transform_seconds = time.perf_counter() - transform_start

            if projector.dimension in seen_dims:
                # TruncatedSVD is capped at n_samples - 1; a larger request
                # collapses onto a dimension we already measured.
                note = (
                    f"{method}: requested d={dimension} collapsed onto d="
                    f"{projector.dimension} (TruncatedSVD is capped at "
                    f"n_fit_samples - 1 = {max(0, len(fit_ids) - 1)}); not run twice"
                )
                logger.warning(note)
                sweep_notes.append(note)
                continue
            seen_dims.add(projector.dimension)
            if projector.dimension != dimension and method != "full":
                note = (
                    f"{method}: requested d={dimension} clamped to d="
                    f"{projector.dimension} by the fit-set size ({len(fit_ids)} articles)"
                )
                logger.warning(note)
                sweep_notes.append(note)

            fit_version = (
                "identity"
                if method == "full"
                else projector.fit_version(embedder.name, len(fit_ids))
            )
            if method != "full":
                _save_projector(
                    session, projector, fit_version, embedder.name, config, split, outdir
                )
                store_vectors(
                    session,
                    full.article_ids,
                    projected_all,
                    model_name=embedder.name,
                    method=method,
                    fit_version=fit_version,
                )
                session.commit()

            eval_rows = [index_by_article[a] for a in eval_ids]
            eval_matrix = projected_all[eval_rows]
            retriever = DenseRetriever(eval_ids, eval_matrix, method=method, projector=projector)

            # Embed each query once, outside the timed region: query encoding is
            # identical across methods, so including it would mask the very
            # differences the sweep exists to measure.
            query_vectors = [retriever.embed_query(q) for q in query_texts]
            if query_vectors:
                retriever.search(query_vectors[0], k=config.top_k)  # warm up BLAS

            rankings: dict[str, list[int]] = {}
            latencies: list[float] = []
            for repeat in range(config.latency_repeats):
                for query, vector in zip(query_texts, query_vectors, strict=True):
                    allowed = set(eval_ids) - {exclude_by_query.get(query, -1)}
                    start = time.perf_counter()
                    hits = retriever.search(vector, k=config.top_k, allowed_ids=allowed)
                    latencies.append((time.perf_counter() - start) * 1000.0)
                    if repeat == 0:
                        rankings[query] = [h.article_id for h in hits]

            if method == "full" and baseline_rankings is None:
                baseline_rankings = rankings

            retrieval_metrics, rows = evaluate_rankings(
                rankings, judgments, baseline=baseline_rankings
            )
            if method == "full" and baseline_query_rows is None:
                baseline_query_rows = [dict(r) for r in rows]
            intervals = bootstrap_intervals(
                rows,
                baseline_rows=None if method == "full" else baseline_query_rows,
                seed=config.seed,
            )
            for row in rows:
                row.update({"method": method, "dimension": projector.dimension})
            per_query_rows.extend(rows)

            distortion = {
                **pairwise_distortion(full.matrix, projected_all, rng=rng),
                **cosine_distortion(full.matrix, projected_all, rng=rng),
            }

            clustering_metrics: dict[str, float] | None = None
            if story_labels:
                predicted = _cluster_predictions_for(
                    session, eval_ids, eval_matrix, settings=settings, method=method
                )
                truth = {aid: story_labels[aid] for aid in eval_ids if aid in story_labels}
                if len(truth) > 1:
                    clustering_metrics = evaluate_clustering(truth, predicted).as_dict()

            latency_array = np.array(latencies) if latencies else np.zeros(1)
            system = {
                "index_memory_mib": eval_matrix.nbytes / (1024 * 1024),
                "bytes_per_vector": float(eval_matrix.nbytes / max(1, eval_matrix.shape[0])),
                "p50_query_latency_ms": float(np.percentile(latency_array, 50)),
                "p95_query_latency_ms": float(np.percentile(latency_array, 95)),
                "latency_samples": float(latency_array.size),
                "projection_fit_seconds": fit_seconds,
                "projection_transform_seconds": transform_seconds,
                "vectors": float(eval_matrix.shape[0]),
                "dimension": float(eval_matrix.shape[1]),
                "peak_memory_mib": peak_memory_mib(),
            }

            results.append(
                MethodResult(
                    method=method,
                    dimension=projector.dimension,
                    fit_version=fit_version,
                    retrieval=retrieval_metrics.as_dict(),
                    system={k: round(v, 5) for k, v in system.items()},
                    distortion={k: round(float(v), 5) for k, v in distortion.items()},
                    clustering=clustering_metrics,
                    intervals=intervals,
                )
            )
            logger.info(
                "%s d=%s: nDCG@10=%.3f recall@10=%.3f mem=%.2fMiB p95=%.2fms",
                method,
                projector.dimension,
                retrieval_metrics.ndcg_at_10,
                retrieval_metrics.recall_at_10,
                system["index_memory_mib"],
                system["p95_query_latency_ms"],
            )

    tfidf_metrics: dict[str, Any] | None = None
    if config.include_tfidf:
        tfidf_metrics = _tfidf_baseline(
            session,
            eval_ids,
            query_texts,
            judgments,
            baseline_rankings,
            config,
            exclude_by_query,
            baseline_rows=baseline_query_rows,
        )

    pagerank_metrics: dict[str, Any] | None = None
    if config.include_pagerank:
        pagerank_metrics = _pagerank_analysis(
            session,
            eval_ids,
            query_texts,
            judgments,
            baseline_rankings,
            config,
            full,
            exclude_by_query,
            baseline_rows=baseline_query_rows,
        )

    summary_metrics = _summary_metrics(session)
    ingest_metrics = _throughput_metrics(session)

    metrics: dict[str, Any] = {
        "split": split.as_dict(),
        "judgment_source": judgment_source,
        "query_count": len(query_texts),
        "article_count": len(articles),
        "corpus_size": corpus_size,
        "window_days": config.since_days,
        "window_cutoff": window_cutoff.isoformat() if window_cutoff else None,
        "embedding_model": embedder.name,
        "embedding_dimension": full.matrix.shape[1],
        "results": [r.flat() for r in results],
        "tfidf_baseline": tfidf_metrics,
        "pagerank": pagerank_metrics,
        "summaries": summary_metrics,
        "ingestion": ingest_metrics,
        "sweep_notes": sweep_notes,
        "retrieval_pool_size": len(eval_ids),
        "task": (
            "Given a short keyword query built from a held-out article, retrieve the "
            "OTHER articles covering the same event. The source article is excluded "
            "from both the candidate pool and the relevant set."
        ),
    }

    _write_artifacts(outdir, config, settings, metrics, per_query_rows, results)
    return metrics


def _save_projector(
    session: Session,
    projector: Any,
    fit_version: str,
    model_name: str,
    config: ExperimentConfig,
    split: ChronologicalSplit,
    outdir: Path,
) -> None:
    path = outdir / "projectors" / f"{projector.method}_{projector.dimension}_{fit_version}.joblib"
    projector.save(path)
    existing = session.execute(
        select(ProjectionArtifact).where(ProjectionArtifact.fit_version == fit_version)
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            ProjectionArtifact(
                method=projector.method,
                dimension=projector.dimension,
                fit_version=fit_version,
                source_model=model_name,
                random_seed=config.seed,
                fit_article_count=len(split.fit_ids),
                fit_cutoff=split.fit_cutoff,
                path=str(path),
                stats={"requested_dimension": projector.meta.get("requested_dimension")},
            )
        )
    else:
        existing.path = str(path)
    session.flush()


def _tfidf_baseline(
    session: Session,
    eval_ids: list[int],
    query_texts: list[str],
    judgments: dict[str, set[int]],
    baseline_rankings: dict[str, list[int]] | None,
    config: ExperimentConfig,
    exclude_by_query: dict[str, int] | None = None,
    baseline_rows: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    articles = {
        a.id: a for a in session.execute(select(Article).where(Article.id.in_(eval_ids))).scalars()
    }
    ordered = [aid for aid in eval_ids if aid in articles]
    retriever = TfidfRetriever(ordered, [articles[a].body_text for a in ordered])
    rankings: dict[str, list[int]] = {}
    latencies: list[float] = []
    excludes = exclude_by_query or {}
    for query in query_texts:
        allowed = set(ordered) - {excludes.get(query, -1)}
        start = time.perf_counter()
        hits = retriever.search(query, k=config.top_k, allowed_ids=allowed)
        latencies.append((time.perf_counter() - start) * 1000.0)
        rankings[query] = [h.article_id for h in hits]
    metrics, rows = evaluate_rankings(rankings, judgments, baseline=baseline_rankings)
    arr = np.array(latencies) if latencies else np.zeros(1)
    return {
        "retrieval": metrics.as_dict(),
        "intervals": bootstrap_intervals(rows, baseline_rows=baseline_rows, seed=config.seed),
        "system": {
            "index_memory_mib": retriever.memory_bytes() / (1024 * 1024),
            "p50_query_latency_ms": float(np.percentile(arr, 50)),
            "p95_query_latency_ms": float(np.percentile(arr, 95)),
        },
    }


def _pagerank_analysis(
    session: Session,
    eval_ids: list[int],
    query_texts: list[str],
    judgments: dict[str, set[int]],
    baseline_rankings: dict[str, list[int]] | None,
    config: ExperimentConfig,
    full: Any,
    exclude_by_query: dict[str, int] | None = None,
    baseline_rows: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    from newstrace.graph.build import build_graph, graph_stats
    from newstrace.graph.pagerank import (
        damping_sensitivity,
        personalized_pagerank,
        seed_from_articles,
    )
    from newstrace.retrieval.rerank import blend_pagerank

    articles = list(session.execute(select(Article).where(Article.id.in_(eval_ids))).scalars())
    graph = build_graph(session, articles=articles)
    stats = graph_stats(graph)

    index = full.index_by_article
    rows = [index[a] for a in eval_ids if a in index]
    retriever = DenseRetriever(
        [a for a in eval_ids if a in index], full.matrix[rows], method="full"
    )

    rankings: dict[str, list[int]] = {}
    iterations: list[int] = []
    latencies: list[float] = []
    converged = 0
    mass_error = 0.0
    excludes = exclude_by_query or {}
    pool = set(retriever.article_ids)
    for query in query_texts:
        hits = retriever.search(
            query, k=config.top_k * 3, allowed_ids=pool - {excludes.get(query, -1)}
        )
        if not hits:
            rankings[query] = []
            continue
        seeds = seed_from_articles([h.article_id for h in hits], [max(h.score, 0.0) for h in hits])
        start = time.perf_counter()
        result = personalized_pagerank(graph, personalization=seeds)
        latencies.append((time.perf_counter() - start) * 1000.0)
        iterations.append(result.iterations)
        converged += int(result.converged)
        mass_error = max(mass_error, abs(result.total_mass() - 1.0))
        reranked = blend_pagerank(hits, result.article_scores())
        rankings[query] = [h.article_id for h in reranked[: config.top_k]]

    metrics, rows = evaluate_rankings(rankings, judgments, baseline=baseline_rankings)
    return {
        "graph": stats.as_dict(),
        "retrieval": metrics.as_dict(),
        "intervals": bootstrap_intervals(rows, baseline_rows=baseline_rows, seed=config.seed),
        "mean_iterations": float(np.mean(iterations)) if iterations else 0.0,
        "converged_fraction": converged / max(1, len(query_texts)),
        "max_abs_mass_error": mass_error,
        "p95_pagerank_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "damping_sensitivity": damping_sensitivity(graph),
    }


def _summary_metrics(session: Session) -> dict[str, Any]:
    """Citation and evidence metrics over the largest stories.

    A story whose articles carry no usable sentence produces no statements, and
    a summary with nothing to cite is not a citation failure.  Averaging its
    zeroes in would report a coverage drop that no statement caused, so empty
    summaries are counted separately.  ``citation_coverage_micro`` weights by
    statement rather than by story, which is the number to quote when stories
    differ in length.
    """
    summarizer = ExtractiveSummarizer(session)
    stories = list(
        session.execute(
            select(StoryCluster).order_by(StoryCluster.article_count.desc()).limit(10)
        ).scalars()
    )
    rows = []
    without_statements = 0
    for story in stories:
        bundle = load_story(session, story.id)
        if bundle is None or not bundle.articles:
            continue
        summary = summarizer.summarize_bundle(bundle)
        if not summary.statements:
            without_statements += 1
            continue
        rows.append(evaluate_summary(summary).as_dict())
    if not rows:
        return {
            "stories_evaluated": without_statements,
            "stories_with_statements": 0,
            "stories_without_statements": without_statements,
        }
    keys = rows[0].keys()
    statements = float(sum(r["statements"] for r in rows))
    cited = float(sum(r["citation_coverage"] * r["statements"] for r in rows))
    return {
        "stories_evaluated": len(rows) + without_statements,
        "stories_with_statements": len(rows),
        "stories_without_statements": without_statements,
        "citation_coverage_micro": (cited / statements) if statements else 0.0,
        **{k: float(np.mean([r[k] for r in rows])) for k in keys},
    }


def _throughput_metrics(session: Session) -> dict[str, Any]:
    from newstrace.models import IngestionRun

    runs = list(
        session.execute(
            select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(5)
        ).scalars()
    )
    rows = []
    for run in runs:
        if not run.finished_at or not run.started_at:
            continue
        seconds = max(1e-6, (run.finished_at - run.started_at).total_seconds())
        rows.append(
            {
                "run_id": run.id,
                "source": run.source,
                "articles": run.fetched_count,
                "seconds": round(seconds, 4),
                "articles_per_second": round(run.fetched_count / seconds, 3),
            }
        )
    return {"runs": rows}


def _write_artifacts(
    outdir: Path,
    config: ExperimentConfig,
    settings: Settings,
    metrics: dict[str, Any],
    per_query_rows: list[dict[str, Any]],
    results: list[MethodResult],
) -> None:
    (outdir / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    (outdir / "environment.json").write_text(
        json.dumps(environment_snapshot(settings), indent=2), encoding="utf-8"
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import pandas as pd

        pd.DataFrame(per_query_rows).to_parquet(outdir / "per_query_metrics.parquet")
        cluster_rows = [r.flat() for r in results]
        pd.DataFrame(cluster_rows).to_parquet(outdir / "per_cluster_metrics.parquet")
    except Exception as exc:
        logger.warning("Skipping parquet export: %s", exc)
        import csv

        if per_query_rows:
            with (outdir / "per_query_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(per_query_rows[0]))
                writer.writeheader()
                writer.writerows(per_query_rows)

    from newstrace.evaluation.report import write_report

    write_report(outdir, metrics)
