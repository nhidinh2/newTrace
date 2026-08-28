"""Report and plot generation from saved machine-readable metrics.

Everything here reads ``metrics.json``; nothing recomputes a number.  Tables and
plots therefore cannot drift from the measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from newstrace.logging import get_logger

logger = get_logger(__name__)

PLOTS = (
    ("quality_vs_dimension.png", "dimension", "retrieval_ndcg_at_10", "nDCG@10 vs dimension"),
    (
        "quality_vs_memory.png",
        "system_index_memory_mib",
        "retrieval_ndcg_at_10",
        "nDCG@10 vs index memory (MiB)",
    ),
    (
        "quality_vs_latency.png",
        "system_p95_query_latency_ms",
        "retrieval_ndcg_at_10",
        "nDCG@10 vs p95 query latency (ms)",
    ),
)


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.4f}" if abs(value) < 1000 else f"{value:.1f}")
            else:
                cells.append("—" if value is None else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _interval(row: dict[str, Any], metric: str) -> str:
    """Render a stored bootstrap interval, or an em dash when absent."""
    lo = row.get(f"ci_{metric}_lo")
    hi = row.get(f"ci_{metric}_hi")
    if lo is None or hi is None:
        return "—"
    return f"[{float(lo):.4f}, {float(hi):.4f}]"


def _uncertainty_table(rows: list[dict[str, Any]]) -> str:
    """nDCG@10 and Recall@10 intervals plus the paired delta against full."""
    if not any("ci_ndcg_at_10_lo" in r for r in rows):
        return ""
    lines = [
        "| method | dim | nDCG@10 | 95% CI | Recall@10 | 95% CI | ΔnDCG vs full | Δ 95% CI | "
        "separable from full? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        delta_lo = row.get("ci_ndcg_at_10_delta_lo")
        delta_hi = row.get("ci_ndcg_at_10_delta_hi")
        if row.get("method") == "full":
            delta, verdict = "baseline", "—"
        elif delta_lo is None or delta_hi is None:
            delta, verdict = "—", "—"
        else:
            point = row.get("ci_ndcg_at_10_delta_mean")
            delta = f"{float(point):+.4f}" if point is not None else "—"
            verdict = "no" if float(delta_lo) <= 0.0 <= float(delta_hi) else "**yes**"
        lines.append(
            f"| {row.get('method')} | {row.get('dimension')} | "
            f"{float(row.get('retrieval_ndcg_at_10') or 0.0):.4f} | "
            f"{_interval(row, 'ndcg_at_10')} | "
            f"{float(row.get('retrieval_recall_at_10') or 0.0):.4f} | "
            f"{_interval(row, 'recall_at_10')} | {delta} | "
            f"{_interval(row, 'ndcg_at_10_delta')} | {verdict} |"
        )
    return "\n".join(lines)


def make_plots(outdir: Path, metrics: dict[str, Any]) -> list[str]:
    """Write the three quality-versus-cost plots.  Missing matplotlib is not fatal."""
    rows = [r for r in metrics.get("results", []) if r.get("method") != "full" or True]
    if not rows:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        logger.warning("matplotlib not installed; skipping plots")
        return []

    written: list[str] = []
    methods = sorted({r["method"] for r in rows})
    for filename, x_key, y_key, title in PLOTS:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in methods:
            series = sorted(
                (r for r in rows if r["method"] == method and x_key in r and y_key in r),
                key=lambda r: r[x_key],
            )
            if not series:
                continue
            ax.plot(
                [r[x_key] for r in series],
                [r[y_key] for r in series],
                marker="o",
                label=method,
            )
        ax.set_xlabel(x_key.replace("_", " "))
        ax.set_ylabel("nDCG@10")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=150)
        plt.close(fig)
        written.append(filename)
    return written


def write_report(outdir: Path, metrics: dict[str, Any]) -> Path:
    """Render ``report.md`` from measured metrics."""
    plots = make_plots(outdir, metrics)
    rows = metrics.get("results", [])
    split = metrics.get("split", {})
    judgment_source = metrics.get("judgment_source", "unknown")

    quality_table = _table(
        rows,
        [
            ("method", "method"),
            ("dim", "dimension"),
            ("Recall@10", "retrieval_recall_at_10"),
            ("nDCG@10", "retrieval_ndcg_at_10"),
            ("MRR", "retrieval_mrr"),
            ("top-10 overlap", "retrieval_topk_overlap_at_10"),
        ],
    )
    cost_table = _table(
        rows,
        [
            ("method", "method"),
            ("dim", "dimension"),
            ("memory MiB", "system_index_memory_mib"),
            ("bytes/vector", "system_bytes_per_vector"),
            ("p50 ms", "system_p50_query_latency_ms"),
            ("p95 ms", "system_p95_query_latency_ms"),
            ("fit s", "system_projection_fit_seconds"),
            ("transform s", "system_projection_transform_seconds"),
        ],
    )
    distortion_table = _table(
        rows,
        [
            ("method", "method"),
            ("dim", "dimension"),
            ("mean |Δd|/d", "distortion_mean_abs_distortion"),
            ("p95 |Δd|/d", "distortion_p95_abs_distortion"),
            ("mean |Δcos|", "distortion_mean_abs_cosine_error"),
        ],
    )
    has_clustering = any("clustering_bcubed_f1" in r for r in rows)
    clustering_table = (
        _table(
            rows,
            [
                ("method", "method"),
                ("dim", "dimension"),
                ("pairwise F1", "clustering_pairwise_f1"),
                ("B-cubed F1", "clustering_bcubed_f1"),
                ("ARI", "clustering_adjusted_rand_index"),
                ("fragmentation", "clustering_fragmentation_rate"),
                ("merge errors", "clustering_merge_error_rate"),
            ],
        )
        if has_clustering
        else "_No labelled clusters available; clustering metrics were not computed._"
    )

    article_count = metrics.get("article_count", 0)
    pool_size = metrics.get("retrieval_pool_size", 0)
    model_name = metrics.get("embedding_model", "unknown")
    model_dim = metrics.get("embedding_dimension", 0)
    fit_n = split.get("fit_count", 0)
    val_n = split.get("validation_count", 0)
    test_n = split.get("test_count", 0)
    sweep_notes = metrics.get("sweep_notes", [])
    skipped_block = chr(10).join(f"- {n}" for n in sweep_notes) or (
        "_Nothing was skipped; every requested (method, dimension) pair ran._"
    )

    baseline = next((r for r in rows if r.get("method") == "full"), None)
    observations: list[str] = []
    if baseline:
        base_ndcg = float(baseline.get("retrieval_ndcg_at_10") or 0.0)
        base_mem = float(baseline.get("system_index_memory_mib") or 0.0)
        base_p95 = float(baseline.get("system_p95_query_latency_ms") or 0.0)
        for row in rows:
            if row.get("method") == "full":
                continue
            ndcg = float(row.get("retrieval_ndcg_at_10") or 0.0)
            mem = float(row.get("system_index_memory_mib") or 0.0)
            p95 = float(row.get("system_p95_query_latency_ms") or 0.0)
            retained = (ndcg / base_ndcg * 100.0) if base_ndcg > 0 else 0.0
            shrink = (base_mem / mem) if mem > 0 else 0.0
            speed = (base_p95 / p95) if p95 > 0 else 0.0
            observations.append(
                f"- `{row['method']}` at d={row['dimension']}: retained "
                f"**{retained:.1f}%** of baseline nDCG@10 with a {shrink:.2f}× smaller "
                f"index; p95 query latency {speed:.2f}× the baseline's "
                f"(>1 = faster than full)."
            )

    window_days = metrics.get("window_days")
    window_note = (
        f", drawn from a stored corpus of {metrics.get('corpus_size', article_count)} by "
        f"keeping only the last {window_days} days (cutoff `{metrics.get('window_cutoff')}`)"
        if window_days
        else ""
    )
    uncertainty_table = _uncertainty_table(rows)
    query_count = metrics.get("query_count", 0)
    if uncertainty_table:
        uncertainty_block = (
            f"Percentile bootstrap over the {query_count} per-query results in "
            "`per_query_metrics.parquet` (or `.csv`), 2000 resamples, seeded. The "
            "delta column is a *paired* bootstrap of the per-query difference "
            "against the full-dimensional baseline: the same queries are easy or "
            "hard for every method, so pairing is what makes the comparison "
            "readable. `separable from full?` is *no* whenever the delta interval "
            "contains zero — that is, whenever this corpus cannot distinguish the "
            "method from the baseline.\n\n"
            f"{uncertainty_table}\n\n"
            "A bootstrap interval describes sampling variability in *this* query "
            "set only. It cannot repair a biased corpus, silver labels, or a "
            "query set too small to represent the task."
        )
    else:
        uncertainty_block = (
            f"Per-query results are in `per_query_metrics.parquet` (or `.csv`). No "
            f"intervals were computed for this run; the tables report means over "
            f"{query_count} queries."
        )

    tfidf = metrics.get("tfidf_baseline") or {}
    pagerank = metrics.get("pagerank") or {}
    summaries = metrics.get("summaries") or {}

    report = f"""# NewsTrace experiment report

Generated from `metrics.json` in this directory. Every number below is a
measurement produced by `scripts/run_experiments.py`; nothing is estimated.

## Task

{metrics.get("task", "Retrieval over the stored corpus.")}

## Setup

- Articles: **{article_count}** (retrieval pool: {pool_size}){window_note}
- Embedding model: `{model_name}` (d={model_dim})
- Chronological split: fit={fit_n}, validation={val_n}, test={test_n}
- Fit cutoff: `{split.get("fit_cutoff")}` — projectors saw **no** test-period article
- Queries: **{metrics.get("query_count", 0)}**
- Relevance judgments: `{judgment_source}`

## Observed results

### Retrieval quality

{quality_table}

### Computational cost

{cost_table}

### Geometry: distance and cosine distortion

{distortion_table}

### Clustering

{clustering_table}

### Lexical baseline (TF-IDF)

{_format_kv(tfidf.get("retrieval", {}))}

### Graph reranking (personalized PageRank)

{_format_pagerank(pagerank)}

### Summaries and evidence

{_format_kv(summaries)}

## Interpretation

{chr(10).join(observations) if observations else "_Only the full-dimensional baseline was run._"}

## What was skipped

{skipped_block}

## Statistical uncertainty

{uncertainty_block}

## Limitations

- Relevance judgments came from `{judgment_source}`. Silver labels derived from the
  system's own clustering flatter the system: they measure self-consistency, not
  human-judged relevance.
- GDELT and the configured feeds do not represent all journalism.
- Publication timestamps do not establish who reported a claim first.
- Similar wording can indicate copying rather than independent corroboration.
- Source counts do not establish truth. NewsTrace is not a fact checker.

## Plots

{chr(10).join(f"- `{p}`" for p in plots) if plots else "_No plots generated._"}
"""
    path = outdir / "report.md"
    path.write_text(report, encoding="utf-8")
    logger.info("Wrote report to %s", path)
    return path


def _format_kv(payload: dict[str, Any]) -> str:
    if not payload:
        return "_Not measured in this run._"
    lines = ["| metric | value |", "| --- | --- |"]
    for key, value in payload.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        elif isinstance(value, (int, str)):
            lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _format_pagerank(payload: dict[str, Any]) -> str:
    if not payload:
        return "_Not measured in this run._"
    graph = payload.get("graph", {})
    parts = [
        _format_kv(payload.get("retrieval", {})),
        "",
        f"Graph: {graph.get('nodes', 0)} nodes, {graph.get('edges', 0)} edges, "
        f"{graph.get('connected_components', 0)} components "
        f"(largest holds {float(graph.get('largest_component_fraction', 0.0)):.1%} of nodes).",
        "",
        f"Convergence: mean {float(payload.get('mean_iterations', 0.0)):.1f} iterations, "
        f"{float(payload.get('converged_fraction', 0.0)):.0%} of queries converged, "
        f"max |sum(p) - 1| = {float(payload.get('max_abs_mass_error', 0.0)):.2e}.",
    ]
    sensitivity = payload.get("damping_sensitivity") or []
    if sensitivity:
        parts.extend(
            [
                "",
                "| damping | iterations | top-10 overlap with first |",
                "| --- | --- | --- |",
                *[
                    f"| {r['damping']:.2f} | {int(r['iterations'])} | "
                    f"{float(r['top10_overlap_with_first']):.2f} |"
                    for r in sensitivity
                ],
            ]
        )
    return "\n".join(parts)


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
