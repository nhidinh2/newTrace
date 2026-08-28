"""Run the dimensionality-reduction sweep and write a timestamped artifact directory."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from newstrace.config import get_settings
from newstrace.db import create_all, session_scope
from newstrace.evaluation.systems import ExperimentConfig, run_experiment
from newstrace.logging import configure_logging
from newstrace.utils import set_global_seed

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NewsTrace retrieval experiments.")
    parser.add_argument("--methods", default="full,svd,gaussian_rp,sparse_rp")
    parser.add_argument("--dimensions", default="32,64,128,256")
    parser.add_argument("--seed", type=int, default=549)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="Evaluate only articles published in the last N days (archive feeds "
        "can otherwise stretch the chronological split over years)",
    )
    parser.add_argument("--no-pagerank", action="store_true")
    parser.add_argument("--no-tfidf", action="store_true")
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(args.seed)
    create_all()

    config = ExperimentConfig(
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        dimensions=[int(d) for d in args.dimensions.split(",") if d.strip()],
        seed=args.seed,
        top_k=args.top_k,
        max_queries=args.max_queries,
        include_pagerank=not args.no_pagerank,
        include_tfidf=not args.no_tfidf,
        since_days=args.since_days,
        artifacts_dir=args.artifacts_dir,
    )

    with session_scope() as session:
        run = run_experiment(session, config)
        console.print(f"[bold]Experiment {run.id}[/bold]: {run.status}")
        console.print(f"Artifacts: {run.artifact_path}")
        if run.status != "completed":
            console.print(f"[red]{run.errors}[/red]")
            return 1
        results = (run.metrics or {}).get("results", [])
        for row in results:
            console.print(
                f"  {row['method']:<12} d={row['dimension']:<4} "
                f"nDCG@10={row.get('retrieval_ndcg_at_10', 0):.3f} "
                f"recall@10={row.get('retrieval_recall_at_10', 0):.3f} "
                f"mem={row.get('system_index_memory_mib', 0):.3f} MiB "
                f"p95={row.get('system_p95_query_latency_ms', 0):.2f} ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
