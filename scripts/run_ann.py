"""Run the approximate-retrieval sweep (IVF and IVF-PQ against exact search)."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from newstrace.config import get_settings
from newstrace.db import create_all, session_scope
from newstrace.evaluation.ann_sweep import AnnSweepConfig, run_ann_sweep
from newstrace.logging import configure_logging
from newstrace.utils import set_global_seed

console = Console()


def _ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lists", default="64", help="IVF cell counts to build")
    parser.add_argument("--probes", default="1,4,8,16", help="Cells scanned per query")
    parser.add_argument(
        "--subvectors",
        default="0,48,96",
        help="PQ subvector counts; 0 means IVF-Flat. Must divide the embedding dimension.",
    )
    parser.add_argument(
        "--compare-dimensions",
        default="32",
        help="Also score the stored compressed indexes at these dimensions",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument("--seed", type=int, default=549)
    parser.add_argument(
        "--since-days",
        type=int,
        default=None,
        help="Evaluate only articles published in the last N days",
    )
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(args.seed)
    create_all()

    config = AnnSweepConfig(
        n_lists=_ints(args.lists),
        n_probes=_ints(args.probes),
        n_subvectors=_ints(args.subvectors),
        compare_dimensions=_ints(args.compare_dimensions),
        top_k=args.top_k,
        max_queries=args.max_queries,
        seed=args.seed,
        since_days=args.since_days,
        artifacts_dir=args.artifacts_dir,
    )

    with session_scope() as session:
        run = run_ann_sweep(session, config)
        console.print(f"[bold]ANN sweep {run.id}[/bold]: {run.status}")
        console.print(f"Artifacts: {run.artifact_path}")
        if run.status != "completed":
            console.print(f"[red]{run.errors}[/red]")
            return 1
        for row in (run.metrics or {}).get("results", []):
            console.print(
                f"  {row['index']:<22} probe={row['n_probe']:<3} "
                f"recall@{config.top_k}={row['recall_at_k_vs_exact']:.2f} "
                f"nDCG@10={row['ndcg_at_10']:.3f} "
                f"mem={row['index_memory_mib']:.2f} MiB "
                f"p95={row['p95_ms']:.2f} ms "
                + (
                    ""
                    if row.get("coverage", 1.0) >= 0.999
                    else f"[yellow](covers {row['coverage'] * 100:.0f}% of the corpus)[/yellow]"
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
