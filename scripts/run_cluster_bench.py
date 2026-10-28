"""Score the story clusterer on a public, human-labelled news stream.

    python scripts/run_cluster_bench.py --download
    python scripts/run_cluster_bench.py --no-tune --max-documents 2000

``--download`` is the only step that needs the network; everything after it
runs offline against the downloaded splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from newstrace.config import get_settings
from newstrace.evaluation.cluster_bench import (
    DEFAULT_GRID,
    ClusterBenchConfig,
    download,
    render_table,
    run_cluster_bench,
)
from newstrace.logging import configure_logging
from newstrace.utils import set_global_seed, utcnow

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/cluster_bench")
    parser.add_argument("--download", action="store_true", help="Fetch the splits first")
    parser.add_argument("--language", default="eng", help="eng, deu or spa")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_GRID),
        help="Comma-separated finite thresholds swept on dev; non-empty unless --no-tune",
    )
    parser.add_argument("--no-tune", action="store_true", help="Score the shipped threshold only")
    parser.add_argument(
        "--no-learn", action="store_true", help="Tune the threshold but keep the shipped weights"
    )
    parser.add_argument("--window-hours", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None, help="Prefix of each stream")
    parser.add_argument("--seed", type=int, default=549)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(args.seed)

    try:
        config = ClusterBenchConfig(
            data_dir=args.data_dir,
            language=args.language,
            thresholds=[float(t) for t in args.thresholds.split(",") if t.strip()],
            tune=not args.no_tune,
            learn_weights=not args.no_learn,
            window_hours=args.window_hours,
            max_documents=args.max_documents,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(f"--thresholds: {exc}")

    if args.download:
        console.print("Downloading the Miranda et al. (2018) splits (this needs the network)...")
        download(config, settings)

    try:
        metrics = run_cluster_bench(config, settings=settings)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    root = Path(args.artifacts_dir or settings.resolve(settings.artifacts_dir))
    outdir = root / "cluster_bench" / f"{config.language}-{utcnow().strftime('%Y-%m-%dT%H%M%SZ')}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "results.md").write_text(render_table(metrics), encoding="utf-8")

    console.print(render_table(metrics))
    console.print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
