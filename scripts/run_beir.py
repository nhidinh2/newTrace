"""Run the compression sweep against a BEIR dataset (human relevance judgments).

    python scripts/run_beir.py --download --dataset scifact
    python scripts/run_beir.py --dataset scifact --dimensions 32,64,128

``--download`` is the only step that needs the network; everything after it
runs offline against the unpacked dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from newstrace.config import get_settings
from newstrace.evaluation.beir import BeirConfig, download, render_table, run_beir
from newstrace.logging import configure_logging
from newstrace.utils import set_global_seed, utcnow

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--download", action="store_true", help="Fetch the dataset first")
    parser.add_argument("--methods", default="full,svd,gaussian_rp,sparse_rp")
    parser.add_argument("--dimensions", default="32,64,128,256")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--no-tfidf", action="store_true")
    parser.add_argument("--seed", type=int, default=549)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(args.seed)

    config = BeirConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        dimensions=[int(d) for d in args.dimensions.split(",") if d.strip()],
        split=args.split,
        top_k=args.top_k,
        max_documents=args.max_documents,
        max_queries=args.max_queries,
        include_tfidf=not args.no_tfidf,
        seed=args.seed,
    )

    if args.download:
        console.print(f"Downloading [bold]{config.dataset}[/bold] (this needs the network)...")
        download(config, settings)

    try:
        metrics = run_beir(config, settings=settings)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    root = Path(args.artifacts_dir or settings.resolve(settings.artifacts_dir))
    outdir = root / "beir" / f"{config.dataset}-{utcnow().strftime('%Y-%m-%dT%H%M%SZ')}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "results.md").write_text(render_table(metrics), encoding="utf-8")

    console.print(render_table(metrics))
    console.print(f"Artifacts: {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
