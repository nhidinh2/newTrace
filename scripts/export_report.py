"""Regenerate report.md and plots from a saved metrics.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from newstrace.config import get_settings
from newstrace.evaluation.report import load_metrics, write_report

console = Console()


def latest_experiment(root: Path) -> Path | None:
    candidates = sorted((p for p in root.glob("*") if (p / "metrics.json").exists()), reverse=True)
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a NewsTrace experiment report.")
    parser.add_argument("--experiment-dir", default=None, help="Defaults to the newest run")
    args = parser.parse_args(argv)

    settings = get_settings()
    root = settings.resolve(settings.artifacts_dir) / "experiments"
    target = Path(args.experiment_dir) if args.experiment_dir else latest_experiment(root)
    if target is None or not (target / "metrics.json").exists():
        console.print(f"[red]No experiment with metrics.json found under {root}[/red]")
        return 1

    path = write_report(target, load_metrics(target / "metrics.json"))
    console.print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
