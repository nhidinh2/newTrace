"""Fixture-backed demo: ingest, cluster and print the resulting stories.

Runs with no network access and no API keys.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from newstrace.config import get_settings
from newstrace.db import create_all, session_scope
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.logging import configure_logging
from newstrace.stories import list_stories, load_story
from newstrace.utils import set_global_seed, truncate

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the NewsTrace fixture demo.")
    parser.add_argument("--source", default="fixtures", choices=["fixtures", "gdelt", "rss"])
    parser.add_argument("--topic", default=None)
    parser.add_argument("--limit", type=int, default=12, help="Stories to display")
    parser.add_argument("--no-index", action="store_true", help="Skip embedding and clustering")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    set_global_seed(settings.random_seed)
    create_all()

    with session_scope() as session:
        run, result = run_ingestion(
            session,
            IngestOptions(source=args.source, topic=args.topic, index=not args.no_index),
        )
        console.print(
            f"[bold]Ingestion run {run.id}[/bold] source={run.source} status={run.status}\n"
            f"  fetched={result.fetched} inserted={result.inserted} "
            f"duplicates={result.duplicates} skipped={result.skipped} failures={result.failures}"
        )
        if run.status == "failed":
            console.print(f"[red]{run.errors}[/red]")
            return 1

        stories = list_stories(session, topic=args.topic, limit=args.limit)
        table = Table(title=f"Top stories ({len(stories)} shown)")
        table.add_column("id", justify="right")
        table.add_column("story", overflow="fold")
        table.add_column("articles", justify="right")
        table.add_column("independent sources", justify="right")
        table.add_column("duplicates", justify="right")
        for story in stories:
            bundle = load_story(session, story.id)
            table.add_row(
                str(story.id),
                truncate(story.display_title, 68),
                str(story.article_count),
                str(story.distinct_domain_count),
                str(len(bundle.duplicates) if bundle else 0),
            )
        console.print(table)
        console.print(
            "\nNext: [cyan]make api[/cyan] and [cyan]make ui[/cyan], or "
            "[cyan]uv run newstrace story <id>[/cyan] for a timeline and grounded summary."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
