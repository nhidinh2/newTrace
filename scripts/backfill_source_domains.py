"""Recompute ``articles.source_domain`` with the current domain rules.

Rows written before ``extract_domain`` learned about multi-tenant publishing
platforms stored the platform (``substack.com``) where they should store the
tenant (``importai.substack.com``).  Corroboration counts read that column, so
until it is rewritten two unrelated newsletters look like one source.

The rewrite is derived from each row's stored URL and is idempotent: running it
twice changes nothing the second time.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from newstrace.db import session_scope
from newstrace.ingestion.normalize import extract_domain
from newstrace.models import Article

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes. Without this the run only reports them.",
    )
    args = parser.parse_args()

    changes: list[tuple[int, str, str]] = []
    with session_scope() as session:
        for article in session.scalars(select(Article)):
            current = article.source_domain or ""
            recomputed = extract_domain(article.canonical_url or article.url)
            # An unparseable URL yields "", which would erase a good value.
            if recomputed and recomputed != current:
                changes.append((article.id, current, recomputed))
                if args.apply:
                    article.source_domain = recomputed
        if not args.apply:
            session.rollback()

    if not changes:
        console.print("[green]No rows need rewriting.[/green]")
        return 0

    table = Table(title=f"{len(changes)} article(s) {'rewritten' if args.apply else 'affected'}")
    table.add_column("id", justify="right")
    table.add_column("was")
    table.add_column("now")
    for article_id, was, now in changes[:40]:
        table.add_row(str(article_id), was, now)
    console.print(table)
    if len(changes) > 40:
        console.print(f"... and {len(changes) - 40} more")
    if not args.apply:
        console.print("[yellow]Dry run. Re-run with --apply to write.[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
