"""Rebuild the denormalised per-story domain aggregates from cluster members.

``StoryCluster.distinct_domain_count`` and ``extra["domains"]`` are maintained
incrementally as articles arrive, so they only ever grow.  Anything that edits
an article after clustering leaves them stale:

* re-running duplicate detection, which removes domains from the independent
  set (top stories are *ranked* by this count, so a stale one keeps a story
  built from syndicated copies at the top of the page);
* rewriting ``source_domain``, which renames them.

This recomputes both from the articles a story actually holds, counting only
the members that are not near-duplicates, which is the rule the ingest path
applies. Idempotent.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from newstrace.db import session_scope
from newstrace.models import Article, ClusterMembership, StoryCluster

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes. Without this the run only reports them.",
    )
    args = parser.parse_args()

    changes: list[tuple[int, int, int]] = []
    with session_scope() as session:
        independent: dict[int, set[str]] = defaultdict(set)
        rows = session.execute(
            select(ClusterMembership.story_cluster_id, Article).join(
                Article, Article.id == ClusterMembership.article_id
            )
        )
        for cluster_id, article in rows:
            # Touch every cluster, so one that lost its last domain is reset too.
            bucket = independent[cluster_id]
            if not article.is_near_duplicate and article.source_domain:
                bucket.add(article.source_domain)

        for cluster in session.scalars(select(StoryCluster)):
            domains = sorted(independent.get(cluster.id, set()))
            extra = dict(cluster.extra or {})
            if cluster.distinct_domain_count == len(domains) and extra.get("domains") == domains:
                continue
            changes.append((cluster.id, cluster.distinct_domain_count, len(domains)))
            if args.apply:
                extra["domains"] = domains
                cluster.extra = extra
                cluster.distinct_domain_count = len(domains)
        if not args.apply:
            session.rollback()

    if not changes:
        console.print("[green]All story aggregates already agree with their members.[/green]")
        return 0

    table = Table(title=f"{len(changes)} story(ies) {'rebuilt' if args.apply else 'affected'}")
    table.add_column("story", justify="right")
    table.add_column("was", justify="right")
    table.add_column("now", justify="right")
    for story_id, was, now in sorted(changes, key=lambda row: row[1] - row[2], reverse=True)[:30]:
        table.add_row(str(story_id), str(was), str(now))
    console.print(table)
    if len(changes) > 30:
        console.print(f"... and {len(changes) - 30} more")
    if not args.apply:
        console.print("[yellow]Dry run. Re-run with --apply to write.[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
