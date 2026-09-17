"""Re-apply near-duplicate detection to stored articles.

Articles ingested before ``title_similarity`` learned to ignore trailing
mastheads kept their ``is_near_duplicate`` flag from the old comparison, so a
wire story republished across co-owned mastheads still reports one independent
source per masthead.

The re-scan is deliberately scoped to articles that already share a story
cluster.  Clustering has done the expensive work of putting related articles
together, and the corroboration counts this repairs are computed per story, so
a cluster is exactly the neighbourhood that matters.  Duplicates whose copies
landed in different clusters are out of scope and stay as they are.

Nothing is deleted: a copy is flagged, linked to the article it copies, and
left in place, which is what the ingest path does too.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from newstrace.db import session_scope
from newstrace.ingestion.deduplicate import title_similarity
from newstrace.models import Article, ClusterMembership
from newstrace.utils import ensure_utc

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.9, help="Title similarity threshold")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes. Without this the run only reports them.",
    )
    args = parser.parse_args()

    flagged: list[tuple[int, str, str, float]] = []
    with session_scope() as session:
        members: dict[int, list[Article]] = defaultdict(list)
        rows = session.execute(
            select(ClusterMembership.story_cluster_id, Article)
            .join(Article, Article.id == ClusterMembership.article_id)
            .where(Article.is_near_duplicate.is_(False))
        )
        for cluster_id, article in rows:
            members[cluster_id].append(article)

        for articles in members.values():
            if len(articles) < 2:
                continue
            # The earliest copy is the one kept, so corroboration is credited to
            # whoever published first rather than to whoever was ingested first.
            articles.sort(key=lambda a: (ensure_utc(a.published_at) is None, a.published_at, a.id))
            kept: list[Article] = [articles[0]]
            for article in articles[1:]:
                best_score, best_original = 0.0, None
                for original in kept:
                    score = title_similarity(article.title, original.title)
                    if score > best_score:
                        best_score, best_original = score, original
                if best_original is not None and best_score >= args.threshold:
                    flagged.append(
                        (article.id, article.source_domain, best_original.source_domain, best_score)
                    )
                    if args.apply:
                        # This article may already be the original for copies
                        # ingested after it. Move them onto the retained article
                        # so no copy ends up pointing at another copy.
                        dependents = session.scalars(
                            select(Article).where(Article.duplicate_of_article_id == article.id)
                        )
                        for dependent in dependents:
                            dependent.duplicate_of_article_id = best_original.id
                        article.is_near_duplicate = True
                        article.duplicate_of_article_id = best_original.id
                        article.duplicate_reason = "title_similarity"
                        article.duplicate_similarity = float(best_score)
                else:
                    kept.append(article)
        if not args.apply:
            session.rollback()

    if not flagged:
        console.print("[green]No further near-duplicates found.[/green]")
        return 0

    table = Table(title=f"{len(flagged)} copy(ies) {'flagged' if args.apply else 'affected'}")
    table.add_column("id", justify="right")
    table.add_column("copy")
    table.add_column("copies")
    table.add_column("score", justify="right")
    for article_id, copy_domain, original_domain, score in flagged[:30]:
        table.add_row(str(article_id), copy_domain, original_domain, f"{score:.3f}")
    console.print(table)
    if len(flagged) > 30:
        console.print(f"... and {len(flagged) - 30} more")
    if not args.apply:
        console.print("[yellow]Dry run. Re-run with --apply to write.[/yellow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
