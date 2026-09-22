"""Populate the blocking keys and the full-text index for existing articles.

Articles ingested before the blocking keys existed carry no SimHash, no
``article_signatures`` rows and no ``articles_fts`` row.  Duplicate detection
would find no candidates for them and the full-text filter would not see them,
so a database created by an earlier version has to be walked once.

Both rewrites are derived from stored columns and are idempotent: a second run
reports the same counts and changes nothing.
"""

from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from sqlalchemy import func, select

from newstrace.db import session_scope
from newstrace.ingestion.deduplicate import index_article
from newstrace.models import Article, ArticleSignature
from newstrace.retrieval import fulltext

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size", type=int, default=500, help="Articles per commit (default: 500)"
    )
    parser.add_argument(
        "--only",
        choices=["all", "signatures", "fulltext"],
        default="all",
        help="Restrict the run to one index",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    with session_scope() as session:
        total = int(session.execute(select(func.count(Article.id))).scalar_one())
        console.print(f"{total} article(s) in this database")

        if args.only in ("all", "signatures"):
            written = 0
            keys = 0
            offset = 0
            while True:
                batch = list(
                    session.execute(
                        select(Article).order_by(Article.id).limit(args.batch_size).offset(offset)
                    ).scalars()
                )
                if not batch:
                    break
                for article in batch:
                    keys += index_article(session, article)
                    written += 1
                session.commit()
                offset += args.batch_size
                console.print(f"  signatures: {written}/{total}", end="\r")
            rows = int(session.execute(select(func.count(ArticleSignature.id))).scalar_one())
            console.print(
                f"[green]Blocking keys: {written} article(s), {keys} key(s) "
                f"({rows} rows in article_signatures)[/green]"
            )

        if args.only in ("all", "fulltext"):
            if not fulltext.available(session):
                console.print(
                    "[yellow]No articles_fts table: run `alembic upgrade head` first "
                    "(or this SQLite build lacks FTS5).[/yellow]"
                )
            else:
                indexed = fulltext.rebuild(session, batch_size=args.batch_size)
                console.print(f"[green]Full-text index: {indexed} article(s)[/green]")

    console.print(f"Done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
