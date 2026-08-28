"""Sample the hardest article pairs for human labelling.

Pairs are chosen *near* the clustering decision boundary, because those are the
ones where a human judgement actually changes the outcome.  This writes a queue
for the project owner to fill in; it never invents labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from rich.console import Console
from sqlalchemy import select

from newstrace.clustering.online import combined_score, content_tokens
from newstrace.config import REPO_ROOT, get_settings
from newstrace.db import create_all, session_scope
from newstrace.models import Article
from newstrace.representations.registry import load_vectors
from newstrace.utils import ensure_utc, jaccard, set_global_seed, truncate

console = Console()

FIELDS = [
    "article_a_id",
    "article_b_id",
    "score",
    "hours_apart",
    "domain_a",
    "domain_b",
    "title_a",
    "title_b",
    "url_a",
    "url_b",
    "same_story",
    "reviewer",
    "notes",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a story-pair labelling queue.")
    parser.add_argument("--limit", type=int, default=60, help="Pairs to emit")
    parser.add_argument("--band", type=float, default=0.12, help="Half-width around the threshold")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    set_global_seed(settings.random_seed)
    create_all()
    output = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "data" / "labels" / "story_pairs.queue.csv"
    )

    with session_scope() as session:
        articles = {
            a.id: a
            for a in session.execute(
                select(Article).where(Article.is_near_duplicate.is_(False))
            ).scalars()
        }
        vectors = load_vectors(session, method="full")
        if len(vectors) == 0:
            console.print("[red]No embeddings found. Run `make demo` first.[/red]")
            return 1

        ids = [aid for aid in vectors.article_ids if aid in articles]
        index = vectors.index_by_article
        matrix = vectors.matrix
        threshold = settings.cluster_threshold
        low, high = threshold - args.band, threshold + args.band

        rows = []
        for i, a_id in enumerate(ids):
            va = matrix[index[a_id]]
            a = articles[a_id]
            for b_id in ids[i + 1 :]:
                b = articles[b_id]
                cosine = float(np.dot(va, matrix[index[b_id]]))
                score = combined_score(
                    cosine,
                    jaccard(content_tokens(a.title), content_tokens(b.title)),
                    0.0,
                    title_weight=settings.cluster_title_overlap_weight,
                    entity_weight=settings.cluster_entity_overlap_weight,
                )
                if not (low <= score <= high):
                    continue
                ta, tb = ensure_utc(a.published_at), ensure_utc(b.published_at)
                hours = abs((ta - tb).total_seconds()) / 3600.0 if ta and tb else -1.0
                rows.append(
                    {
                        "article_a_id": a_id,
                        "article_b_id": b_id,
                        "score": round(score, 4),
                        "hours_apart": round(hours, 2),
                        "domain_a": a.source_domain,
                        "domain_b": b.source_domain,
                        "title_a": truncate(a.title, 140),
                        "title_b": truncate(b.title, 140),
                        "url_a": a.url,
                        "url_b": b.url,
                        "same_story": "",
                        "reviewer": "",
                        "notes": "",
                    }
                )

        rows.sort(key=lambda r: abs(float(r["score"]) - threshold))
        rows = rows[: args.limit]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    console.print(f"Wrote {len(rows)} borderline pairs to {output}")
    console.print(
        "Fill in [bold]same_story[/bold] (1/0), [bold]reviewer[/bold] and [bold]notes[/bold], "
        "then save as data/labels/story_pairs.csv"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
