"""Loading human/synthetic labels used by the evaluation suite.

Real labels are the project owner's job; NewsTrace ships only templates plus a
synthetic label file for the committed fixtures.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from newstrace.config import REPO_ROOT
from newstrace.models import Article

STORY_PAIRS_TEMPLATE = "story_pairs.template.csv"
STORY_LABELS_FILE = "fixture_story_labels.csv"
RETRIEVAL_TEMPLATE = "retrieval_judgments.template.csv"
LABELS_DIR = REPO_ROOT / "data" / "labels"


def load_story_labels(articles: Sequence[Article], path: Path | None = None) -> dict[int, int]:
    """Map ``article_id -> ground-truth story label`` via canonical URL."""
    target = path or (LABELS_DIR / STORY_LABELS_FILE)
    if not target.exists():
        return {}
    by_url = {a.canonical_url: a.id for a in articles}
    labels: dict[int, int] = {}
    label_ids: dict[str, int] = {}
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            url = (row.get("canonical_url") or "").strip()
            label = (row.get("story_label") or "").strip()
            if not url or not label or url not in by_url:
                continue
            labels[by_url[url]] = label_ids.setdefault(label, len(label_ids) + 1)
    return labels


def load_story_pairs(path: Path | None = None) -> list[tuple[int, int, int]]:
    """Load reviewed ``(article_a_id, article_b_id, same_story)`` triples."""
    target = path or (LABELS_DIR / "story_pairs.csv")
    if not target.exists():
        return []
    rows: list[tuple[int, int, int]] = []
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                a = int(row["article_a_id"])
                b = int(row["article_b_id"])
                same = int(row["same_story"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append((a, b, same))
    return rows


def load_retrieval_judgments(path: Path | None = None) -> dict[str, dict[int, float]]:
    """Load ``query -> {article_id: relevance}`` reviewer judgments."""
    target = path or (LABELS_DIR / "retrieval_judgments.csv")
    if not target.exists():
        return {}
    out: dict[str, dict[int, float]] = {}
    with target.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query = (row.get("query") or "").strip()
            try:
                article_id = int(row["article_id"])
                relevance = float(row.get("relevance") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if query and relevance > 0:
                out.setdefault(query, {})[article_id] = relevance
    return out
