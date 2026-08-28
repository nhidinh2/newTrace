"""Threshold tuning on the fit/validation window only.

The clustering threshold is a fitted parameter.  Selecting it on the same
articles that report the final number would inflate the result, so this module
only ever sees the fit and validation periods; the test period is untouched.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.config import Settings, get_settings
from newstrace.evaluation.clustering import evaluate_clustering
from newstrace.evaluation.labels import load_story_labels
from newstrace.evaluation.retrieval import chronological_split
from newstrace.logging import get_logger
from newstrace.models import Article
from newstrace.representations.registry import load_vectors

logger = get_logger(__name__)

DEFAULT_GRID = tuple(round(x, 3) for x in np.arange(0.40, 1.00, 0.02))


@dataclass
class ThresholdResult:
    threshold: float
    bcubed_f1: float
    pairwise_f1: float
    predicted_clusters: int
    true_clusters: int

    def as_dict(self) -> dict[str, float]:
        return {
            "threshold": self.threshold,
            "bcubed_f1": round(self.bcubed_f1, 4),
            "pairwise_f1": round(self.pairwise_f1, 4),
            "predicted_clusters": float(self.predicted_clusters),
            "true_clusters": float(self.true_clusters),
        }


def tune_cluster_threshold(
    session: Session,
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    settings: Settings | None = None,
) -> tuple[ThresholdResult | None, list[ThresholdResult]]:
    """Sweep the clustering threshold over the fit+validation period."""
    from newstrace.evaluation.systems import _cluster_predictions_for

    settings = settings or get_settings()
    articles = list(session.execute(select(Article).order_by(Article.id)).scalars())
    labels = load_story_labels(articles)
    if not labels:
        logger.warning("No story labels available; cannot tune the clustering threshold")
        return None, []

    split = chronological_split(articles)
    split.assert_no_leakage()
    tune_ids = [aid for aid in (split.fit_ids + split.validation_ids) if aid in labels]
    if len(tune_ids) < 4:
        logger.warning("Not enough labelled fit/validation articles to tune")
        return None, []

    vectors = load_vectors(session, method="full")
    index = vectors.index_by_article
    rows = [index[a] for a in tune_ids if a in index]
    tune_ids = [a for a in tune_ids if a in index]
    matrix = vectors.matrix[rows]
    truth = {aid: labels[aid] for aid in tune_ids}

    results: list[ThresholdResult] = []
    for threshold in grid:
        probe = settings.model_copy(
            update={
                "cluster_threshold": float(threshold),
                "cluster_threshold_hashing": float(threshold),
            }
        )
        predicted = _cluster_predictions_for(
            session, tune_ids, matrix, settings=probe, method="full"
        )
        metrics = evaluate_clustering(truth, predicted)
        results.append(
            ThresholdResult(
                threshold=float(threshold),
                bcubed_f1=metrics.bcubed_f1,
                pairwise_f1=metrics.pairwise_f1,
                predicted_clusters=metrics.predicted_clusters,
                true_clusters=metrics.true_clusters,
            )
        )
    best = max(results, key=lambda r: (r.bcubed_f1, r.pairwise_f1, -r.threshold))
    logger.info(
        "Best clustering threshold on fit+validation: %.2f (B-cubed F1=%.3f)",
        best.threshold,
        best.bcubed_f1,
    )
    return best, results
