"""Regression coverage for the dimensionality experiment runner."""

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newstrace.config import Settings
from newstrace.evaluation.systems import _cluster_predictions_for
from newstrace.models import Article


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("date_mode", ["mixed", "dated", "undated"])
def test_cluster_predictions_date_order_and_vector_alignment(
    session: Session, date_mode: str, reverse: bool
) -> None:
    # IDs 2 and 4 represent the same instant, expressed as naive UTC and +02:00.
    early = datetime(2026, 10, 20, 8)
    dates = [
        None,
        early,
        None,
        datetime(2026, 10, 20, 10, tzinfo=timezone(timedelta(hours=2))),
        datetime(2026, 10, 20, 9, tzinfo=UTC),
    ]
    if date_mode == "dated":
        dates[0] = datetime(2026, 10, 21, tzinfo=UTC)
        dates[2] = datetime(2026, 10, 22, tzinfo=UTC)
    elif date_mode == "undated":
        dates = [None] * 5
    articles = [
        Article(
            id=aid,
            url=f"https://example.test/{aid}",
            canonical_url=f"https://example.test/{aid}",
            title="Independent report",
            source_domain="example.test",
            normalized_text_hash=str(aid),
            normalized_title_hash=str(aid),
            published_at=date,
        )
        for aid, date in enumerate(dates, start=1)
    ]
    session.add_all(articles)
    # Flush without expiring: SQLite round-trips discard timezone offsets.
    session.flush()
    ids = [article.id for article in articles]
    if reverse:
        ids.reverse()
    vectors = {1: [0, 1], 2: [1, 0], 3: [0, 1], 4: [1, 0], 5: [1, 0]}
    settings = Settings(
        embedding_backend="hashing",
        cluster_threshold_hashing=0.8,
        cluster_title_overlap_weight=0.0,
        cluster_entity_overlap_weight=0.0,
    )

    predictions = _cluster_predictions_for(
        session,
        ids,
        np.array([vectors[aid] for aid in ids], dtype=np.float32),
        settings=settings,
        method="full",
    )

    if date_mode == "undated":
        assert list(predictions) == [1, 2, 3, 4, 5]
        assert predictions == {1: 0, 2: 1, 3: 0, 4: 1, 5: 1}
    else:
        assert list(predictions) == [2, 4, 5, 1, 3]
        assert predictions == {2: 0, 4: 0, 5: 0, 1: 1, 3: 1}


def test_cluster_predictions_empty_population(session: Session) -> None:
    assert (
        _cluster_predictions_for(
            session,
            [],
            np.empty((0, 2), dtype=np.float32),
            settings=Settings(embedding_backend="hashing"),
            method="full",
        )
        == {}
    )
