"""Bad embedding inputs must not leave durable clustering state behind."""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy.orm import Session

from newstrace.clustering.online import OnlineClusterer
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.representations.embedder import from_blob


def articles_for(session: Session, raw_articles: list) -> list[Article]:
    result = persist_articles(session, raw_articles)
    session.commit()
    return [session.get(Article, article_id) for article_id in result.article_ids]


def snapshot(session: Session) -> tuple:
    session.flush()
    return (
        [
            (
                s.id,
                s.centroid_blob,
                s.centroid_version,
                s.article_count,
                s.extra,
                s.first_published_at,
                s.last_published_at,
                s.updated_at,
            )
            for s in session.query(StoryCluster).order_by(StoryCluster.id)
        ],
        [
            (m.article_id, m.story_cluster_id, m.similarity)
            for m in session.query(ClusterMembership).order_by(ClusterMembership.id)
        ],
    )


@pytest.mark.parametrize(
    "vector",
    [
        np.array([np.nan, 0], dtype=np.float32),
        np.array([np.inf, 0], dtype=np.float32),
        np.array([-np.inf, 0], dtype=np.float32),
        np.array([], dtype=np.float32),
        np.array(1, dtype=np.float32),
        np.array([[1, 0]], dtype=np.float32),
        np.array([1 + 2j, 0]),
        np.array([1e40, 0], dtype=np.float64),
        np.array(["bad", "vector"]),
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_vector_has_no_clustering_side_effects(
    session: Session, raw_articles: list, vector: np.ndarray, existing: bool
) -> None:
    articles = articles_for(session, raw_articles)
    clusterer = OnlineClusterer(session, threshold=0)
    if existing:
        clusterer.assign(articles[0], np.array([1, 0], dtype=np.float32))
        session.commit()
    before = snapshot(session)
    with pytest.raises(ValueError, match="vector"):
        clusterer.assign(articles[1], vector)
    session.commit()
    assert snapshot(session) == before


@pytest.mark.parametrize(
    "vectors",
    [
        np.ones((3, 2), dtype=np.float32),  # too few rows: old code fails after writes
        np.ones((5, 2), dtype=np.float32),  # extra rows used to be silently ignored
        np.ones(4, dtype=np.float32),
        np.ones((4, 0), dtype=np.float32),
        np.ones((4, 1, 2), dtype=np.float32),
        np.array([[1, 0], [1, 0], [np.nan, 0], [1, 0]], dtype=np.float32),
        np.array([[1, 0], [1, 0], [np.inf, 0], [1, 0]], dtype=np.float32),
    ],
)
def test_batch_preflight_precedes_first_assignment(
    session: Session, raw_articles: list, vectors: np.ndarray
) -> None:
    articles = articles_for(session, raw_articles)
    before = snapshot(session)
    with pytest.raises(ValueError, match="vector"):
        OnlineClusterer(session, threshold=0).assign_many(articles, vectors)
    session.commit()
    assert snapshot(session) == before


def test_dimension_mismatch_on_duplicate_link_is_rejected(
    session: Session, raw_articles: list
) -> None:
    articles = articles_for(session, raw_articles)
    clusterer = OnlineClusterer(session, threshold=0)
    parent = articles[0]
    copy = next(a for a in articles if a.duplicate_of_article_id == parent.id)
    clusterer.assign(parent, np.array([1, 0], dtype=np.float32))
    before = snapshot(session)
    with pytest.raises(ValueError, match="dimension"):
        clusterer.assign(copy, np.array([1, 0, 0], dtype=np.float32))
    session.commit()
    assert snapshot(session) == before


def test_valid_batch_preserves_alignment_zero_vectors_and_idempotency(
    session: Session, raw_articles: list
) -> None:
    articles = articles_for(session, raw_articles)
    clusterer = OnlineClusterer(session, threshold=1.01)
    vectors = np.array([[1, 0], [0, 1], [1, 0], [0, 0]], dtype=np.float32)
    assignments = clusterer.assign_many(articles[::-1], vectors[::-1])
    session.commit()
    by_id = {a.article_id: a.story_id for a in assignments}
    for article, vector in zip(articles, vectors, strict=True):
        if article.is_near_duplicate:
            continue
        story = session.get(StoryCluster, by_id[article.id])
        np.testing.assert_array_equal(from_blob(story.centroid_blob, 2), vector)
    before = snapshot(session)
    clusterer.assign_many(articles, vectors)
    session.commit()
    assert snapshot(session) == before
    assert clusterer.assign_many([], np.empty((0, 2), dtype=np.float32)) == []


def test_large_finite_vectors_do_not_overflow_scoring_or_updates(
    session: Session, raw_articles: list
) -> None:
    articles = articles_for(session, raw_articles)
    clusterer = OnlineClusterer(session, threshold=0.9)
    vector = np.array([3e38, 0], dtype=np.float32)
    first = clusterer.assign(articles[0], vector)
    second = clusterer.assign(articles[1], vector)
    assert first.story_id == second.story_id
    assert second.similarity == pytest.approx(1)
    session.commit()
    story = session.get(StoryCluster, first.story_id)
    np.testing.assert_allclose(from_blob(story.centroid_blob, 2), [1, 0])


def test_existing_corrupt_centroid_fails_without_mutation(
    session: Session, raw_articles: list
) -> None:
    articles = articles_for(session, raw_articles)
    clusterer = OnlineClusterer(session, threshold=0)
    first = clusterer.assign(articles[0], np.array([1, 0], dtype=np.float32))
    story = session.get(StoryCluster, first.story_id)
    story.centroid_blob = np.array([np.nan, 0], dtype=np.float32).tobytes()
    session.commit()
    before = snapshot(session)
    with pytest.raises(ValueError, match="nonfinite centroid"):
        clusterer.assign(articles[1], np.array([1, 0], dtype=np.float32))
    session.commit()
    assert snapshot(session) == before


def test_candidate_dimension_error_preserves_story_and_allows_retry(
    session: Session, raw_articles: list
) -> None:
    articles = articles_for(session, raw_articles)
    clusterer = OnlineClusterer(session, threshold=0)
    vector = np.array([1, 0], dtype=np.float32)
    first = clusterer.assign(articles[0], vector)
    session.commit()
    before = snapshot(session)
    with pytest.raises(ValueError, match="dimension"):
        clusterer.assign(articles[1], np.array([1, 0, 0], dtype=np.float32))
    session.commit()
    assert snapshot(session) == before
    retry = clusterer.assign(articles[1], vector)
    assert retry.story_id == first.story_id
    session.commit()
