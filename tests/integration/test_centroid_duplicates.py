"""Copies must not dilute later original reports in the online centroid."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.clustering.online import OnlineClusterer
from newstrace.config import get_settings
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.representations.embedder import from_blob
from tests.conftest import BASE_TIME


def _article(session: Session, *, parent: Article | None = None) -> Article:
    """Persist explicit duplicate flags; detection itself is outside this test."""
    row = session.query(Article).count() + 1
    article = Article(
        url=f"https://source{row}.example/report",
        canonical_url=f"https://source{row}.example/report",
        title=f"Report {row}",
        source_domain=f"source{row}.example",
        published_at=BASE_TIME + timedelta(minutes=row),
        normalized_text_hash=str(row),
        normalized_title_hash=str(row),
        is_near_duplicate=parent is not None,
        duplicate_of_article_id=parent.id if parent else None,
    )
    session.add(article)
    session.flush()
    return article


def _clusterer(session: Session, *, threshold: float = 0.3) -> OnlineClusterer:
    settings = get_settings().model_copy(
        update={"cluster_title_overlap_weight": 0.0, "cluster_entity_overlap_weight": 0.0}
    )
    return OnlineClusterer(session, settings=settings, threshold=threshold)


def _centroid(story: StoryCluster) -> np.ndarray:
    return from_blob(story.centroid_blob, story.centroid_dimension or 0)


@pytest.mark.parametrize("copies", [0, 1, 20])
@pytest.mark.parametrize("reload", [False, True])
def test_copies_do_not_dilute_the_next_original_or_split_its_followup(
    session: Session, copies: int, reload: bool
) -> None:
    clusterer = _clusterer(session)
    original = _article(session)
    first = clusterer.assign(original, np.array([1.0, 0.0], dtype=np.float32))
    story = session.get(StoryCluster, first.story_id)
    assert story is not None
    original_blob, original_version = story.centroid_blob, story.centroid_version

    for _ in range(copies):
        copy = _article(session, parent=original)
        assigned = clusterer.assign(copy, np.array([1.0, 0.0], dtype=np.float32))
        assert assigned.story_id == first.story_id
        assert assigned.method == "duplicate_link"
    assert story.centroid_blob == original_blob
    assert story.centroid_version == original_version
    assert story.article_count == 1 + copies
    assert story.distinct_domain_count == 1

    if reload:
        session.commit()
        session.expunge_all()
        clusterer = _clusterer(session)
        story = session.get(StoryCluster, first.story_id)
        assert story is not None

    second = _article(session)
    assigned = clusterer.assign(second, np.array([0.6, 0.8], dtype=np.float32))
    assert assigned.story_id == first.story_id
    expected = np.array([0.8, 0.4], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(_centroid(story), expected, atol=1e-6)

    # With copies in the denominator the y component falls below 0.3, and
    # this next original is incorrectly split into a new story.
    followup = _article(session)
    assigned = clusterer.assign(followup, np.array([0.0, 1.0], dtype=np.float32))
    assert assigned.story_id == first.story_id
    assert not assigned.created_new
    assert assigned.similarity == pytest.approx(float(expected[1]))
    assert story.article_count == copies + 3
    assert story.distinct_domain_count == 3
    assert story.centroid_version == original_version + 2

    expected += (np.array([0.0, 1.0], dtype=np.float32) - expected) / 3.0
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(_centroid(story), expected, atol=1e-6)
    before = (story.article_count, story.centroid_version, story.centroid_blob)
    clusterer.assign(followup, np.array([0.0, 1.0], dtype=np.float32))
    assert (story.article_count, story.centroid_version, story.centroid_blob) == before
    memberships = list(session.scalars(select(ClusterMembership)))
    assert len(memberships) == copies + 3
    assert {m.story_cluster_id for m in memberships} == {first.story_id}


def test_first_original_replaces_a_duplicate_only_seed(session: Session) -> None:
    clusterer = _clusterer(session, threshold=0.0)
    # A copy can arrive before its original has been clustered. It may seed a
    # provisional story but must not acquire weight in the first real update.
    original = _article(session)
    copy = _article(session, parent=original)
    first = clusterer.assign(copy, np.array([1.0, 0.0], dtype=np.float32))
    assert first.created_new
    story = session.get(StoryCluster, first.story_id)
    assert story is not None
    assert story.centroid_version == 1
    assigned = clusterer.assign(original, np.array([0.0, 2.0], dtype=np.float32))
    assert assigned.story_id == first.story_id
    np.testing.assert_allclose(_centroid(story), [0.0, 1.0], atol=1e-6)
    assert story.centroid_version == 2
    assert story.article_count == 2
    assert story.distinct_domain_count == 1


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.6, 0.8]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ],
)
def test_repeated_copy_bursts_match_original_only_replay_after_restart(
    session: Session, vectors: list[list[float]]
) -> None:
    """Compare public assignments, including same-domain and zero vectors."""
    story_ids: dict[str, int] = {}
    copy_count = 0
    for index, values in enumerate(vectors):
        # Reload ORM objects to exercise the persisted version convention.
        session.commit()
        session.expunge_all()
        clusterer = _clusterer(session, threshold=0.0)
        for topic in ("control", "syndicated"):
            article = _article(session)
            article.topic = topic
            article.source_domain = "one.example"
            assignment = clusterer.assign(article, np.array(values, dtype=np.float32))
            if topic not in story_ids:
                story_ids[topic] = assignment.story_id
            assert assignment.story_id == story_ids[topic]
            if topic == "syndicated":
                for _ in range(index + 2):
                    copy = _article(session, parent=article)
                    copied = clusterer.assign(copy, np.array(values, dtype=np.float32))
                    assert copied.method == "duplicate_link"
                    assert copied.story_id == assignment.story_id
                    copy_count += 1

        control = session.get(StoryCluster, story_ids["control"])
        syndicated = session.get(StoryCluster, story_ids["syndicated"])
        assert control is not None and syndicated is not None
        assert control.id != syndicated.id
        np.testing.assert_allclose(_centroid(syndicated), _centroid(control), atol=1e-6)
        assert np.isfinite(_centroid(syndicated)).all()
        assert control.centroid_version == syndicated.centroid_version == index + 2
        assert control.distinct_domain_count == syndicated.distinct_domain_count == 1
        assert control.article_count == index + 1
        assert syndicated.article_count == index + 1 + copy_count
