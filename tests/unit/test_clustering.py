"""Online clustering behaviour and clustering metrics."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.clustering.online import OnlineClusterer, combined_score, content_tokens
from newstrace.evaluation.clustering import (
    bcubed_scores,
    evaluate_clustering,
    fragmentation_and_merge,
    labels_from_pairs,
    pairwise_scores,
)
from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article, ClusterMembership, StoryCluster
from newstrace.pipeline import index_articles
from newstrace.representations.embedder import from_blob
from tests.conftest import BASE_TIME


def test_combined_score_boosts_and_clamps() -> None:
    assert combined_score(0.5, 0.0, 0.0, title_weight=0.2, entity_weight=0.1) == 0.5
    boosted = combined_score(0.5, 1.0, 1.0, title_weight=0.2, entity_weight=0.1)
    assert boosted == pytest.approx(0.8)
    assert combined_score(1.0, 1.0, 1.0, title_weight=0.2, entity_weight=0.1) == 1.0
    assert combined_score(-5.0, 0.0, 0.0, title_weight=0.2, entity_weight=0.1) == 0.0


def test_content_tokens_drops_stopwords() -> None:
    assert "the" not in content_tokens("The Atlas model")
    assert "atlas" in content_tokens("The Atlas model")


def test_same_event_articles_land_in_one_cluster(populated_session: Session) -> None:
    session = populated_session
    articles = {a.source_domain: a for a in session.query(Article).all()}
    memberships = {m.article_id: m.story_cluster_id for m in session.query(ClusterMembership).all()}
    alpha = memberships[articles["alpha.example"].id]
    beta = memberships[articles["beta.example"].id]
    wire = memberships[articles["wire.example"].id]
    gamma = memberships[articles["gamma.example"].id]

    assert alpha == beta, "two reports of the same event must share a story"
    assert wire == alpha, "a syndicated copy follows the article it duplicates"
    assert gamma != alpha, "an unrelated article must open its own story"


def test_duplicates_do_not_inflate_independent_source_counts(populated_session: Session) -> None:
    from newstrace.stories import load_story

    session = populated_session
    story_id = session.query(ClusterMembership).first().story_cluster_id
    bundle = load_story(session, story_id)
    assert bundle is not None
    domains = bundle.independent_domains
    assert "wire.example" not in domains
    story = session.get(StoryCluster, story_id)
    assert story is not None
    assert story.distinct_domain_count == len(domains)
    assert story.article_count > len(bundle.independent)


def test_threshold_controls_assignment(session: Session, raw_articles: list) -> None:
    result = persist_articles(session, raw_articles)
    session.commit()
    index_articles(session, result.article_ids)
    lenient = session.query(StoryCluster).count()

    # A threshold above 1.0 can never be met, so every article opens a story.
    session.query(ClusterMembership).delete()
    session.query(StoryCluster).delete()
    session.commit()
    clusterer = OnlineClusterer(session, threshold=1.01)
    for article in session.query(Article).order_by(Article.id).all():
        record = article.embeddings[0]
        clusterer.assign(article, from_blob(record.vector_blob, record.dimension))
    session.commit()
    strict = session.query(StoryCluster).count()

    # Near-duplicates always follow the article they duplicate, whatever the
    # threshold, so an unreachable threshold yields one story per *independent*
    # article rather than one per row.
    independent = session.query(Article).filter(Article.is_near_duplicate.is_(False)).count()
    assert strict == independent
    assert strict > lenient


def test_centroid_updates_incrementally(session: Session) -> None:
    raws = [
        RawArticle(
            url=f"https://d{i}.example/a",
            title="Northwind Labs releases Atlas 3 agent model",
            description="Northwind Labs said Atlas 3 scores 71.4 percent on the benchmark.",
            published_at=BASE_TIME + timedelta(hours=i),
        )
        for i in range(3)
    ]
    result = persist_articles(session, raws)
    session.commit()
    index_articles(session, result.article_ids)
    story = session.query(StoryCluster).first()
    assert story is not None
    assert story.centroid_version >= 1
    centroid = from_blob(story.centroid_blob, story.centroid_dimension or 0)
    assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-5)


def test_assignment_is_idempotent(populated_session: Session) -> None:
    session = populated_session
    before = session.query(ClusterMembership).count()
    ids = [a.id for a in session.query(Article).all()]
    index_articles(session, ids)
    assert session.query(ClusterMembership).count() == before


def test_pairwise_and_bcubed_on_perfect_clustering() -> None:
    truth = {1: 1, 2: 1, 3: 2, 4: 2}
    metrics = evaluate_clustering(truth, dict(truth))
    assert metrics.pairwise_f1 == 1.0
    assert metrics.bcubed_f1 == 1.0
    assert metrics.adjusted_rand_index == 1.0
    assert metrics.fragmentation_rate == 0.0
    assert metrics.merge_error_rate == 0.0


def test_fragmentation_and_merge_are_distinguished() -> None:
    truth = {1: 1, 2: 1, 3: 2, 4: 2}
    fragmented = {1: 10, 2: 11, 3: 12, 4: 12}
    merged = {1: 20, 2: 20, 3: 20, 4: 20}
    frag_rate, merge_rate = fragmentation_and_merge(truth, fragmented)
    assert frag_rate == 0.5 and merge_rate == 0.0
    frag_rate, merge_rate = fragmentation_and_merge(truth, merged)
    assert frag_rate == 0.0 and merge_rate == 1.0


def test_bcubed_penalises_a_single_giant_cluster() -> None:
    truth = {i: i % 4 for i in range(16)}
    everything_together = dict.fromkeys(truth, 0)
    precision, recall, _ = bcubed_scores(truth, everything_together)
    assert recall == 1.0
    assert precision < 0.5


def test_pairwise_scores_on_empty_input() -> None:
    assert pairwise_scores({}, {}) == (0.0, 0.0, 0.0)


def test_labels_from_pairs_builds_transitive_groups() -> None:
    labels = labels_from_pairs([(1, 2, 1), (2, 3, 1), (4, 5, 0)])
    assert labels[1] == labels[2] == labels[3]
    assert labels[4] != labels[5]


# -- silver relevance judgments ---------------------------------------------


def test_silver_judgments_exclude_near_duplicates(populated_session: Session) -> None:
    """A syndicated copy must never be what makes a query "answered"."""
    from newstrace.evaluation.systems import silver_judgments

    articles = list(populated_session.execute(select(Article)).scalars())
    duplicates = {a.id for a in articles if a.is_near_duplicate}
    assert duplicates, (
        "the fixture set must contain a near-duplicate for this test to mean anything"
    )

    queries = [(f"q{a.id}", a.id) for a in articles]
    judgments, source = silver_judgments(populated_session, queries, story_labels={})

    assert source == "silver_from_full_dim_clustering_excluding_near_duplicates"
    for relevant in judgments.values():
        assert not (relevant & duplicates), "duplicates leaked into a relevant set"


def test_silver_judgments_prefer_curated_story_labels(populated_session: Session) -> None:
    from newstrace.evaluation.systems import silver_judgments

    articles = list(populated_session.execute(select(Article)).scalars())
    labels = {a.id: 1 for a in articles[:2]}
    queries = [(f"q{a.id}", a.id) for a in articles]
    judgments, source = silver_judgments(populated_session, queries, story_labels=labels)

    assert source == "synthetic_fixture_labels"
    assert judgments, "curated labels should still produce judgments"
