"""Configuration loading, database idempotency and LLM defaults."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from newstrace.config import (
    SourcePolicy,
    get_llm_settings,
    get_settings,
    load_source_policy,
    load_topics,
)
from newstrace.ingestion.base import RawArticle
from newstrace.ingestion.pipeline import persist_articles
from newstrace.models import Article, ClusterMembership, EmbeddingRecord, StoryCluster
from tests.conftest import BASE_TIME


def test_defaults_require_no_api_key() -> None:
    assert get_llm_settings().llm_provider == "none"
    settings = get_settings()
    assert settings.random_seed == 549
    assert settings.embedding_model.startswith("sentence-transformers/")


def test_topics_load_from_yaml() -> None:
    topics = load_topics()
    assert "ai_models" in topics
    topic = topics["ai_models"]
    assert topic.gdelt_query
    assert topic.label()
    assert isinstance(topic.feeds, list)


def test_missing_topics_file_is_not_fatal(tmp_path: Path) -> None:
    assert load_topics(tmp_path / "nope.yaml") == {}


def test_source_policy_defaults_disable_extraction() -> None:
    policy = load_source_policy()
    assert policy.extraction.enabled is False
    assert policy.extraction.allows("example.com") is False


def test_extraction_allowlist_is_respected() -> None:
    policy = SourcePolicy.model_validate(
        {"extraction": {"enabled": True, "allowlist": ["Allowed.example"]}}
    )
    assert policy.extraction.allows("allowed.example")
    assert not policy.extraction.allows("other.example")


def test_rate_limit_lookup() -> None:
    policy = SourcePolicy.model_validate(
        {"rate_limits": {"default_min_interval_seconds": 2.0, "per_domain": {"slow.example": 9.0}}}
    )
    assert policy.min_interval_for("slow.example", 1.0) == 9.0
    assert policy.min_interval_for("other.example", 1.0) == 2.0


def test_settings_resolve_relative_paths() -> None:
    settings = get_settings()
    resolved = settings.resolve(Path("configs/topics.example.yaml"))
    assert resolved.is_absolute()
    assert resolved.exists()


def test_canonical_url_uniqueness_is_enforced(session: Session) -> None:
    raw = RawArticle(url="https://alpha.example/a", title="One", published_at=BASE_TIME)
    persist_articles(session, [raw])
    session.commit()
    session.add(
        Article(
            url="https://alpha.example/a",
            canonical_url="https://alpha.example/a",
            source_domain="alpha.example",
            normalized_text_hash="x",
            normalized_title_hash="y",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_membership_uniqueness_is_enforced(populated_session: Session) -> None:
    membership = populated_session.query(ClusterMembership).first()
    populated_session.add(
        ClusterMembership(
            article_id=membership.article_id, story_cluster_id=membership.story_cluster_id
        )
    )
    with pytest.raises(IntegrityError):
        populated_session.commit()
    populated_session.rollback()


def test_embedding_identity_uniqueness_is_enforced(populated_session: Session) -> None:
    record = populated_session.query(EmbeddingRecord).first()
    populated_session.add(
        EmbeddingRecord(
            article_id=record.article_id,
            model_name=record.model_name,
            method=record.method,
            dimension=record.dimension,
            fit_version=record.fit_version,
            cache_key="dup",
        )
    )
    with pytest.raises(IntegrityError):
        populated_session.commit()
    populated_session.rollback()


def test_body_text_keeps_the_headline_a_separate_sentence(populated_session: Session) -> None:
    article = populated_session.query(Article).first()
    assert article is not None
    assert ". " in article.body_text
    assert article.body_text.startswith(article.title.rstrip("."))


def test_deleting_a_story_cascades_to_memberships(populated_session: Session) -> None:
    story = populated_session.query(StoryCluster).first()
    story_id = story.id
    populated_session.delete(story)
    populated_session.commit()
    assert (
        populated_session.query(ClusterMembership)
        .filter(ClusterMembership.story_cluster_id == story_id)
        .count()
        == 0
    )
