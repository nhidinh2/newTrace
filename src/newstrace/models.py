"""SQLAlchemy ORM models.

Embedding vectors are stored as a single serialized float32 blob per record --
never one column per dimension.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Article(Base, TimestampMixin):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    excerpt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    language: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    topic: Mapped[str | None] = mapped_column(String(128), index=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    normalized_text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    normalized_title_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    extraction_method: Mapped[str] = mapped_column(String(64), nullable=False, default="metadata")
    source_adapter: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    is_near_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duplicate_of_article_id: Mapped[int | None] = mapped_column(
        ForeignKey("articles.id", ondelete="SET NULL"), index=True
    )
    duplicate_reason: Mapped[str | None] = mapped_column(String(64))
    duplicate_similarity: Mapped[float | None] = mapped_column(Float)
    ingestion_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL")
    )
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    duplicate_of: Mapped[Article | None] = relationship(
        "Article", remote_side="Article.id", backref="duplicates"
    )
    memberships: Mapped[list[ClusterMembership]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    embeddings: Mapped[list[EmbeddingRecord]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("canonical_url", name="uq_articles_canonical_url"),
        Index("ix_articles_published_domain", "published_at", "source_domain"),
    )

    @property
    def body_text(self) -> str:
        """Best available text for embedding/summarisation (metadata only by default).

        The headline is terminated with a period so that sentence splitting -- and
        therefore claim extraction and timeline novelty -- does not glue the
        headline onto the first sentence of the body.
        """
        title = (self.title or "").strip()
        body = (self.excerpt or self.description or "").strip()
        if title and title[-1] not in ".!?":
            title += "."
        return " ".join(part for part in (title, body) if part)


class StoryCluster(Base, TimestampMixin):
    __tablename__ = "story_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    display_title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    topic: Mapped[str | None] = mapped_column(String(128), index=True)
    centroid_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    centroid_blob: Mapped[bytes | None] = mapped_column(LargeBinary)
    centroid_dimension: Mapped[int | None] = mapped_column(Integer)
    embedding_method: Mapped[str] = mapped_column(String(32), nullable=False, default="full")
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    distinct_domain_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    first_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    memberships: Mapped[list[ClusterMembership]] = relationship(
        back_populates="story", cascade="all, delete-orphan"
    )


class ClusterMembership(Base):
    __tablename__ = "cluster_memberships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True, nullable=False
    )
    story_cluster_id: Mapped[int] = mapped_column(
        ForeignKey("story_clusters.id", ondelete="CASCADE"), index=True, nullable=False
    )
    similarity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    assignment_method: Mapped[str] = mapped_column(String(64), nullable=False, default="online")
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    article: Mapped[Article] = relationship(back_populates="memberships")
    story: Mapped[StoryCluster] = relationship(back_populates="memberships")

    __table_args__ = (
        UniqueConstraint("article_id", "story_cluster_id", name="uq_membership_article_story"),
    )


class EmbeddingRecord(Base):
    __tablename__ = "embedding_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False, default="full")
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    fit_version: Mapped[str] = mapped_column(String(64), nullable=False, default="identity")
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    vector_blob: Mapped[bytes | None] = mapped_column(LargeBinary)
    vector_path: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    article: Mapped[Article] = relationship(back_populates="embeddings")

    __table_args__ = (
        UniqueConstraint(
            "article_id",
            "model_name",
            "method",
            "dimension",
            "fit_version",
            name="uq_embedding_identity",
        ),
    )


class Claim(Base, TimestampMixin):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    story_cluster_id: Mapped[int | None] = mapped_column(
        ForeignKey("story_clusters.id", ondelete="CASCADE"), index=True
    )
    normalized_claim: Mapped[str] = mapped_column(Text, nullable=False)
    claim_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    subject: Mapped[str | None] = mapped_column(Text)
    predicate: Mapped[str | None] = mapped_column(Text)
    object: Mapped[str | None] = mapped_column(Text)
    event_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extraction_method: Mapped[str] = mapped_column(String(64), nullable=False, default="rule")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    evidence: Mapped[list[ClaimEvidence]] = relationship(
        back_populates="claim", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("story_cluster_id", "claim_hash", name="uq_claim_story_hash"),
    )


class ClaimEvidence(Base):
    __tablename__ = "claim_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), index=True, nullable=False
    )
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True, nullable=False
    )
    excerpt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    excerpt_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    excerpt_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    stance: Mapped[str] = mapped_column(String(16), nullable=False, default="reports")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    claim: Mapped[Claim] = relationship(back_populates="evidence")
    article: Mapped[Article] = relationship()


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    topic: Mapped[str | None] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    git_commit: Mapped[str | None] = mapped_column(String(64))
    random_seed: Mapped[int | None] = mapped_column(Integer)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    fetched_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    inserted_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    errors: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(Text)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(64), nullable=False, default="experiment")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    git_commit: Mapped[str | None] = mapped_column(String(64))
    random_seed: Mapped[int | None] = mapped_column(Integer)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    errors: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[str | None] = mapped_column(Text)


class ProjectionArtifact(Base):
    """Saved fit metadata for a projector (SVD / RP) so runs are reproducible."""

    __tablename__ = "projection_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    fit_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_model: Mapped[str] = mapped_column(String(255), nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False, default=549)
    fit_article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fit_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    path: Mapped[str] = mapped_column(Text, nullable=False)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (UniqueConstraint("fit_version", name="uq_projection_fit_version"),)
