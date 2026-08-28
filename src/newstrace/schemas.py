"""Pydantic v2 request/response schemas for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from newstrace.summarization.base import Evidence, StorySummary, SummaryStatement

RetrievalMethod = Literal["full", "svd", "gaussian_rp", "sparse_rp", "tfidf"]

DEFAULT_EXPERIMENT_METHODS: list[RetrievalMethod] = ["full", "svd", "gaussian_rp", "sparse_rp"]
DEFAULT_EXPERIMENT_DIMENSIONS: list[int] = [32, 64, 128, 256]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    version: str
    environment: str
    database: str
    embedding_model: str
    embedding_backend: str
    embedding_dimension: int
    device: str
    llm_provider: str
    article_count: int
    story_count: int
    duplicate_count: int
    last_ingestion_run: dict[str, Any] | None = None
    representations: list[dict[str, Any]] = Field(default_factory=list)


class TopicResponse(BaseModel):
    key: str
    display_name: str
    gdelt_query: str
    feed_count: int
    article_count: int
    story_count: int


class ArticleResponse(BaseModel):
    id: int
    title: str
    url: str
    source_domain: str
    language: str
    topic: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    excerpt: str = ""
    is_near_duplicate: bool = False
    duplicate_of_article_id: int | None = None
    extraction_method: str = "metadata"


class StorySummaryResponse(BaseModel):
    id: int
    display_title: str
    topic: str | None = None
    article_count: int
    independent_article_count: int
    distinct_domain_count: int
    duplicate_count: int
    first_published_at: datetime | None = None
    last_published_at: datetime | None = None
    keywords: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class StoryDetailResponse(StorySummaryResponse):
    articles: list[ArticleResponse] = Field(default_factory=list)


class TimelineEntryResponse(BaseModel):
    article: ArticleResponse
    published_at: datetime | None = None
    similarity: float = 0.0
    is_near_duplicate: bool = False
    duplicate_of_article_id: int | None = None
    is_first_report: bool = False
    new_sentences: list[str] = Field(default_factory=list)
    repeated_sentences: list[str] = Field(default_factory=list)


class TimelineResponse(BaseModel):
    story_id: int
    display_title: str
    entries: list[TimelineEntryResponse] = Field(default_factory=list)
    note: str = (
        "Ordering reflects publication timestamps in this dataset only; the earliest "
        "available report is not proof of who reported a claim first."
    )


class ClaimEvidenceResponse(BaseModel):
    article_id: int
    title: str
    source_domain: str
    url: str
    published_at: datetime | None = None
    excerpt: str
    stance: str
    confidence: float
    is_near_duplicate: bool = False


class ClaimResponse(BaseModel):
    id: int
    normalized_claim: str
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    event_time: datetime | None = None
    extraction_method: str
    confidence: float
    status: Literal["repeated", "single_source", "unclear"]
    independent_domain_count: int
    duplicate_evidence_count: int
    evidence: list[ClaimEvidenceResponse] = Field(default_factory=list)


class ClaimsResponse(BaseModel):
    story_id: int
    claims: list[ClaimResponse] = Field(default_factory=list)
    note: str = (
        "NewsTrace reports repetition, provenance and disagreement between sources. "
        "It does not assess whether a claim is true or whether a publisher is reliable."
    )


class SearchFiltersRequest(BaseModel):
    topic: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    source_domain: str | None = None
    language: str | None = None
    entity: str | None = None
    include_duplicates: bool = False


class SearchRequestModel(BaseModel):
    query: str = Field(min_length=1)
    topic: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    source_domain: str | None = None
    language: str | None = None
    entity: str | None = None
    include_duplicates: bool = False
    method: RetrievalMethod = "full"
    dimension: int | None = None
    fit_version: str | None = None
    rerank_with_pagerank: bool = False
    top_k: int = Field(default=10, ge=1, le=100)


class SearchResultResponse(BaseModel):
    article_id: int
    story_id: int | None = None
    title: str
    source_domain: str
    published_at: datetime | None = None
    url: str
    score: float
    ranking_method: str
    signals: dict[str, float] = Field(default_factory=dict)
    excerpt: str
    explanation: str
    is_near_duplicate: bool = False


class SearchResponseModel(BaseModel):
    query: str
    method: str
    took_ms: float
    candidates_considered: int
    results: list[SearchResultResponse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class IngestionRunRequest(BaseModel):
    source: Literal["fixtures", "gdelt", "rss"] = "fixtures"
    topic: str | None = None
    timespan: str = "24h"
    max_records: int = Field(default=250, ge=1, le=250)
    windows: int = Field(default=1, ge=1, le=96)
    refresh: bool = False
    index: bool = True


class IngestionRunResponse(BaseModel):
    id: int
    source: str
    topic: str | None = None
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    fetched_count: int
    inserted_count: int
    duplicate_count: int
    skipped_count: int
    failure_count: int
    errors: dict[str, Any] = Field(default_factory=dict)
    git_commit: str | None = None
    random_seed: int | None = None


class SummaryRequest(BaseModel):
    method: Literal["extractive", "llm"] = "extractive"
    max_per_section: int = Field(default=5, ge=1, le=20)


class SummaryResponse(BaseModel):
    summary: StorySummary
    citation_coverage: float
    statement_count: int


class ExperimentRequest(BaseModel):
    methods: list[RetrievalMethod] = Field(default_factory=lambda: list(DEFAULT_EXPERIMENT_METHODS))
    dimensions: list[int] = Field(default_factory=lambda: list(DEFAULT_EXPERIMENT_DIMENSIONS))
    seed: int = 549
    top_k: int = 10
    include_pagerank: bool = True


class ExperimentResponse(BaseModel):
    id: int
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    artifact_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ArticleResponse",
    "ClaimResponse",
    "ClaimsResponse",
    "Evidence",
    "ExperimentRequest",
    "ExperimentResponse",
    "HealthResponse",
    "IngestionRunRequest",
    "IngestionRunResponse",
    "SearchFiltersRequest",
    "SearchRequestModel",
    "SearchResponseModel",
    "SearchResultResponse",
    "StoryDetailResponse",
    "StorySummary",
    "StorySummaryResponse",
    "SummaryRequest",
    "SummaryResponse",
    "SummaryStatement",
    "TimelineResponse",
    "TopicResponse",
]
