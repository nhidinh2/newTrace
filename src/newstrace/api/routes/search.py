"""Search endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from newstrace.api.serializers import search_item_to_schema
from newstrace.db import get_db
from newstrace.schemas import SearchRequestModel, SearchResponseModel
from newstrace.search import SearchFilters, SearchRequest, search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponseModel)
def post_search(
    payload: SearchRequestModel, session: Session = Depends(get_db)
) -> SearchResponseModel:
    request = SearchRequest(
        query=payload.query,
        filters=SearchFilters(
            topic=payload.topic,
            start_time=payload.start_time,
            end_time=payload.end_time,
            source_domain=payload.source_domain,
            language=payload.language,
            entity=payload.entity,
            include_duplicates=payload.include_duplicates,
        ),
        method=payload.method,
        dimension=payload.dimension,
        fit_version=payload.fit_version,
        rerank_with_pagerank=payload.rerank_with_pagerank,
        top_k=payload.top_k,
    )
    response = search(session, request)
    return SearchResponseModel(
        query=response.query,
        method=response.method,
        took_ms=round(response.took_ms, 3),
        candidates_considered=response.candidates_considered,
        results=[search_item_to_schema(item) for item in response.items],
        notes=response.notes,
    )
