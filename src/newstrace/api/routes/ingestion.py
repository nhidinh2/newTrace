"""Ingestion run endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.api.serializers import run_to_schema
from newstrace.db import get_db
from newstrace.ingestion.pipeline import IngestOptions, run_ingestion
from newstrace.models import IngestionRun
from newstrace.schemas import IngestionRunRequest, IngestionRunResponse

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


@router.post("/runs", response_model=IngestionRunResponse, status_code=201)
def create_run(
    payload: IngestionRunRequest, session: Session = Depends(get_db)
) -> IngestionRunResponse:
    options = IngestOptions(
        source=payload.source,
        topic=payload.topic,
        timespan=payload.timespan,
        max_records=payload.max_records,
        windows=payload.windows,
        refresh=payload.refresh,
        index=payload.index,
    )
    run, _ = run_ingestion(session, options)
    return run_to_schema(run)


@router.get("/runs", response_model=list[IngestionRunResponse])
def list_runs(session: Session = Depends(get_db), limit: int = 20) -> list[IngestionRunResponse]:
    runs = session.execute(
        select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(limit)
    ).scalars()
    return [run_to_schema(run) for run in runs]


@router.get("/runs/{run_id}", response_model=IngestionRunResponse)
def get_run(run_id: int, session: Session = Depends(get_db)) -> IngestionRunResponse:
    run = session.get(IngestionRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"ingestion run {run_id} not found")
    return run_to_schema(run)
