"""Experiment endpoints backed by :class:`~newstrace.models.EvaluationRun`."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from newstrace.db import get_db
from newstrace.evaluation.systems import ExperimentConfig, run_experiment
from newstrace.models import EvaluationRun
from newstrace.schemas import ExperimentRequest, ExperimentResponse
from newstrace.utils import ensure_utc

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _to_schema(run: EvaluationRun) -> ExperimentResponse:
    return ExperimentResponse(
        id=run.id,
        status=run.status,
        started_at=ensure_utc(run.started_at),  # type: ignore[arg-type]
        finished_at=ensure_utc(run.finished_at),
        artifact_path=run.artifact_path,
        config=run.config or {},
        metrics=run.metrics or {},
    )


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(
    payload: ExperimentRequest, session: Session = Depends(get_db)
) -> ExperimentResponse:
    config = ExperimentConfig(
        methods=list(payload.methods),
        dimensions=list(payload.dimensions),
        seed=payload.seed,
        top_k=payload.top_k,
        include_pagerank=payload.include_pagerank,
    )
    run = run_experiment(session, config)
    return _to_schema(run)


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    session: Session = Depends(get_db), limit: int = 20
) -> list[ExperimentResponse]:
    runs = session.execute(
        select(EvaluationRun).order_by(EvaluationRun.started_at.desc()).limit(limit)
    ).scalars()
    return [_to_schema(r) for r in runs]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
def get_experiment(experiment_id: int, session: Session = Depends(get_db)) -> ExperimentResponse:
    run = session.get(EvaluationRun, experiment_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"experiment {experiment_id} not found")
    return _to_schema(run)
