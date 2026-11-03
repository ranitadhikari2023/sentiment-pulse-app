"""
System health, liveness, and readiness monitoring endpoints.
"""

import os
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.session import get_db
from db.models import Feedback, ImportJob, FeedbackStaging
from services.sentiment_service import sentiment_service
from schemas.health import HealthResponse, ReadinessResponse
from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["System"])


def load_model_metrics() -> Dict[str, Any]:
    if os.path.exists(settings.METRICS_PATH):
        try:
            with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics file: {e}")
    return {}


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def healthcheck():
    """
    Standard health check endpoint for Render monitoring and uptime checks.
    Includes model evaluation metadata and benchmarked inference latency.
    """
    metrics = load_model_metrics()
    model_version = metrics.get("version", "3.0.0")
    is_loaded = sentiment_service.is_ready()

    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded,
        version=model_version,
        model_version=model_version,
        training_date=metrics.get("training_date"),
        dataset_name=metrics.get("dataset_name"),
        test_macro_f1=metrics.get("test_macro_f1"),
        inference_latency_ms=sentiment_service.benchmark_latency_ms
    )


@router.get("/health/live")
def liveness_check():
    """
    Lightweight liveness probe indicating process is up and responsive.
    Does not trigger database or inference operations.
    """
    return {"status": "alive"}


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe verifying database connectivity and model initialization.
    Returns 503 if any core subsystem is unavailable.
    """
    # 1. Check DB connectivity and core schema tables
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
        db.query(Feedback).first()
        db.query(ImportJob).first()
        db.query(FeedbackStaging).first()
    except Exception as e:
        logger.error(f"Readiness probe DB connection or schema query failed: {e}", exc_info=True)
        db_status = f"unhealthy: {str(e)}"

    # 2. Check Model availability
    model_status = "ok" if sentiment_service.is_ready() else "model_not_ready"

    is_ready = (db_status == "ok") and (model_status == "ok")
    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "database": db_status,
                "model": model_status,
                "ready": False
            }
        )

    return ReadinessResponse(
        status="ready",
        database=db_status,
        model=model_status,
        ready=True
    )
