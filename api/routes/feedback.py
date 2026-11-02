"""
Customer feedback management routes.
Provides persistent feedback creation, querying, pagination, filtering, deletion, statistics, and secure CSV export.
No authentication required.
"""

from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response, Request
from sqlalchemy.orm import Session

from db.session import get_db
from services.feedback_service import FeedbackService
from services.import_service import ImportService
from core.security import resolve_trusted_tenant_identity
from schemas.feedback import (
    FeedbackCreate,
    FeedbackResponse,
    FeedbackListResponse,
    FeedbackStatsResponse
)
from core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["Customer Feedback"])
api_router = APIRouter(prefix="/api/feedback", tags=["Customer Feedback"])


def _create_feedback_handler(request: Request, payload: FeedbackCreate, db: Session):
    try:
        user_id = resolve_trusted_tenant_identity(request)
        feedback = FeedbackService.create_feedback(db, payload, user_id=user_id)
        return FeedbackResponse.from_orm_model(feedback)
    except ValueError as ve:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error persisting feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist feedback record."
        )


def _list_feedback_handler(
    request: Request,
    page: int,
    page_size: int,
    sentiment: Optional[str],
    category: Optional[str],
    source: Optional[str],
    min_confidence: Optional[float],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    search: Optional[str],
    sort_by: str,
    sort_order: str,
    db: Session
):
    user_id = resolve_trusted_tenant_identity(request)

    return FeedbackService.list_feedback(
        db=db,
        page=page,
        page_size=page_size,
        sentiment=sentiment,
        category=category,
        source=source,
        min_confidence=min_confidence,
        start_date=start_date,
        end_date=end_date,
        search=search,
        sort_by=sort_by,
        sort_order=sort_order,
        user_id=user_id
    )


def _get_feedback_by_id_handler(request: Request, feedback_id: int, db: Session):
    fb = FeedbackService.get_feedback_by_id(db, feedback_id)
    if not fb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback record {feedback_id} not found."
        )

    user_id = resolve_trusted_tenant_identity(request)
    if fb.user_id is not None:
        if user_id is None or user_id != fb.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Feedback record belongs to another tenant."
            )

    return FeedbackResponse.from_orm_model(fb)


def _delete_feedback_handler(request: Request, feedback_id: int, db: Session):
    fb = FeedbackService.get_feedback_by_id(db, feedback_id)
    if not fb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback record {feedback_id} not found."
        )

    user_id = resolve_trusted_tenant_identity(request)
    if fb.user_id is not None:
        if user_id is None or user_id != fb.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Feedback record belongs to another tenant."
            )

    success = FeedbackService.delete_feedback(db, feedback_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feedback record {feedback_id} not found."
        )
    return {"message": f"Feedback record {feedback_id} deleted successfully."}


def _stats_handler(request: Request, db: Session):
    user_id = resolve_trusted_tenant_identity(request)
    return FeedbackService.get_feedback_stats(db, user_id=user_id)


def _export_handler(request: Request, db: Session):
    user_id = resolve_trusted_tenant_identity(request)
    csv_content = ImportService.generate_csv_export(db, user_id=user_id)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"feedback_export_{timestamp_str}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# Attach routes to both /feedback and /api/feedback for seamless client compatibility
for r in (router, api_router):
    @r.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
    def create_feedback_endpoint(
        request: Request,
        payload: FeedbackCreate,
        db: Session = Depends(get_db)
    ):
        """Creates feedback, infers sentiment, and persists record."""
        return _create_feedback_handler(request, payload, db)

    @r.get("/stats", response_model=FeedbackStatsResponse)
    def get_stats_endpoint(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """Returns sentiment statistics aggregated across feedback records."""
        return _stats_handler(request, db)

    @r.get("/export")
    def export_feedback_endpoint(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """Exports feedback as CSV with formula injection protection."""
        return _export_handler(request, db)

    @r.get("", response_model=FeedbackListResponse)
    def list_feedback_endpoint(
        request: Request,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        sentiment: Optional[str] = Query(None, description="Filter by sentiment (positive, neutral, negative)"),
        category: Optional[str] = Query(None, description="Filter by category"),
        source: Optional[str] = Query(None, description="Filter by source"),
        min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence"),
        start_date: Optional[datetime] = Query(None, description="Start date (ISO 8601)"),
        end_date: Optional[datetime] = Query(None, description="End date (ISO 8601)"),
        search: Optional[str] = Query(None, description="Text search substring"),
        sort_by: str = Query("created_at", description="Sort column (created_at, confidence, rating)"),
        sort_order: str = Query("desc", description="Sort direction (asc, desc)"),
        db: Session = Depends(get_db)
    ):
        """Retrieves paginated feedback records."""
        return _list_feedback_handler(
            request, page, page_size, sentiment, category, source, min_confidence,
            start_date, end_date, search, sort_by, sort_order, db
        )

    @r.get("/{feedback_id}", response_model=FeedbackResponse)
    def get_feedback_endpoint(
        request: Request,
        feedback_id: int,
        db: Session = Depends(get_db)
    ):
        """Retrieves a single feedback record."""
        return _get_feedback_by_id_handler(request, feedback_id, db)

    @r.delete("/{feedback_id}")
    def delete_feedback_endpoint(
        request: Request,
        feedback_id: int,
        db: Session = Depends(get_db)
    ):
        """Deletes a feedback record."""
        return _delete_feedback_handler(request, feedback_id, db)
