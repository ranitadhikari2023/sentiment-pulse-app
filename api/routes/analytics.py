"""
Customer feedback intelligence analytics routes.
Provides comprehensive sentiment breakdown, temporal trends, and negative feedback monitoring.
No authentication required.
"""

from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from db.session import get_db
from services.analytics_service import AnalyticsService
from schemas.analytics import FeedbackAnalyticsResponse
from core.security import resolve_trusted_tenant_identity

router = APIRouter(prefix="/feedback/analytics", tags=["Feedback Analytics"])
api_router = APIRouter(prefix="/api/feedback/analytics", tags=["Feedback Analytics"])


def _analytics_handler(
    request: Request,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    db: Session
):
    user_id = resolve_trusted_tenant_identity(request)

    return AnalyticsService.get_analytics(
        db=db,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id
    )


for r in (router, api_router):
    @r.get("", response_model=FeedbackAnalyticsResponse)
    def get_analytics_endpoint(
        request: Request,
        start_date: Optional[datetime] = Query(None, description="Start date filter (ISO 8601)"),
        end_date: Optional[datetime] = Query(None, description="End date filter (ISO 8601)"),
        db: Session = Depends(get_db)
    ):
        """
        Retrieves business sentiment analytics, time trends, category distributions,
        and high-priority negative customer feedback.
        """
        return _analytics_handler(request, start_date, end_date, db)
