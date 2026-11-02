"""
Service layer for customer feedback management.
Handles persistent feedback creation, querying, pagination, filtering, statistics, and deletion.
Authentication and user-isolation checks have been permanently removed.
"""

from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone
import math
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc

from db.models import Feedback
from schemas.feedback import FeedbackCreate, FeedbackListResponse, FeedbackResponse, FeedbackStatsResponse
from services.sentiment_service import sentiment_service
from core.logging import get_logger

logger = get_logger(__name__)


class FeedbackService:
    @staticmethod
    def create_feedback(db: Session, payload: FeedbackCreate, user_id: Optional[int] = None) -> Feedback:
        """
        Runs sentiment inference on incoming customer feedback, persists record,
        and returns the created model.
        """
        prediction = sentiment_service.predict(payload.text)
        scores = prediction.get("scores", {})

        clean_category = payload.category.strip() if payload.category and payload.category.strip() else None
        clean_source = payload.source.strip() if payload.source and payload.source.strip() else "web"

        feedback = Feedback(
            user_id=user_id,
            text=payload.text,
            clean_text=prediction.get("text", payload.text),
            sentiment=prediction.get("sentiment", "unknown"),
            confidence=float(prediction.get("confidence", 0.0)),
            positive_score=float(scores.get("positive", 0.0)),
            neutral_score=float(scores.get("neutral", 0.0)),
            negative_score=float(scores.get("negative", 0.0)),
            category=clean_category,
            source=clean_source,
            rating=payload.rating,
            model_version="3.0.0",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        try:
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            logger.info(f"Persisted feedback {feedback.id} with sentiment '{feedback.sentiment}'.")
            return feedback
        except Exception as exc:
            db.rollback()
            logger.error(f"Failed to persist feedback in database: {exc}", exc_info=True)
            raise

    @staticmethod
    def get_feedback_by_id(db: Session, feedback_id: int) -> Optional[Feedback]:
        """
        Retrieves a feedback record by ID.
        """
        return db.query(Feedback).filter(Feedback.id == feedback_id).first()

    @staticmethod
    def delete_feedback(db: Session, feedback_id: int) -> bool:
        """
        Deletes a feedback record by ID.
        """
        record = db.query(Feedback).filter(Feedback.id == feedback_id).first()
        if not record:
            return False

        db.delete(record)
        db.commit()
        logger.info(f"Deleted feedback record {feedback_id}.")
        return True

    @staticmethod
    def list_feedback(
        db: Session,
        page: int = 1,
        page_size: int = 20,
        sentiment: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        search: Optional[str] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        user_id: Optional[int] = None
    ) -> FeedbackListResponse:
        """
        Retrieves paginated feedback records with multi-criteria filtering across the database.
        """
        page = max(1, page)
        page_size = min(max(1, page_size), 100)  # Bound between 1 and 100

        query = db.query(Feedback)

        if user_id is not None:
            query = query.filter(Feedback.user_id == user_id)

        # Filters
        if sentiment:
            query = query.filter(Feedback.sentiment == sentiment.lower().strip())
        if category:
            query = query.filter(Feedback.category == category.strip())
        if source:
            query = query.filter(Feedback.source == source.strip())
        if min_confidence is not None:
            query = query.filter(Feedback.confidence >= min_confidence)
        if start_date:
            query = query.filter(Feedback.created_at >= start_date)
        if end_date:
            query = query.filter(Feedback.created_at <= end_date)
        if search:
            query = query.filter(Feedback.text.ilike(f"%{search.strip()}%"))

        total = query.count()
        total_pages = math.ceil(total / page_size) if total > 0 else 1

        # Sorting
        sort_column = getattr(Feedback, sort_by, Feedback.created_at)
        order_func = desc if sort_order.lower() == "desc" else asc
        query = query.order_by(order_func(sort_column))

        # Pagination offset
        offset = (page - 1) * page_size
        records = query.offset(offset).limit(page_size).all()

        items = [FeedbackResponse.from_orm_model(r) for r in records]
        return FeedbackListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )

    @staticmethod
    def get_feedback_stats(db: Session, user_id: Optional[int] = None) -> FeedbackStatsResponse:
        """
        Aggregates summary sentiment statistics across all feedback records.
        """
        base_query = db.query(Feedback)
        if user_id is not None:
            base_query = base_query.filter(Feedback.user_id == user_id)

        # Count by sentiment
        counts = base_query.with_entities(
            Feedback.sentiment,
            func.count(Feedback.id)
        ).group_by(
            Feedback.sentiment
        ).all()

        sentiment_map = {row[0]: row[1] for row in counts}
        pos = sentiment_map.get("positive", 0)
        neu = sentiment_map.get("neutral", 0)
        neg = sentiment_map.get("negative", 0)
        unk = sentiment_map.get("unknown", 0)
        total = sum(sentiment_map.values())

        pos_pct = round((pos / total) * 100, 2) if total > 0 else 0.0
        neu_pct = round((neu / total) * 100, 2) if total > 0 else 0.0
        neg_pct = round((neg / total) * 100, 2) if total > 0 else 0.0

        # Average confidence
        avg_conf_query = base_query.with_entities(func.avg(Feedback.confidence))
        avg_conf = avg_conf_query.scalar()
        avg_conf = round(float(avg_conf), 3) if avg_conf is not None else 0.0

        # Categories count
        cat_counts_query = base_query.with_entities(
            Feedback.category,
            func.count(Feedback.id)
        ).filter(
            Feedback.category.isnot(None)
        ).group_by(Feedback.category).all()
        category_counts = {row[0]: row[1] for row in cat_counts_query if row[0]}

        # Source count
        source_counts_query = base_query.with_entities(
            Feedback.source,
            func.count(Feedback.id)
        ).filter(
            Feedback.source.isnot(None)
        ).group_by(Feedback.source).all()
        source_counts = {row[0]: row[1] for row in source_counts_query if row[0]}

        return FeedbackStatsResponse(
            total_count=total,
            positive_count=pos,
            neutral_count=neu,
            negative_count=neg,
            unknown_count=unk,
            positive_pct=pos_pct,
            neutral_pct=neu_pct,
            negative_pct=neg_pct,
            average_confidence=avg_conf,
            category_counts=category_counts,
            source_counts=source_counts
        )
