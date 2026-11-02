"""
Business analytics and customer sentiment insights service.
Aggregates persisted feedback data into distributions, trends over time, and category breakdowns.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from db.models import Feedback
from schemas.analytics import (
    FeedbackAnalyticsResponse,
    SentimentTrendPoint,
    CategoryBreakdown,
    SourceBreakdown,
    RecentNegativeItem
)


class AnalyticsService:
    @staticmethod
    def get_analytics(
        db: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[int] = None
    ) -> FeedbackAnalyticsResponse:
        """
        Computes business intelligence metrics across persisted feedback records.
        """
        base_query = db.query(Feedback)
        if user_id is not None:
            base_query = base_query.filter(Feedback.user_id == user_id)
        if start_date:
            base_query = base_query.filter(Feedback.created_at >= start_date)
        if end_date:
            base_query = base_query.filter(Feedback.created_at <= end_date)

        total_feedback = base_query.count()

        # If zero records, return clean empty structure
        if total_feedback == 0:
            return FeedbackAnalyticsResponse(
                total_feedback=0,
                sentiment_counts={"positive": 0, "neutral": 0, "negative": 0, "unknown": 0},
                sentiment_percentages={"positive": 0.0, "neutral": 0.0, "negative": 0.0, "unknown": 0.0},
                average_confidence=0.0,
                sentiment_trends=[],
                category_breakdown=[],
                source_breakdown=[],
                recent_negative_feedback=[]
            )

        # 1. Overall sentiment distribution
        sentiment_counts_query = base_query.with_entities(
            Feedback.sentiment, func.count(Feedback.id)
        ).group_by(Feedback.sentiment).all()

        counts = {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0}
        for sentiment, count in sentiment_counts_query:
            if sentiment in counts:
                counts[sentiment] = count
            else:
                counts[sentiment] = count

        percentages = {
            s: round((cnt / total_feedback) * 100, 2)
            for s, cnt in counts.items()
        }

        # 2. Average confidence
        avg_conf = base_query.with_entities(func.avg(Feedback.confidence)).scalar() or 0.0
        avg_conf = round(float(avg_conf), 3)

        # 3. Sentiment trends over time (by date)
        # We group by date string YYYY-MM-DD
        records_for_trend = base_query.with_entities(
            func.date(Feedback.created_at).label("dt"),
            Feedback.sentiment,
            func.count(Feedback.id),
            func.avg(Feedback.confidence)
        ).group_by(
            func.date(Feedback.created_at),
            Feedback.sentiment
        ).order_by(func.date(Feedback.created_at)).all()

        daily_buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0, "total": 0, "conf_sum": 0.0, "conf_cnt": 0}
        )
        for dt, sentiment, cnt, conf_val in records_for_trend:
            date_str = str(dt) if dt else "Unknown"
            if sentiment in daily_buckets[date_str]:
                daily_buckets[date_str][sentiment] = cnt
            daily_buckets[date_str]["total"] += cnt
            if conf_val:
                daily_buckets[date_str]["conf_sum"] += (float(conf_val) * cnt)
                daily_buckets[date_str]["conf_cnt"] += cnt

        trends: List[SentimentTrendPoint] = []
        for date_str in sorted(daily_buckets.keys()):
            bucket = daily_buckets[date_str]
            avg_c = round(bucket["conf_sum"] / bucket["conf_cnt"], 3) if bucket["conf_cnt"] > 0 else 0.0
            trends.append(SentimentTrendPoint(
                date=date_str,
                positive=bucket["positive"],
                neutral=bucket["neutral"],
                negative=bucket["negative"],
                unknown=bucket["unknown"],
                total=bucket["total"],
                average_confidence=avg_c
            ))

        # 4. Category breakdown
        category_records = base_query.filter(
            Feedback.category.isnot(None)
        ).with_entities(
            Feedback.category,
            Feedback.sentiment,
            func.count(Feedback.id)
        ).group_by(Feedback.category, Feedback.sentiment).all()

        cat_buckets = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0, "unknown": 0, "total": 0})
        for cat, sentiment, cnt in category_records:
            cat_str = str(cat)
            if sentiment in cat_buckets[cat_str]:
                cat_buckets[cat_str][sentiment] = cnt
            cat_buckets[cat_str]["total"] += cnt

        category_breakdown: List[CategoryBreakdown] = []
        for cat_str, b in sorted(cat_buckets.items(), key=lambda x: x[1]["total"], reverse=True):
            tot = b["total"]
            category_breakdown.append(CategoryBreakdown(
                category=cat_str,
                positive=b["positive"],
                neutral=b["neutral"],
                negative=b["negative"],
                unknown=b["unknown"],
                total=tot,
                positive_pct=round((b["positive"] / tot) * 100, 2) if tot > 0 else 0.0,
                negative_pct=round((b["negative"] / tot) * 100, 2) if tot > 0 else 0.0
            ))

        # 5. Source breakdown
        source_records = base_query.filter(
            Feedback.source.isnot(None)
        ).with_entities(
            Feedback.source,
            Feedback.sentiment,
            func.count(Feedback.id)
        ).group_by(Feedback.source, Feedback.sentiment).all()

        src_buckets = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0, "total": 0})
        for src, sentiment, cnt in source_records:
            src_str = str(src)
            if sentiment in src_buckets[src_str]:
                src_buckets[src_str][sentiment] = cnt
            src_buckets[src_str]["total"] += cnt

        source_breakdown = [
            SourceBreakdown(
                source=src_name,
                total=b["total"],
                positive=b["positive"],
                neutral=b["neutral"],
                negative=b["negative"]
            )
            for src_name, b in sorted(src_buckets.items(), key=lambda x: x[1]["total"], reverse=True)
        ]

        # 6. Priority: Recent negative feedback items (first 10)
        recent_negatives_query = base_query.filter(
            Feedback.sentiment == "negative"
        ).order_by(desc(Feedback.created_at)).limit(10).all()

        recent_negatives = [
            RecentNegativeItem(
                id=neg.id,
                text=neg.text,
                confidence=neg.confidence,
                rating=neg.rating,
                category=neg.category,
                created_at=neg.created_at
            )
            for neg in recent_negatives_query
        ]

        return FeedbackAnalyticsResponse(
            total_feedback=total_feedback,
            sentiment_counts=counts,
            sentiment_percentages=percentages,
            average_confidence=avg_conf,
            sentiment_trends=trends,
            category_breakdown=category_breakdown,
            source_breakdown=source_breakdown,
            recent_negative_feedback=recent_negatives
        )
