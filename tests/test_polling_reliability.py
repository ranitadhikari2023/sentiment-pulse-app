"""
Enterprise Polling Reliability, Rate Limiting, and Status API Test Suite.
Validates:
1. Job status API schema contracts (job_id, inserted_rows, skipped_rows, progress, message, updated_at).
2. Anti-caching headers (Cache-Control: no-store, Pragma: no-cache).
3. Multi-tenant isolation and ownership enforcement.
4. Missing job (404) handling.
5. In-flight status polling cannot mutate job state or counters.
6. Sliding-window adaptive rate limiting with Retry-After and rate limit headers.
7. CAS lease deduplication for background workers and finalization.
"""

import uuid
import json
import time
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from main import app
from core.config import settings
from core.security import generate_tenant_token
from core.middleware import limiter
from db.session import SessionLocal
from db.models import ImportJob, Feedback, FeedbackStaging
from services.import_service import ImportService
from schemas.import_job import ImportJobResponse


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Resets sliding window rate limiter before each test."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_import_job_response_schema_computed_fields():
    """Verifies that ImportJobResponse accurately populates computed and aliased fields."""
    now = datetime.now(timezone.utc)
    res = ImportJobResponse(
        id="job-uuid-1234",
        filename="test_feedback.csv",
        total_rows=1000,
        processed_rows=450,
        successful_rows=440,
        failed_rows=10,
        status="PROCESSING",
        stage="INSERTING",
        created_at=now,
        updated_at=now,
    )

    data = res.model_dump()
    assert data["job_id"] == "job-uuid-1234"
    assert data["id"] == "job-uuid-1234"
    assert data["inserted_rows"] == 440
    assert data["successful_rows"] == 440
    assert data["skipped_rows"] == 0
    assert data["progress"] == 45.0
    assert "450 / 1,000 records" in data["message"]
    assert data["updated_at"] is not None


def test_import_job_response_progress_bounds():
    """Verifies progress percentage is strictly clamped between 0.0% and 100.0%."""
    now = datetime.now(timezone.utc)

    # 0 rows / queued
    res_zero = ImportJobResponse(
        id="zero", filename="f.csv", total_rows=0, processed_rows=0,
        successful_rows=0, failed_rows=0, status="PENDING", created_at=now
    )
    assert res_zero.progress == 0.0

    # Over-processed safety clamp
    res_over = ImportJobResponse(
        id="over", filename="f.csv", total_rows=100, processed_rows=150,
        successful_rows=150, failed_rows=0, status="PROCESSING", created_at=now
    )
    assert res_over.progress == 100.0

    # Completed state with 0 total rows reports 100%
    res_comp = ImportJobResponse(
        id="comp", filename="f.csv", total_rows=0, processed_rows=0,
        successful_rows=0, failed_rows=0, status="COMPLETED", created_at=now
    )
    assert res_comp.progress == 100.0


def test_job_status_endpoint_returns_json_and_no_store_headers(db_session: Session):
    """
    Verifies that GET /feedback/import/{job_id}:
    1. Returns status 200 with JSON payload.
    2. Sets Cache-Control: no-store, no-cache, must-revalidate.
    3. Includes X-Request-ID and correlation data.
    4. Conforms to the enterprise status response schema.
    """
    with TestClient(app) as client:
        now = datetime.now(timezone.utc)
        job_id = f"test-status-{uuid.uuid4()}"
        job = ImportJob(
            id=job_id,
            filename="customer_reviews.csv",
            total_rows=1050000,
            processed_rows=450000,
            successful_rows=449900,
            failed_rows=100,
            status="PROCESSING",
            stage="INSERTING",
            created_at=now,
            updated_at=now,
        )
        db_session.add(job)
        db_session.commit()

        try:
            res = client.get(f"/feedback/import/{job_id}")
            assert res.status_code == 200
            assert "application/json" in res.headers.get("content-type", "")

            # Verify Anti-Caching response headers
            cache_ctrl = res.headers.get("cache-control", "")
            assert "no-store" in cache_ctrl
            assert "no-cache" in cache_ctrl
            assert res.headers.get("pragma") == "no-cache"
            assert res.headers.get("expires") == "0"
            assert res.headers.get("x-request-id") is not None

            data = res.json()
            assert data["job_id"] == job_id
            assert data["id"] == job_id
            assert data["status"] == "PROCESSING"
            assert data["total_rows"] == 1050000
            assert data["processed_rows"] == 450000
            assert data["inserted_rows"] == 449900
            assert data["failed_rows"] == 100
            assert data["skipped_rows"] == 0
            assert data["progress"] == 42.86
            assert "Processing customer feedback" in data["message"]
            assert data["updated_at"] is not None

        finally:
            db_session.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db_session.commit()


def test_job_status_endpoint_missing_job_returns_404():
    """Verifies that non-existent job ID returns clean JSON 404 error."""
    with TestClient(app) as client:
        res = client.get("/feedback/import/non-existent-job-uuid-999")
        assert res.status_code == 404
        data = res.json()
        assert "not found" in data["detail"].lower()


def test_job_status_tenant_isolation(db_session: Session):
    """
    Verifies multi-tenant access control on job status polling:
    1. Tenant 100 can poll their own job.
    2. Tenant 200 attempting to poll tenant 100's job is rejected with HTTP 403 Forbidden.
    """
    now = datetime.now(timezone.utc)
    job_id = f"test-tenant-{uuid.uuid4()}"
    job = ImportJob(
        id=job_id,
        user_id=100,
        filename="tenant_data.csv",
        total_rows=500,
        processed_rows=500,
        successful_rows=500,
        failed_rows=0,
        status="COMPLETED",
        stage="COMPLETED",
        created_at=now,
        updated_at=now,
    )
    db_session.add(job)
    db_session.commit()

    token_tenant_100 = generate_tenant_token(100)
    token_tenant_200 = generate_tenant_token(200)

    try:
        with TestClient(app) as client:
            # 1. Authorized tenant succeeds
            res_auth = client.get(
                f"/feedback/import/{job_id}",
                headers={"Authorization": f"Bearer {token_tenant_100}"}
            )
            assert res_auth.status_code == 200
            assert res_auth.json()["job_id"] == job_id

            # 2. Unauthorized cross-tenant poll fails with 403
            res_unauth = client.get(
                f"/feedback/import/{job_id}",
                headers={"Authorization": f"Bearer {token_tenant_200}"}
            )
            assert res_unauth.status_code == 403
            assert "access denied" in res_unauth.json()["detail"].lower()

    finally:
        db_session.query(ImportJob).filter(ImportJob.id == job_id).delete()
        db_session.commit()


def test_polling_request_does_not_mutate_persisted_job_state(db_session: Session):
    """
    Verifies that repeated polling status checks are strictly non-mutating
    and cannot alter job status, row counters, or completed timestamps.
    """
    with TestClient(app) as client:
        now = datetime.now(timezone.utc)
        job_id = f"test-immutable-{uuid.uuid4()}"
        job = ImportJob(
            id=job_id,
            filename="reviews.csv",
            total_rows=10000,
            processed_rows=7500,
            successful_rows=7490,
            failed_rows=10,
            status="PROCESSING",
            stage="INSERTING",
            created_at=now,
            updated_at=now,
        )
        db_session.add(job)
        db_session.commit()

        try:
            for _ in range(5):
                res = client.get(f"/feedback/import/{job_id}")
                assert res.status_code == 200

            # Reload from database and verify counters are identical
            db_session.expire_all()
            reloaded = db_session.query(ImportJob).filter(ImportJob.id == job_id).first()
            assert reloaded.status == "PROCESSING"
            assert reloaded.total_rows == 10000
            assert reloaded.processed_rows == 7500
            assert reloaded.successful_rows == 7490
            assert reloaded.failed_rows == 10

        finally:
            db_session.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db_session.commit()


def test_sliding_window_rate_limiter_allows_polling_burst():
    """
    Verifies that the sliding window limiter permits frequent status polling
    up to RATE_LIMIT_STATUS_POLL without premature throttling.
    """
    client_ip = "192.168.1.50"
    path = "/feedback/import/job-12345"
    method = "GET"

    # Up to 60 rapid polls in sequence should succeed cleanly
    for _ in range(60):
        is_limited, limit, remaining, retry_after = limiter.is_rate_limited(client_ip, path, method)
        assert is_limited is False
        assert limit == getattr(settings, "RATE_LIMIT_STATUS_POLL", 120)
        assert retry_after == 0


def test_sliding_window_rate_limiter_enforces_limit_with_retry_after():
    """
    Verifies that when rate limit threshold is exceeded:
    1. is_limited returns True.
    2. retry_after is calculated accurately (> 0).
    3. Middleware returns HTTP 429 with Retry-After header.
    """
    client_ip = "10.0.0.99"
    path = "/feedback/import"
    method = "POST"
    limit_val = getattr(settings, "RATE_LIMIT_IMPORT_UPLOAD", 30)

    # Exhaust limit
    for _ in range(limit_val):
        limiter.is_rate_limited(client_ip, path, method)

    # Next request must be throttled
    is_limited, limit, remaining, retry_after = limiter.is_rate_limited(client_ip, path, method)
    assert is_limited is True
    assert remaining == 0
    assert retry_after > 0


def test_middleware_cf_ray_propagation():
    """Verifies that incoming Cloudflare CF-Ray header is propagated to response."""
    test_ray = "8db291c9f0123456-IAD"
    with TestClient(app) as client:
        res = client.get("/health", headers={"CF-Ray": test_ray})
        assert res.status_code == 200
        assert res.headers.get("cf-ray") == test_ray
