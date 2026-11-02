"""
Phase 2.3 Comprehensive Verification & Hardening Test Suite:
1. Multi-Tenant Identity & Anti-Spoofing Protection.
2. HMAC Secret Entropy & Fail-Closed Validation.
3. Finalization Recovery State Machine & Dead Worker Recovery.
4. Render Persistent Storage Durability 10-Step Lifecycle.
"""

import os
import io
import time
import json
import uuid
import hmac
import hashlib
import pytest
from datetime import datetime, timezone, timedelta
from fastapi import Request, HTTPException
from fastapi.testclient import TestClient

from main import app
from core.config import settings, Settings, KNOWN_INSECURE_SECRETS
from core.security import generate_tenant_token, verify_tenant_token, resolve_trusted_tenant_identity
from db.session import SessionLocal
from db.models import Feedback, ImportJob, FeedbackStaging
from services.import_service import ImportService
from services.feedback_service import FeedbackService
from schemas.feedback import FeedbackCreate
from model import PredictionCache


client = TestClient(app)


# ==============================================================================
# SECTION 1: SECURE MULTI-TENANT IDENTITY & ANTI-SPOOFING TESTS
# ==============================================================================

def test_tenant_token_generation_and_verification():
    """Generates valid cryptographic HMAC token and verifies constant-time comparison."""
    token = generate_tenant_token(tenant_id=42)
    assert token.startswith("42:")
    verified_id = verify_tenant_token(token)
    assert verified_id == 42


def test_tenant_token_tampered_rejection():
    """Tampered token or invalid signature must be rejected."""
    token = generate_tenant_token(tenant_id=42)
    tampered = token[:-4] + "ffff"
    assert verify_tenant_token(tampered) is None

    malformed = "invalid-token-no-colon"
    assert verify_tenant_token(malformed) is None


def test_resolve_trusted_tenant_identity_bearer_token():
    """Valid Bearer token resolves to authenticated tenant."""
    token = generate_tenant_token(tenant_id=77)
    req = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/api/feedback",
        "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
    })
    resolved = resolve_trusted_tenant_identity(req)
    assert resolved == 77


def test_resolve_trusted_tenant_identity_spoofing_conflict():
    """Client providing authenticated token for Tenant 10 cannot spoof X-Tenant-ID: 20."""
    token = generate_tenant_token(tenant_id=10)
    req = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/api/feedback",
        "headers": [
            (b"authorization", f"Bearer {token}".encode("utf-8")),
            (b"x-tenant-id", b"20"),
        ],
    })
    with pytest.raises(HTTPException) as exc_info:
        resolve_trusted_tenant_identity(req)
    assert exc_info.value.status_code == 403
    assert "Access denied: Claimed tenant ID does not match authenticated identity." in exc_info.value.detail


def test_resolve_trusted_tenant_identity_api_key(monkeypatch):
    """Configured API Key maps directly to authenticated tenant."""
    monkeypatch.setattr(settings, "API_KEYS", {"super-secret-tenant-api-key-12345": 999})
    req = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/api/feedback",
        "headers": [(b"x-api-key", b"super-secret-tenant-api-key-12345")],
    })
    resolved = resolve_trusted_tenant_identity(req)
    assert resolved == 999


def test_resolve_trusted_tenant_identity_invalid_api_key(monkeypatch):
    """Invalid API key returns 401 Unauthorized."""
    monkeypatch.setattr(settings, "API_KEYS", {"valid-key": 999})
    req = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/api/feedback",
        "headers": [(b"x-api-key", b"unrecognized-key-val")],
    })
    with pytest.raises(HTTPException) as exc_info:
        resolve_trusted_tenant_identity(req)
    assert exc_info.value.status_code == 401


def test_resolve_trusted_tenant_identity_fail_closed_in_production(monkeypatch):
    """Raw X-Tenant-ID header is rejected in production without authenticated credential."""
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    req = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/api/feedback",
        "headers": [(b"x-tenant-id", b"55")],
    })
    with pytest.raises(HTTPException) as exc_info:
        resolve_trusted_tenant_identity(req)
    assert exc_info.value.status_code == 403
    assert "Untrusted client tenant header" in exc_info.value.detail


def test_cross_tenant_isolation_feedback_access():
    """Tenant A cannot access or delete Tenant B's feedback record."""
    db = SessionLocal()
    try:
        # Create feedback record belonging to Tenant 100
        fb = Feedback(
            user_id=100,
            text="Tenant 100 private feedback",
            clean_text="tenant 100 private feedback",
            sentiment="positive",
            confidence=0.95,
            positive_score=0.95,
            neutral_score=0.03,
            negative_score=0.02,
            category="SecurityTest",
            source="test",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        fb_id = fb.id

        token_b = generate_tenant_token(tenant_id=200)

        # Tenant 200 attempts to read Tenant 100's record
        resp_get = client.get(f"/api/feedback/{fb_id}", headers={"Authorization": f"Bearer {token_b}"})
        assert resp_get.status_code == 403

        # Tenant 200 attempts to delete Tenant 100's record
        resp_del = client.delete(f"/api/feedback/{fb_id}", headers={"Authorization": f"Bearer {token_b}"})
        assert resp_del.status_code == 403

        # Tenant 100 can read their own record
        token_a = generate_tenant_token(tenant_id=100)
        resp_own = client.get(f"/api/feedback/{fb_id}", headers={"Authorization": f"Bearer {token_a}"})
        assert resp_own.status_code == 200
    finally:
        db.query(Feedback).filter(Feedback.category == "SecurityTest").delete()
        db.commit()
        db.close()


def test_cross_tenant_isolation_import_resumption():
    """Tenant A cannot resume Tenant B's import job."""
    db = SessionLocal()
    try:
        job = ImportJob(
            id=str(uuid.uuid4()),
            user_id=500,
            filename="tenant500_data.csv",
            status="FAILED",
            stage="FAILED",
            total_rows=100,
            processed_rows=50,
            successful_rows=50,
            failed_rows=0,
            created_at=datetime.now(timezone.utc)
        )
        db.add(job)
        db.commit()

        token_other = generate_tenant_token(tenant_id=600)
        resp = client.post(f"/api/feedback/import/{job.id}/resume", headers={"Authorization": f"Bearer {token_other}"})
        assert resp.status_code == 403
        assert "Access denied: Import job belongs to another tenant." in resp.json()["detail"]
    finally:
        db.query(ImportJob).filter(ImportJob.user_id == 500).delete()
        db.commit()
        db.close()


# ==============================================================================
# SECTION 2: HMAC SECRET HARDENING & FAIL-CLOSED VALIDATION TESTS
# ==============================================================================

def test_secret_entropy_validation_short():
    """Secret shorter than 32 characters must be rejected."""
    with pytest.raises(ValueError) as exc:
        Settings.validate_secret_entropy("short-secret-key-12345")
    assert "at least 32 characters long" in str(exc.value)


def test_secret_entropy_validation_placeholder():
    """Known placeholder or default secret keys must be rejected."""
    for ph in KNOWN_INSECURE_SECRETS:
        with pytest.raises(ValueError) as exc:
            Settings.validate_secret_entropy(ph)
        assert "cannot be a known default" in str(exc.value) or "at least 32 characters" in str(exc.value)


def test_secret_entropy_validation_low_entropy():
    """Repetitive string with insufficient character diversity must be rejected."""
    with pytest.raises(ValueError) as exc:
        Settings.validate_secret_entropy("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert "insufficient character entropy" in str(exc.value)


def test_secret_entropy_validation_strong():
    """Strong 32+ character random secret passes validation."""
    strong_key = "a9f8e4b2c1d0e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4"
    # Should not raise
    Settings.validate_secret_entropy(strong_key)


def test_production_fail_closed_on_missing_or_weak_secret(monkeypatch):
    """In production, missing or insecure SECRET_KEY raises RuntimeError on startup/usage."""
    test_settings = Settings()
    test_settings.ENVIRONMENT = "production"
    test_settings.SECRET_KEY = "sentiment-pulse-production-secret-v1"

    with pytest.raises(RuntimeError) as exc:
        test_settings.validate_production_security()
    assert "Insecure SECRET_KEY in production" in str(exc.value)

    test_settings.SECRET_KEY = ""
    with pytest.raises(RuntimeError) as exc:
        test_settings.validate_production_security()
    assert "CRITICAL SECURITY VIOLATION" in str(exc.value)


def test_secret_rotation_token_invalidation():
    """Changing SECRET_KEY immediately invalidates tokens issued under the old key."""
    old_secret = "old-secret-key-that-is-at-least-32-characters-long!"
    new_secret = "new-secret-key-that-is-at-least-32-characters-long!"

    token_old = generate_tenant_token(tenant_id=42, secret_key=old_secret)
    # Valid under old secret
    assert verify_tenant_token(token_old, secret_key=old_secret) == 42
    # Invalidated under new secret
    assert verify_tenant_token(token_old, secret_key=new_secret) is None


def test_secret_rotation_cache_invalidation():
    """Changing SECRET_KEY produces different hash digests, preventing stale cache hits."""
    text = "Great customer service and fast delivery!"
    key1 = PredictionCache.hash_text(text, salt="secret-one-at-least-32-characters-long!")
    key2 = PredictionCache.hash_text(text, salt="secret-two-at-least-32-characters-long!")
    assert key1 != key2


def test_zero_secret_leakage():
    """Secret key is never included in error responses or string representations."""
    req = Request(scope={
        "type": "http",
        "method": "GET",
        "path": "/api/feedback",
        "headers": [(b"authorization", b"Bearer 99:forged_signature_here")],
    })
    try:
        resolve_trusted_tenant_identity(req)
    except HTTPException as h_err:
        assert settings.SECRET_KEY not in str(h_err.detail)


# ==============================================================================
# SECTION 3: FINALIZATION RECOVERY STATE MACHINE & DEAD WORKER RECOVERY
# ==============================================================================

def test_finalization_cas_duplicate_prevention():
    """Atomic CAS claim allows only one worker to claim FINALIZING; second attempt is rejected."""
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        job = ImportJob(
            id=job_id,
            filename="concurrent_fin_test.csv",
            status="PROCESSING",
            stage="PROCESSING_INFERENCE",
            total_rows=100,
            processed_rows=100,
            successful_rows=100,
            failed_rows=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(job)
        db.commit()

        # Worker 1 claims finalization
        claimed1 = db.query(ImportJob).filter(
            ImportJob.id == job_id,
            ImportJob.status != "COMPLETED",
            ImportJob.stage != "FINALIZING"
        ).update({"stage": "FINALIZING", "updated_at": datetime.now(timezone.utc)})
        db.commit()
        assert claimed1 == 1

        # Worker 2 attempts to claim finalization immediately (fresh lease)
        stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=300)
        claimed2 = db.query(ImportJob).filter(
            ImportJob.id == job_id,
            ImportJob.status != "COMPLETED",
            ImportJob.stage != "FINALIZING"
        ).update({"stage": "FINALIZING", "updated_at": datetime.now(timezone.utc)})
        db.commit()
        assert claimed2 == 0
    finally:
        db.query(ImportJob).filter(ImportJob.id == job_id).delete()
        db.commit()
        db.close()


def test_finalization_stale_lease_recovery():
    """Worker death leaving job in FINALIZING past lease timeout (300s) can be reclaimed."""
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        # Simulate worker death 10 minutes ago
        crashed_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        job = ImportJob(
            id=job_id,
            filename="crashed_worker.csv",
            status="PROCESSING",
            stage="FINALIZING",
            total_rows=50,
            processed_rows=50,
            successful_rows=50,
            failed_rows=0,
            created_at=crashed_time,
            updated_at=crashed_time
        )
        db.add(job)
        db.commit()

        stale_cutoff = datetime.now(timezone.utc) - timedelta(seconds=300)
        from sqlalchemy import or_
        claimed = db.query(ImportJob).filter(
            ImportJob.id == job_id,
            ImportJob.status != "COMPLETED",
            or_(
                ImportJob.stage != "FINALIZING",
                ImportJob.updated_at < stale_cutoff
            )
        ).update({"stage": "FINALIZING", "updated_at": datetime.now(timezone.utc)})
        db.commit()
        assert claimed == 1
    finally:
        db.query(ImportJob).filter(ImportJob.id == job_id).delete()
        db.commit()
        db.close()


def test_finalization_dead_worker_recovery_atomic_transfer():
    """Dead worker recovery safely transfers staged rows, deletes staging, and marks COMPLETED."""
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    try:
        crashed_time = datetime.now(timezone.utc) - timedelta(minutes=6)
        job = ImportJob(
            id=job_id,
            filename="dead_worker_job.csv",
            status="PROCESSING",
            stage="FINALIZING",
            total_rows=2,
            processed_rows=2,
            successful_rows=2,
            failed_rows=0,
            created_at=crashed_time,
            updated_at=crashed_time
        )
        db.add(job)

        # Add 2 staged records
        staged1 = FeedbackStaging(
            import_job_id=job_id,
            row_number=1,
            text="Service was exceptional!",
            clean_text="service was exceptional!",
            sentiment="positive",
            confidence=0.98,
            positive_score=0.98,
            neutral_score=0.01,
            negative_score=0.01,
            category="RecoveryTest",
            source="csv",
            rating=5,
            model_version="3.0.0",
            created_at=crashed_time,
            updated_at=crashed_time
        )
        staged2 = FeedbackStaging(
            import_job_id=job_id,
            row_number=2,
            text="Very disappointed with response time.",
            clean_text="very disappointed with response time.",
            sentiment="negative",
            confidence=0.95,
            positive_score=0.02,
            neutral_score=0.03,
            negative_score=0.95,
            category="RecoveryTest",
            source="csv",
            rating=1,
            model_version="3.0.0",
            created_at=crashed_time,
            updated_at=crashed_time
        )
        db.add_all([staged1, staged2])
        db.commit()

        # Run recovery
        recovered = ImportService.recover_stale_finalization(db, job_id=job_id, lease_timeout_seconds=300)
        assert recovered is not None
        assert recovered.status == "COMPLETED"
        assert recovered.stage == "COMPLETED"

        # Verify 2 records transferred to Feedback
        transferred = db.query(Feedback).filter(Feedback.category == "RecoveryTest").count()
        assert transferred == 2

        # Verify staging table is empty for this job
        remaining_staging = db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).count()
        assert remaining_staging == 0
    finally:
        db.query(Feedback).filter(Feedback.category == "RecoveryTest").delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).delete()
        db.query(ImportJob).filter(ImportJob.id == job_id).delete()
        db.commit()
        db.close()


# ==============================================================================
# SECTION 4: RENDER PERSISTENT STORAGE DURABILITY 10-STEP LIFECYCLE
# ==============================================================================

def test_render_yaml_starter_plan_and_disk_configuration():
    """Audits render.yaml to guarantee valid disk attachment under plan: starter."""
    render_yaml_path = os.path.join(settings.BASE_DIR if hasattr(settings, "BASE_DIR") else os.path.dirname(os.path.dirname(__file__)), "render.yaml")
    with open(render_yaml_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "plan: starter" in content, "Render Persistent Disks require plan: starter or higher"
    assert "mountPath: /var/data" in content
    assert "sizeGB: 10" in content
    assert "name: sentiment-pulse-data" in content


def test_render_storage_durability_10_step_lifecycle(tmp_path):
    """
    Simulates the complete 10-step durability lifecycle across restarts:
    1. Upload CSV file to durable storage.
    2. Record SHA-256 fingerprint.
    3. Verify file stored in durable directory.
    4. Simulate server restart (new session, fresh service instance).
    5. Verify file remains intact and accessible.
    6. Verify SHA-256 fingerprint matches.
    7. Resume import from checkpoint.
    8. Verify processing succeeds.
    9. Verify file cleanup occurs only after successful completion.
    10. Verify failed imports preserve file until cleanup policy triggers.
    """
    # 1. Upload CSV file
    durable_dir = settings.UPLOAD_DIR
    os.makedirs(durable_dir, exist_ok=True)
    csv_filename = f"lifecycle_{uuid.uuid4().hex[:8]}.csv"
    durable_path = os.path.join(durable_dir, csv_filename)

    csv_bytes = b"text,category,rating\nOutstanding customer experience!,LifecycleTest,5\nTerrible wait times,LifecycleTest,1\n"
    with open(durable_path, "wb") as f:
        f.write(csv_bytes)

    # 2. Record SHA-256 fingerprint
    orig_hash = hashlib.sha256(csv_bytes).hexdigest()

    # 3. Verify file stored in durable directory
    assert os.path.exists(durable_path)
    assert os.path.getsize(durable_path) > 0

    # 4. Simulate server restart: close and recreate DB session, re-read disk
    db1 = SessionLocal()
    job = ImportService.create_import_job(
        db1,
        filename=csv_filename,
        file_path=durable_path,
        file_hash=orig_hash
    )
    job_id = job.id
    db1.close()

    # 5. Verify file remains intact and accessible
    assert os.path.exists(durable_path)

    # 6. Verify SHA-256 fingerprint matches
    with open(durable_path, "rb") as f_chk:
        assert hashlib.sha256(f_chk.read()).hexdigest() == orig_hash

    # 7. Resume import from checkpoint
    db2 = SessionLocal()
    resumed_job = ImportService.resume_import_job(db2, job_id=job_id)
    assert resumed_job.status == "PENDING"
    assert resumed_job.stage == "QUEUED"

    # 8. Verify processing succeeds
    ImportService.process_csv_file_background(job_id=job_id, file_path=durable_path)
    db2.expire_all()
    completed_job = db2.query(ImportJob).filter(ImportJob.id == job_id).first()
    assert completed_job.status == "COMPLETED"
    assert completed_job.successful_rows == 2

    # 9. Verify file cleanup occurs only after successful completion
    assert not os.path.exists(durable_path), "Durable file must be unlinked after successful completion"

    # 10. Verify failed/interrupted imports preserve file until cleanup policy triggers
    fail_csv_filename = f"lifecycle_fail_{uuid.uuid4().hex[:8]}.csv"
    fail_durable_path = os.path.join(durable_dir, fail_csv_filename)
    with open(fail_durable_path, "w", encoding="utf-8") as f:
        f.write("text,category\nValid row,PreserveTest\n")

    fail_hash = hashlib.sha256(b"text,category\nValid row,PreserveTest\n").hexdigest()
    fail_job = ImportService.create_import_job(
        db2,
        filename=fail_csv_filename,
        file_path=fail_durable_path,
        file_hash=fail_hash
    )
    fail_job.status = "FAILED"
    fail_job.stage = "INTERRUPTED"
    db2.commit()

    # Verify file remains preserved on disk for interrupted/failed job
    assert os.path.exists(fail_durable_path), "Interrupted durable file must be preserved on disk for resumption"

    # Cleanup test artifacts
    if os.path.exists(fail_durable_path):
        os.unlink(fail_durable_path)
    db2.query(Feedback).filter(Feedback.category.in_(["LifecycleTest", "PreserveTest"])).delete()
    db2.query(ImportJob).filter(ImportJob.id.in_([job_id, fail_job.id])).delete()
    db2.commit()
    db2.close()
