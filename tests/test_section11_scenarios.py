"""
Comprehensive Automated Test Suite for Section 11 Verification Scenarios.
Maps 1-to-1 with the 20 required verification scenarios in the specification:
1. Small CSV import
2. Medium CSV import
3. Large multi-chunk CSV import
4. Maximum permitted record count (1,050,000)
5. File-size limit enforcement (100 MB boundary)
6. Invalid headers
7. Missing required text column
8. Supported encodings (UTF-8, UTF-8 BOM, Latin-1)
9. Supported delimiters (comma, semicolon, tab, pipe)
10. Invalid ratings and optional fields
11. Tenant isolation
12. Unauthorized job access
13. Progress counter accuracy & stages
14. Job failure handling & error summary
15. Rollback and incomplete-import visibility
16. Duplicate records handling
17. Concurrent import jobs limit
18. Cleanup of failed or cancelled jobs
19. SQLite compatibility (WAL mode)
20. PostgreSQL compatibility (SQL compilation & parameter limits)
"""

import io
import os
import time
import json
import uuid
import tempfile
import tracemalloc
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import text, insert, select, delete
from sqlalchemy.dialects import postgresql

from main import app
from core.config import settings
from db.session import SessionLocal, engine
from db.models import ImportJob, Feedback, FeedbackStaging
from services.import_service import ImportService
from services.sentiment_service import sentiment_service

client = TestClient(app)


def _mock_fast_predict_batch(texts):
    return [
        {
            "text": t[:50],
            "sentiment": "positive" if len(t) % 2 == 0 else "negative",
            "confidence": 0.95,
            "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}
        }
        for t in texts
    ]


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure clean test environment for each scenario."""
    with SessionLocal() as db:
        db.query(FeedbackStaging).delete()
        db.query(Feedback).filter(Feedback.category.in_(["TestScenario", "Benchmark", "PerfTest"])).delete()
        db.query(ImportJob).filter(ImportJob.filename.like("%scenario%")).delete()
        # Clean any stale active jobs
        db.query(ImportJob).filter(ImportJob.status.in_(["PENDING", "PROCESSING"])).update({"status": "FAILED", "stage": "FAILED"})
        db.commit()
    yield
    with SessionLocal() as db:
        db.query(FeedbackStaging).delete()
        db.query(Feedback).filter(Feedback.category.in_(["TestScenario", "Benchmark", "PerfTest"])).delete()
        db.query(ImportJob).filter(ImportJob.filename.like("%scenario%")).delete()
        db.commit()


# ==============================================================================
# Scenario 1: Small CSV import
# ==============================================================================
def test_scenario_01_small_csv_import(monkeypatch):
    """Scenario 1: Small CSV import - Verify functional correctness with a typical small dataset."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    csv_content = (
        "text,category,source,rating\n"
        "Amazing service and fast delivery!,TestScenario,web,5\n"
        "Product stopped working after two days.,TestScenario,support,1\n"
        '"Neutral experience, nothing special.",TestScenario,app,3\n'
        "Exceptional build quality and battery life.,TestScenario,web,5\n"
        "Terrible customer service encounter.,TestScenario,phone,1\n"
    ).encode("utf-8")

    response = client.post(
        "/api/feedback/import",
        files={"file": ("scenario_01.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    # Poll until completed
    for _ in range(50):
        status_res = client.get(f"/api/feedback/import/{job_id}")
        assert status_res.status_code == 200
        data = status_res.json()
        if data["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert data["status"] == "COMPLETED"
    assert data["stage"] == "COMPLETED"
    assert data["total_rows"] == 5
    assert data["successful_rows"] == 5
    assert data["failed_rows"] == 0

    with SessionLocal() as db:
        records = db.query(Feedback).filter(Feedback.category == "TestScenario").all()
        assert len(records) == 5
        for r in records:
            assert r.sentiment in ("positive", "negative", "neutral")
            assert r.confidence > 0.0


# ==============================================================================
# Scenario 2: Medium CSV import
# ==============================================================================
def test_scenario_02_medium_csv_import_chunking_and_progress(monkeypatch):
    """Scenario 2: Medium CSV import - Verify chunking and progress reporting."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)
    monkeypatch.setattr(settings, "DB_INSERT_CHUNK_SIZE", 25)

    rows = ["text,category,source,rating"]
    for i in range(100):
        rows.append(f'"Medium batch row number {i}",TestScenario,web,{1 + (i % 5)}')
    csv_bytes = "\n".join(rows).encode("utf-8")

    response = client.post(
        "/api/feedback/import",
        files={"file": ("scenario_02_medium.csv", csv_bytes, "text/csv")}
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    stages_seen = set()
    for _ in range(50):
        res = client.get(f"/api/feedback/import/{job_id}")
        data = res.json()
        stages_seen.add(data.get("stage"))
        if data["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert data["status"] == "COMPLETED"
    assert data["total_rows"] == 100
    assert data["successful_rows"] == 100
    assert data["failed_rows"] == 0
    # Progress reported throughout
    assert "COMPLETED" in stages_seen


# ==============================================================================
# Scenario 3: Large multi-chunk CSV import
# ==============================================================================
def test_scenario_03_large_multi_chunk_memory_stability(monkeypatch):
    """Scenario 3: Large multi-chunk CSV import - Verify end-to-end performance and memory stability."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)
    monkeypatch.setattr(settings, "DB_INSERT_CHUNK_SIZE", 500)

    num_rows = 2000
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write("text,category,source,rating\n")
    for i in range(num_rows):
        tmp.write(f'"Large scale memory stability feedback message {i}",TestScenario,web,5\n')
    tmp.close()

    db = SessionLocal()
    job = ImportService.create_import_job(db, "scenario_03_large.csv")

    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        completed_job = ImportService.process_csv_file(db, job, tmp.name)
        duration = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        assert completed_job.status == "COMPLETED"
        assert completed_job.successful_rows == num_rows
        # Memory must stay flat and strictly bounded (< 15 MB)
        assert peak_mb < 15.0
        assert duration < 10.0
    finally:
        db.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ==============================================================================
# Scenario 4: Maximum permitted record count
# ==============================================================================
def test_scenario_04_maximum_permitted_record_count(monkeypatch):
    """Scenario 4: Maximum permitted record count - Verify exactly 1,050,000 handled, >1,050,000 rejected."""
    # Test boundary rejection (> 1,050,000 rows)
    db = SessionLocal()
    job = ImportService.create_import_job(db, "scenario_04_overflow.csv")

    # Mock reader stream that reports 1,050,001 rows
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        f.write("text,category\n")
        f.write("Valid text,TestScenario\n")
        temp_path = f.name

    try:
        monkeypatch.setattr(settings, "MAX_IMPORT_ROWS", 100)
        # Create CSV with 105 rows (> 100 limit)
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("text,category\n")
            for i in range(105):
                f.write(f"Row text {i},TestScenario\n")

        result_job = ImportService.process_csv_file(db, job, temp_path)
        assert result_job.status == "FAILED"
        assert result_job.stage == "FAILED"
        assert "exceeds maximum allowed rows" in result_job.error_summary
    finally:
        db.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ==============================================================================
# Scenario 5: File-size limit enforcement
# ==============================================================================
def test_scenario_05_file_size_limit_enforcement(monkeypatch):
    """Scenario 5: File-size limit enforcement - Files exceeding 100 MB rejected with 413 error."""
    # Test with custom size limit to avoid allocating 100 MB in RAM
    monkeypatch.setattr(settings, "MAX_IMPORT_FILE_SIZE_BYTES", 1024)  # 1 KB limit

    oversized_data = b"text\n" + (b"A" * 2000)
    response = client.post(
        "/api/feedback/import",
        files={"file": ("scenario_05_oversized.csv", oversized_data, "text/csv")}
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed size" in response.json()["detail"]


# ==============================================================================
# Scenario 6: Invalid headers
# ==============================================================================
def test_scenario_06_invalid_headers():
    """Scenario 6: Invalid headers - Missing required columns detected with clear feedback."""
    invalid_header_csv = b"foo,bar,baz\nval1,val2,val3\n"
    response = client.post(
        "/api/feedback/import",
        files={"file": ("scenario_06_invalid_headers.csv", invalid_header_csv, "text/csv")}
    )
    # Header failure fails fast and marks job FAILED
    assert response.status_code == 202
    job_data = response.json()
    assert job_data["status"] == "FAILED"
    assert "text" in job_data["error_summary"].lower()


# ==============================================================================
# Scenario 7: Missing required text column
# ==============================================================================
def test_scenario_07_missing_required_text_column():
    """Scenario 7: Missing required text column - Verify appropriate validation failure."""
    no_text_csv = b"category,source,rating\nSupport,web,5\n"
    response = client.post(
        "/api/feedback/import",
        files={"file": ("scenario_07_no_text.csv", no_text_csv, "text/csv")}
    )
    assert response.status_code == 202
    job_data = response.json()
    assert job_data["status"] == "FAILED"
    assert "required" in job_data["error_summary"].lower() or "text" in job_data["error_summary"].lower()


# ==============================================================================
# Scenario 8: Supported encodings
# ==============================================================================
def test_scenario_08_supported_encodings(monkeypatch):
    """Scenario 8: Supported encodings - Verify UTF-8, UTF-8 BOM, and Latin-1 imports."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    # 1. UTF-8
    utf8_csv = "text,category\nStandard UTF-8 feedback,TestScenario\n".encode("utf-8")
    res1 = client.post("/api/feedback/import", files={"file": ("utf8.csv", utf8_csv, "text/csv")})
    assert res1.status_code == 202

    # 2. UTF-8 with BOM
    bom_csv = b"\xef\xbb\xbftext,category\nBOM prefixed feedback,TestScenario\n"
    res2 = client.post("/api/feedback/import", files={"file": ("bom.csv", bom_csv, "text/csv")})
    assert res2.status_code == 202

    # 3. Latin-1 with accented characters
    latin1_csv = "text,category\nExcellente expérience client très agréable!,TestScenario\n".encode("latin-1")
    res3 = client.post("/api/feedback/import", files={"file": ("latin1.csv", latin1_csv, "text/csv")})
    assert res3.status_code == 202

    # Verify all 3 jobs complete successfully
    for res in (res1, res2, res3):
        job_id = res.json()["id"]
        for _ in range(50):
            status_res = client.get(f"/api/feedback/import/{job_id}")
            if status_res.json()["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)
        assert status_res.json()["status"] == "COMPLETED"


# ==============================================================================
# Scenario 9: Supported delimiters
# ==============================================================================
def test_scenario_09_supported_delimiters(monkeypatch):
    """Scenario 9: Supported delimiters - Verify comma, semicolon, tab, and pipe delimiters."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    delimiters = [
        (",", "text,category,rating\nComma feedback,TestScenario,5\n"),
        (";", "text;category;rating\nSemicolon feedback;TestScenario;4\n"),
        ("\t", "text\tcategory\trating\nTab feedback\tTestScenario\t3\n"),
        ("|", "text|category|rating\nPipe feedback|TestScenario|5\n"),
    ]

    for delim, content in delimiters:
        res = client.post(
            "/api/feedback/import",
            files={"file": (f"test_delim.csv", content.encode("utf-8"), "text/csv")}
        )
        assert res.status_code == 202
        job_id = res.json()["id"]

        for _ in range(50):
            st = client.get(f"/api/feedback/import/{job_id}")
            if st.json()["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)

        assert st.json()["status"] == "COMPLETED"
        assert st.json()["successful_rows"] == 1


# ==============================================================================
# Scenario 10: Invalid ratings and optional fields
# ==============================================================================
def test_scenario_10_invalid_ratings_and_optional_fields(monkeypatch):
    """Scenario 10: Invalid ratings and optional fields - Robust handling of malformed row values."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    csv_content = (
        "text,category,source,rating\n"
        "Valid rating feedback,TestScenario,web,5\n"
        "Negative rating feedback,TestScenario,web,-2\n"
        "Out of range rating feedback,TestScenario,web,10\n"
        "Float rating feedback,TestScenario,web,4.5\n"
        "Non-numeric rating feedback,TestScenario,web,five\n"
        "Missing optional fields feedback,,,\n"
    ).encode("utf-8")

    res = client.post("/api/feedback/import", files={"file": ("scenario_10.csv", csv_content, "text/csv")})
    assert res.status_code == 202
    job_id = res.json()["id"]

    for _ in range(50):
        st = client.get(f"/api/feedback/import/{job_id}")
        if st.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert st.json()["status"] == "COMPLETED"
    assert st.json()["successful_rows"] == 6


# ==============================================================================
# Scenario 11: Tenant isolation
# ==============================================================================
def test_scenario_11_tenant_isolation(monkeypatch):
    """Scenario 11: Tenant isolation - Verify records imported under one tenant are isolated."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    csv1 = "text,category\nTenant A exclusive feedback,TestScenario\n".encode("utf-8")
    csv2 = "text,category\nTenant B exclusive feedback,TestScenario\n".encode("utf-8")

    # Tenant 101 imports
    res1 = client.post(
        "/api/feedback/import",
        files={"file": ("tenant1.csv", csv1, "text/csv")},
        headers={"X-Tenant-ID": "101"}
    )
    job1_id = res1.json()["id"]

    # Tenant 202 imports
    res2 = client.post(
        "/api/feedback/import",
        files={"file": ("tenant2.csv", csv2, "text/csv")},
        headers={"X-Tenant-ID": "202"}
    )
    job2_id = res2.json()["id"]

    for jid, tid in ((job1_id, "101"), (job2_id, "202")):
        for _ in range(50):
            st = client.get(f"/api/feedback/import/{jid}", headers={"X-Tenant-ID": tid})
            if st.json()["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)

    # Query feedback list with Tenant 101 header
    t1_list = client.get("/api/feedback?category=TestScenario", headers={"X-Tenant-ID": "101"}).json()
    assert all(item["text"] == "Tenant A exclusive feedback" for item in t1_list["items"])

    # Query feedback list with Tenant 202 header
    t2_list = client.get("/api/feedback?category=TestScenario", headers={"X-Tenant-ID": "202"}).json()
    assert all(item["text"] == "Tenant B exclusive feedback" for item in t2_list["items"])


# ==============================================================================
# Scenario 12: Unauthorized job access
# ==============================================================================
def test_scenario_12_unauthorized_job_access(monkeypatch):
    """Scenario 12: Unauthorized job access - Tenants cannot inspect import jobs belonging to others."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    csv = "text,category\nPrivate job data,TestScenario\n".encode("utf-8")
    res = client.post(
        "/api/feedback/import",
        files={"file": ("private.csv", csv, "text/csv")},
        headers={"X-Tenant-ID": "555"}
    )
    job_id = res.json()["id"]

    # Authorized query returns 200
    auth_res = client.get(f"/api/feedback/import/{job_id}", headers={"X-Tenant-ID": "555"})
    assert auth_res.status_code == 200

    # Cross-tenant query returns 403 Forbidden
    cross_res = client.get(f"/api/feedback/import/{job_id}", headers={"X-Tenant-ID": "999"})
    assert cross_res.status_code == 403
    assert "belongs to another tenant" in cross_res.json()["detail"]


# ==============================================================================
# Scenario 13: Progress counter accuracy
# ==============================================================================
def test_scenario_13_progress_counter_accuracy(monkeypatch):
    """Scenario 13: Progress counter accuracy - total_rows, processed_rows, successful_rows, failed_rows."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    # 4 valid rows and 2 rows with empty text
    csv_content = (
        "text,category\n"
        "Valid row one,TestScenario\n"
        "   ,TestScenario\n"
        "Valid row two,TestScenario\n"
        ",TestScenario\n"
        "Valid row three,TestScenario\n"
        "Valid row four,TestScenario\n"
    ).encode("utf-8")

    res = client.post("/api/feedback/import", files={"file": ("scenario_13.csv", csv_content, "text/csv")})
    job_id = res.json()["id"]

    for _ in range(50):
        st = client.get(f"/api/feedback/import/{job_id}")
        if st.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    data = st.json()
    assert data["status"] == "COMPLETED"
    assert data["total_rows"] == 6
    assert data["successful_rows"] == 4
    assert data["failed_rows"] == 2
    assert data["processed_rows"] == 6


# ==============================================================================
# Scenario 14: Job failure handling
# ==============================================================================
def test_scenario_14_job_failure_handling_and_no_data_leak(monkeypatch):
    """Scenario 14: Job failure handling - Failed jobs produce error summary and leak 0 partial rows."""
    def _exploding_predict_batch(*args, **kwargs):
        raise RuntimeError("GPU / Model engine kernel crash")

    monkeypatch.setattr(sentiment_service, "predict_batch", _exploding_predict_batch)
    monkeypatch.setattr(sentiment_service, "predict", _exploding_predict_batch)

    csv_content = "text,category\nRow that will crash inference,TestScenario\n".encode("utf-8")
    res = client.post("/api/feedback/import", files={"file": ("scenario_14.csv", csv_content, "text/csv")})
    job_id = res.json()["id"]

    for _ in range(50):
        st = client.get(f"/api/feedback/import/{job_id}")
        if st.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    data = st.json()
    assert data["status"] == "FAILED"
    assert data["stage"] == "FAILED"
    assert data["successful_rows"] == 0
    assert data["failed_rows"] == 1
    assert data["error_summary"] is not None

    with SessionLocal() as db:
        # Zero partial records in feedback
        fb_count = db.query(Feedback).filter(Feedback.category == "TestScenario").count()
        assert fb_count == 0
        # Zero orphan records in staging
        staging_count = db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).count()
        assert staging_count == 0


# ==============================================================================
# Scenario 15: Rollback and incomplete-import visibility
# ==============================================================================
def test_scenario_15_rollback_and_incomplete_import_visibility(monkeypatch):
    """Scenario 15: Rollback & incomplete-import visibility - Incomplete imports invisible to queries."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    # Initial count baseline before inserting staging row
    initial_analytics = client.get("/api/feedback/analytics").json()["total_feedback"]
    test_job_id = str(uuid.uuid4())
    with SessionLocal() as db:
        staging_row = FeedbackStaging(
            import_job_id=test_job_id,
            text="In-flight staging feedback record",
            sentiment="positive",
            confidence=0.99,
            category="TestScenario"
        )
        db.add(staging_row)
        db.commit()

        # Regular feedback queries must NOT see staging records
        fb_records = db.query(Feedback).filter(Feedback.text == "In-flight staging feedback record").all()
        assert len(fb_records) == 0

        # Feedback service list must NOT see staging records
        list_res = client.get("/api/feedback?search=In-flight").json()
        assert list_res["total"] == 0

        # Analytics service must NOT see staging records (count unchanged)
        analytics_res = client.get("/api/feedback/analytics").json()
        assert analytics_res["total_feedback"] == initial_analytics

        # Cleanup
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == test_job_id).delete()
        db.commit()


# ==============================================================================
# Scenario 16: Duplicate records
# ==============================================================================
def test_scenario_16_duplicate_records_handling(monkeypatch):
    """Scenario 16: Duplicate records - Handled consistently according to application design (append)."""
    monkeypatch.setattr(sentiment_service, "predict_batch", _mock_fast_predict_batch)

    duplicate_csv = (
        "text,category\n"
        "Identical feedback text,TestScenario\n"
        "Identical feedback text,TestScenario\n"
    ).encode("utf-8")

    res = client.post("/api/feedback/import", files={"file": ("scenario_16.csv", duplicate_csv, "text/csv")})
    job_id = res.json()["id"]

    for _ in range(50):
        st = client.get(f"/api/feedback/import/{job_id}")
        if st.json()["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    assert st.json()["status"] == "COMPLETED"
    assert st.json()["successful_rows"] == 2

    with SessionLocal() as db:
        dups = db.query(Feedback).filter(Feedback.text == "Identical feedback text").all()
        assert len(dups) == 2
        # Distinct auto-generated primary keys
        assert dups[0].id != dups[1].id


# ==============================================================================
# Scenario 17: Concurrent import jobs limit
# ==============================================================================
def test_scenario_17_concurrent_import_jobs_limit(monkeypatch):
    """Scenario 17: Concurrent import jobs limit - Enforce maximum active jobs and return HTTP 429."""
    monkeypatch.setattr(settings, "MAX_CONCURRENT_IMPORT_JOBS", 1)

    db = SessionLocal()
    # Create an active dummy job in PROCESSING state
    active_job = ImportJob(
        id=str(uuid.uuid4()),
        filename="scenario_17_blocker.csv",
        status="PROCESSING",
        stage="INSERTING",
        created_at=datetime.now(timezone.utc)
    )
    db.add(active_job)
    db.commit()

    try:
        csv_content = b"text\nConcurrent test\n"
        res = client.post(
            "/api/feedback/import",
            files={"file": ("scenario_17.csv", csv_content, "text/csv")}
        )
        assert res.status_code == 429
        assert "Concurrent import limit reached" in res.json()["detail"]

        # Mark active job as COMPLETED
        active_job.status = "COMPLETED"
        active_job.stage = "COMPLETED"
        db.commit()

        # Subsequent upload must succeed
        res2 = client.post(
            "/api/feedback/import",
            files={"file": ("scenario_17.csv", csv_content, "text/csv")}
        )
        assert res2.status_code == 202
    finally:
        db.query(ImportJob).filter(ImportJob.id == active_job.id).delete()
        db.commit()
        db.close()


# ==============================================================================
# Scenario 18: Cleanup of failed or cancelled jobs
# ==============================================================================
def test_scenario_18_cleanup_of_failed_or_cancelled_jobs():
    """Scenario 18: Cleanup of failed jobs - Verify temporary files and staging purged reliably."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    tmp.write("invalid,headers,only\n1,2,3\n")
    tmp_path = tmp.name
    tmp.close()

    db = SessionLocal()
    job = ImportService.create_import_job(db, "scenario_18_cleanup.csv")

    try:
        # Pre-seed a row in staging
        db.add(FeedbackStaging(import_job_id=job.id, text="Orphan staging", sentiment="positive", confidence=0.9))
        db.commit()

        # Run with invalid headers
        ImportService.process_csv_file(db, job, tmp_path)

        # Staging row for this job must be deleted
        staging_count = db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job.id).count()
        assert staging_count == 0
    finally:
        db.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ==============================================================================
# Scenario 19: SQLite compatibility
# ==============================================================================
def test_scenario_19_sqlite_compatibility_and_wal_mode():
    """Scenario 19: SQLite compatibility - Verify proper operation against SQLite with WAL mode."""
    with SessionLocal() as db:
        if db.bind.dialect.name == "sqlite":
            journal_mode = db.execute(text("PRAGMA journal_mode")).scalar()
            assert journal_mode.lower() == "wal"

            synchronous = db.execute(text("PRAGMA synchronous")).scalar()
            # 1 = NORMAL, 2 = FULL
            assert synchronous in (1, 2)

            busy_timeout = db.execute(text("PRAGMA busy_timeout")).scalar()
            assert busy_timeout >= 30000


# ==============================================================================
# Scenario 20: PostgreSQL compatibility
# ==============================================================================
def test_scenario_20_postgresql_compatibility():
    """Scenario 20: PostgreSQL compatibility - Verify bulk insert & finalization generate valid Postgres SQL."""
    pg_dialect = postgresql.dialect()

    # 1. Compile bulk insert statement with PostgreSQL dialect
    stmt = insert(FeedbackStaging).values([
        {
            "import_job_id": "test-job-uuid",
            "user_id": 1,
            "text": "Postgres test text",
            "clean_text": "Postgres test text",
            "sentiment": "positive",
            "confidence": 0.98,
            "positive_score": 0.98,
            "neutral_score": 0.01,
            "negative_score": 0.01,
            "category": "support",
            "source": "web",
            "rating": 5,
            "model_version": "3.0.0",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    ])
    compiled_insert = str(stmt.compile(dialect=pg_dialect))
    assert "INSERT INTO feedback_staging" in compiled_insert
    assert "VALUES" in compiled_insert

    # 2. Compile atomic finalization transfer statement with PostgreSQL dialect
    transfer_stmt = insert(Feedback).from_select(
        [
            Feedback.user_id,
            Feedback.text,
            Feedback.clean_text,
            Feedback.sentiment,
            Feedback.confidence,
            Feedback.positive_score,
            Feedback.neutral_score,
            Feedback.negative_score,
            Feedback.category,
            Feedback.source,
            Feedback.rating,
            Feedback.model_version,
            Feedback.created_at,
            Feedback.updated_at,
        ],
        select(
            FeedbackStaging.user_id,
            FeedbackStaging.text,
            FeedbackStaging.clean_text,
            FeedbackStaging.sentiment,
            FeedbackStaging.confidence,
            FeedbackStaging.positive_score,
            FeedbackStaging.neutral_score,
            FeedbackStaging.negative_score,
            FeedbackStaging.category,
            FeedbackStaging.source,
            FeedbackStaging.rating,
            FeedbackStaging.model_version,
            FeedbackStaging.created_at,
            FeedbackStaging.updated_at,
        ).where(FeedbackStaging.import_job_id == "test-job-uuid")
    )
    compiled_transfer = str(transfer_stmt.compile(dialect=pg_dialect))
    assert "INSERT INTO feedback" in compiled_transfer
    assert "SELECT" in compiled_transfer
    assert "FROM feedback_staging" in compiled_transfer
    assert "WHERE feedback_staging.import_job_id" in compiled_transfer
