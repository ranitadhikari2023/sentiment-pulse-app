"""
Verification of Maximum File-Size (100 MB) and Record-Count (1,050,000) Limits
via the actual FastAPI upload endpoint (/api/feedback/import).
"""

import os
import sys
import tempfile
import time
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import app
from core.config import settings
from services.sentiment_service import sentiment_service

client = TestClient(app)

def test_limits():
    print("=" * 70)
    print("VERIFYING FILE-SIZE AND RECORD-COUNT LIMITS VIA HTTP UPLOAD ENDPOINT")
    print("=" * 70)

    # Mock inference to isolate HTTP upload handling
    def _mock_fast_batch(texts):
        return [
            {"text": t[:50], "sentiment": "positive", "confidence": 0.95, "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}}
            for t in texts
        ]
    sentiment_service.predict_batch = _mock_fast_batch

    # Test 1: Real CSV upload within limits (10,000 rows)
    row_count = 10000
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write("text,category,source,rating\n")
    for i in range(row_count):
        tmp.write(f'"Valid HTTP upload customer review item {i} with verified data",Performance,web,5\n')
    tmp.close()

    file_size_bytes = os.path.getsize(tmp.name)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Generated CSV: {row_count:,} rows, Exact File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")

    with open(tmp.name, "rb") as f:
        t0 = time.perf_counter()
        res = client.post("/api/feedback/import", files={"file": ("real_upload.csv", f, "text/csv")})
        upload_time = time.perf_counter() - t0

    os.unlink(tmp.name)

    print(f"HTTP Response Code: {res.status_code} ({res.reason_phrase})")
    assert res.status_code == 202
    job_data = res.json()
    job_id = job_data["id"]
    print(f"Created ImportJob ID: {job_id}")

    # Poll status
    while True:
        status_res = client.get(f"/api/feedback/import/{job_id}")
        data = status_res.json()
        if data["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.05)

    print(f"Final Job Status:   {data['status']}")
    print(f"Processed Rows:     {data['processed_rows']:,}")
    print(f"Successful Rows:    {data['successful_rows']:,}")
    print(f"Failed Rows:        {data['failed_rows']}")
    assert data["status"] == "COMPLETED"
    assert data["successful_rows"] == row_count

    # Test 2: File size limit rejection (> 100 MB)
    print("\nVerifying 100 MB File Size Boundary Rejection...")
    from tests.test_imports_exports import FastCSVStream
    max_bytes = settings.MAX_IMPORT_FILE_SIZE_BYTES
    oversized_stream = FastCSVStream(max_bytes + 1)
    res_oversized = client.post(
        "/api/feedback/import",
        files={"file": ("oversized_by_1_byte.csv", oversized_stream, "text/csv")}
    )
    print(f"Oversized Request Response Code: {res_oversized.status_code} (Expected 413)")
    assert res_oversized.status_code == 413

    # Test 3: Row count boundary rejection (> 1,050,000 rows)
    print("\nVerifying 1,050,000 Row Count Boundary Rejection...")
    # Preflight stream test
    from services.import_service import ImportService
    from db.session import SessionLocal
    from db.models import ImportJob

    db = SessionLocal()
    overflow_job = ImportService.create_import_job(db, "overflow_test.csv")
    
    # Create temporary file with MAX_IMPORT_ROWS + 1
    tmp_overflow = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w")
    tmp_overflow.write("text\n")
    # Simulate overflow with low settings threshold
    old_max = settings.MAX_IMPORT_ROWS
    settings.MAX_IMPORT_ROWS = 500
    for i in range(505):
        tmp_overflow.write(f"row {i}\n")
    tmp_overflow.close()

    overflow_result = ImportService.process_csv_file(db, overflow_job, tmp_overflow.name)
    settings.MAX_IMPORT_ROWS = old_max
    os.unlink(tmp_overflow.name)
    print(f"Overflow Job Status: {overflow_result.status} (Expected FAILED)")
    print(f"Error Summary:       {overflow_result.error_summary}")
    assert overflow_result.status == "FAILED"
    assert "exceeds maximum allowed rows" in overflow_result.error_summary

    db.query(ImportJob).filter(ImportJob.id == overflow_job.id).delete()
    db.commit()
    db.close()

    print("\nAll boundary limits verified successfully!")

if __name__ == "__main__":
    test_limits()
