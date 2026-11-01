"""
Test suite for CSV batch imports, job tracking, and secure exports.
Validates 100 MB upload limits, bounded streaming parsing, formula injection sanitization,
row count boundaries, and export functionality without authentication.
"""

import io
import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from core.config import settings
from services.import_service import ImportService


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client


class FastCSVStream:
    """
    Zero-memory streaming file-like generator that produces valid CSV content of exact byte size
    with rows under MAX_IMPORT_ROWS (<= 5000) and fields well within the 1 MB CSV limit.
    """
    def __init__(self, size: int, num_rows: int = 500):
        self.size = size
        self.read_so_far = 0
        self.header = b"text,category,rating,extra\n"

        body_bytes = size - len(self.header)
        if body_bytes <= 0:
            self.header = self.header[:size]
            self.num_rows = 0
            self.standard_row = b""
            self.last_row = b""
        else:
            self.num_rows = num_rows
            self.row_size = body_bytes // num_rows
            self.remainder = body_bytes % num_rows

            prefix = b'"Outstanding customer experience and quality!",Support,5,"'
            suffix = b'"\n'
            pad_len = max(0, self.row_size - len(prefix) - len(suffix))
            self.standard_row = prefix + (b"X" * pad_len) + suffix

            last_pad_len = max(0, self.row_size + self.remainder - len(prefix) - len(suffix))
            self.last_row = prefix + (b"X" * last_pad_len) + suffix

        self._current_row_idx = 0
        self._current_buf = self.header
        self._buf_pos = 0

    def read(self, n: int = 65536) -> bytes:
        if self.read_so_far >= self.size:
            return b""

        chunks = []
        bytes_needed = min(n, self.size - self.read_so_far)

        while bytes_needed > 0:
            avail = len(self._current_buf) - self._buf_pos
            if avail > 0:
                take = min(bytes_needed, avail)
                chunks.append(self._current_buf[self._buf_pos:self._buf_pos + take])
                self._buf_pos += take
                bytes_needed -= take
                self.read_so_far += take
            else:
                if self._current_row_idx < self.num_rows - 1:
                    self._current_buf = self.standard_row
                    self._buf_pos = 0
                    self._current_row_idx += 1
                elif self._current_row_idx == self.num_rows - 1:
                    self._current_buf = self.last_row
                    self._buf_pos = 0
                    self._current_row_idx += 1
                else:
                    break
        return b"".join(chunks)


def test_csv_upload_and_job_processing(client):
    """Verifies that a valid CSV below the limit imports successfully with ONNX inference."""
    csv_data = """text,category,rating
"The application is fast and intuitive.",UX,5
"The service failed to connect to database.",Bug,1
"Average experience overall.",General,3
"""
    file_bytes = io.BytesIO(csv_data.encode("utf-8"))
    upload_res = client.post(
        "/feedback/import",
        files={"file": ("customer_reviews.csv", file_bytes, "text/csv")}
    )
    assert upload_res.status_code == 202
    job_info = upload_res.json()
    assert job_info["filename"] == "customer_reviews.csv"
    assert job_info["status"] in ("PENDING", "PROCESSING", "COMPLETED")
    job_id = job_info["id"]

    # Poll status
    status_res = client.get(f"/feedback/import/{job_id}")
    assert status_res.status_code == 200
    job_status = status_res.json()
    assert job_status["status"] == "COMPLETED"
    assert job_status["total_rows"] == 3
    assert job_status["successful_rows"] == 3
    assert job_status["failed_rows"] == 0

    # Verify feedback records were actually persisted
    list_res = client.get("/feedback")
    assert list_res.status_code == 200
    assert list_res.json()["total"] >= 3


def test_csv_upload_at_exact_100mb_boundary(client, monkeypatch):
    """
    BOUNDARY TEST:
    Verifies that a valid CSV file of exactly 100 MB (104,857,600 bytes)
    is accepted by the server and completes processing.
    """
    # Fast mock for inference to keep test execution fast
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "positive",
            "confidence": 0.98,
            "scores": {"positive": 0.98, "neutral": 0.01, "negative": 0.01}
        }
    )

    exact_size = settings.MAX_IMPORT_FILE_SIZE_BYTES  # 104,857,600 bytes
    stream = FastCSVStream(exact_size)

    res = client.post(
        "/feedback/import",
        files={"file": ("boundary_100mb.csv", stream, "text/csv")}
    )
    assert res.status_code == 202
    job_data = res.json()
    assert job_data["status"] in ("PENDING", "PROCESSING", "COMPLETED")

    # Poll status endpoint to verify background completion
    status_res = client.get(f"/feedback/import/{job_data['id']}")
    assert status_res.status_code == 200
    final_job = status_res.json()
    assert final_job["status"] == "COMPLETED"
    assert final_job["total_rows"] == 500
    assert final_job["successful_rows"] == 500


def test_csv_upload_exceeding_100mb_by_one_byte_rejected(client):
    """
    BOUNDARY TEST:
    Verifies that an upload exceeding 100 MB by exactly 1 byte (104,857,601 bytes)
    is rejected with HTTP 413 Request Entity Too Large.
    """
    oversized = settings.MAX_IMPORT_FILE_SIZE_BYTES + 1  # 104,857,601 bytes
    stream = FastCSVStream(oversized)

    res = client.post(
        "/feedback/import",
        files={"file": ("oversized_by_1_byte.csv", stream, "text/csv")}
    )
    assert res.status_code == 413
    body = res.json()
    assert "Uploaded file exceeds maximum allowed size" in body["detail"]
    assert "100 MB" in body["detail"]


def test_csv_upload_substantially_oversized_rejected(client):
    """
    Verifies that substantially oversized uploads (e.g. 120 MB)
    are rejected immediately with HTTP 413.
    """
    substantially_oversized = 120 * 1024 * 1024  # 120 MB
    stream = FastCSVStream(substantially_oversized)

    res = client.post(
        "/feedback/import",
        files={"file": ("oversized_120mb.csv", stream, "text/csv")}
    )
    assert res.status_code == 413
    body = res.json()
    assert "Uploaded file exceeds maximum allowed size" in body["detail"]


def test_empty_csv_file_rejected(client):
    """Verifies that an empty CSV file (0 bytes) is rejected with HTTP 400."""
    empty_bytes = io.BytesIO(b"")
    res = client.post(
        "/feedback/import",
        files={"file": ("empty.csv", empty_bytes, "text/csv")}
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()


def test_invalid_file_extension_rejected(client):
    """Verifies that non-CSV file extensions are rejected with HTTP 400."""
    file_bytes = io.BytesIO(b"binary data")
    res = client.post(
        "/feedback/import",
        files={"file": ("notes.txt", file_bytes, "text/plain")}
    )
    assert res.status_code == 400
    assert "Only CSV files are accepted" in res.json()["detail"]


def test_invalid_csv_headers_rejected(client):
    """
    Verifies that a CSV missing required text/feedback column
    fails gracefully with a descriptive error summary in the ImportJob.
    """
    bad_csv = "category,rating,timestamp\nUX,5,2026-01-01\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("bad_headers.csv", io.BytesIO(bad_csv.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    assert job["status"] == "FAILED"
    errors = json.loads(job["error_summary"])
    assert any("must contain a 'text'" in e for e in errors)


def test_csv_with_no_data_rows(client):
    """Verifies that a CSV with only a header and no data rows is marked as FAILED."""
    only_header = "text,category,rating\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("no_data.csv", io.BytesIO(only_header.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    status_res = client.get(f"/feedback/import/{job['id']}")
    assert status_res.status_code == 200
    final_job = status_res.json()
    assert final_job["status"] == "FAILED"
    errors = json.loads(final_job["error_summary"])
    assert any("contains no data rows" in e for e in errors)


def test_csv_row_count_exceeding_1050000_rejected(client):
    """
    Verifies that CSV files containing more than 1,050,000 data rows (1,050,000)
    are safely rejected without partial ingestion.
    """
    class OversizedCSVStream(io.RawIOBase):
        def __init__(self, num_rows=1050001):
            self.num_rows = num_rows
            self.header = b"text,category,rating\n"
            self.row_template = b"Review,General,4\n"
            self.header_sent = False
            self.rows_sent = 0
            self.buffer = b""

        def readable(self):
            return True

        def readinto(self, b):
            if not self.header_sent:
                self.buffer += self.header
                self.header_sent = True

            while len(self.buffer) < len(b) and self.rows_sent < self.num_rows:
                batch = min(10000, self.num_rows - self.rows_sent)
                self.buffer += self.row_template * batch
                self.rows_sent += batch

            if not self.buffer:
                return 0

            n = min(len(b), len(self.buffer))
            b[:n] = self.buffer[:n]
            self.buffer = self.buffer[n:]
            return n

    res = client.post(
        "/feedback/import",
        files={"file": ("too_many_rows.csv", OversizedCSVStream(1050001), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    status_res = client.get(f"/feedback/import/{job['id']}")
    assert status_res.status_code == 200
    final_job = status_res.json()
    assert final_job["status"] == "FAILED"
    assert final_job["total_rows"] == 1050001
    assert final_job["successful_rows"] == 0
    assert final_job["failed_rows"] == 0
    errors = json.loads(final_job["error_summary"])
    assert any("1050001 > 1050000" in e for e in errors)
    assert any("Maximum batch limit is 1,050,000 records" in e for e in errors)


def test_supported_encodings(client):
    """
    Verifies that UTF-8 with BOM (utf-8-sig) and Latin-1 (ISO-8859-1) encodings
    are correctly detected and decoded.
    """
    # 1. UTF-8 with BOM (utf-8-sig automatically attaches \xef\xbb\xbf)
    bom_csv = "text,category,rating\n\"Great with BOM!\",BOM,5\n".encode("utf-8-sig")
    res_bom = client.post(
        "/feedback/import",
        files={"file": ("bom.csv", io.BytesIO(bom_csv), "text/csv")}
    )
    assert res_bom.status_code == 202
    job_bom = client.get(f"/feedback/import/{res_bom.json()['id']}").json()
    assert job_bom["status"] == "COMPLETED"

    # Also verify explicit raw BOM bytes
    raw_bom = b"\xef\xbb\xbftext,category,rating\n\"Explicit raw BOM!\",BOM,5\n"
    res_raw_bom = client.post(
        "/feedback/import",
        files={"file": ("raw_bom.csv", io.BytesIO(raw_bom), "text/csv")}
    )
    assert res_raw_bom.status_code == 202
    job_raw_bom = client.get(f"/feedback/import/{res_raw_bom.json()['id']}").json()
    assert job_raw_bom["status"] == "COMPLETED"

    # 2. Latin-1 / ISO-8859-1 with accented characters
    latin1_text = "text,category,rating\n\"Café & Crème brûlée service quality!\",Dining,5\n"
    latin1_bytes = latin1_text.encode("latin-1")
    res_latin = client.post(
        "/feedback/import",
        files={"file": ("latin1.csv", io.BytesIO(latin1_bytes), "text/csv")}
    )
    assert res_latin.status_code == 202
    job_latin = client.get(f"/feedback/import/{res_latin.json()['id']}").json()
    assert job_latin["status"] == "COMPLETED"


def test_row_level_validation_errors(client):
    """
    Verifies that individual corrupt or empty rows are tracked in failed_rows
    and error_summary while valid rows are successfully processed.
    """
    mixed_csv = """text,category,rating
"Valid review text",UX,5
,UX,5
"Missing category and rating but valid text"
"""
    res = client.post(
        "/feedback/import",
        files={"file": ("mixed.csv", io.BytesIO(mixed_csv.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    init_job = res.json()
    status_res = client.get(f"/feedback/import/{init_job['id']}")
    assert status_res.status_code == 200
    job = status_res.json()
    assert job["status"] == "COMPLETED"
    assert job["total_rows"] == 3
    assert job["successful_rows"] == 2
    assert job["failed_rows"] == 1
    errors = json.loads(job["error_summary"])
    assert any("Row 2: Empty text" in e for e in errors)


def test_temporary_files_cleaned_up_on_upload(client):
    """Verifies that temporary files are deleted after upload processing."""
    temp_dir = tempfile.gettempdir()
    before_files = set(os.listdir(temp_dir))

    csv_data = "text,category,rating\n\"Temporary file cleanup test.\",Test,5\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("cleanup_test.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    after_files = set(os.listdir(temp_dir))

    # Verify no dangling temporary files with our pattern remained in tempdir
    new_csv_temps = [f for f in (after_files - before_files) if f.endswith(".csv")]
    assert len(new_csv_temps) == 0


def test_database_error_rolls_back_cleanly(client, monkeypatch):
    """
    TRANSACTION INTEGRITY TEST:
    Verifies that if a database transaction failure occurs after at least one batch
    has been processed, all previous batches are rolled back completely.
    Zero partial records remain in the database, the job status reflects FAILED with
    successful_rows=0, and a subsequent retry imports cleanly without duplicates.
    """
    from db.session import SessionLocal
    from db.models import Feedback
    from sqlalchemy.orm import Session

    # Fast mock for sentiment inference
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "positive",
            "confidence": 0.95,
            "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}
        }
    )

    db = SessionLocal()
    try:
        initial_count = db.query(Feedback).count()

        # 250 rows CSV with chunk size = 100 -> spans 3 chunks
        lines = ["text,category,rating\n"] + [f"Atomic Feedback {i},General,5\n" for i in range(250)]
        csv_data = "".join(lines).encode("utf-8")

        original_flush = Session.flush
        flush_count = [0]
        def failing_flush(self):
            flush_count[0] += 1
            # Fail on the 2nd chunk flush (after 1st chunk was processed)
            if flush_count[0] == 2:
                raise RuntimeError("Simulated database engine write failure on chunk 2")
            return original_flush(self)

        monkeypatch.setattr(Session, "flush", failing_flush)

        res = client.post(
            "/feedback/import",
            files={"file": ("atomic_fail_test.csv", io.BytesIO(csv_data), "text/csv")}
        )
        assert res.status_code == 202
        job_init = res.json()
        status_res = client.get(f"/feedback/import/{job_init['id']}")
        assert status_res.status_code == 200
        job = status_res.json()
        assert job["status"] == "FAILED"
        assert job["successful_rows"] == 0, "No records must be reported as successful after rollback"
        errors = json.loads(job["error_summary"])
        assert any("rolled back completely" in e.lower() for e in errors)

        # CRITICAL ASSERTION: Assert zero partial records leaked from chunk 1
        post_failure_count = db.query(Feedback).count()
        assert post_failure_count == initial_count, (
            f"Expected {initial_count} records, got {post_failure_count}. "
            "Batch 1 records leaked into database despite failure on batch 2!"
        )

        # Restore flush and verify clean retry without duplicates
        monkeypatch.undo()
        # Re-apply fast mock
        monkeypatch.setattr(
            "services.import_service.sentiment_service.predict",
            lambda text: {
                "text": text[:50],
                "sentiment": "positive",
                "confidence": 0.95,
                "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}
            }
        )

        retry_res = client.post(
            "/feedback/import",
            files={"file": ("atomic_fail_test.csv", io.BytesIO(csv_data), "text/csv")}
        )
        assert retry_res.status_code == 202
        retry_init = retry_res.json()
        status_retry = client.get(f"/feedback/import/{retry_init['id']}")
        assert status_retry.status_code == 200
        retry_job = status_retry.json()
        assert retry_job["status"] == "COMPLETED"
        assert retry_job["successful_rows"] == 250

        final_count = db.query(Feedback).count()
        assert final_count == initial_count + 250, (
            f"Expected {initial_count + 250} records, got {final_count}. "
            "Duplicate records detected after retry!"
        )
    finally:
        db.close()


def test_csv_formula_injection_protection_on_export(client):
    """
    CRITICAL SECURITY TEST:
    Verifies that customer feedback beginning with dangerous spreadsheet formula characters
    (=, +, -, @) are sanitized with a leading single quote (') upon CSV export.
    """
    payloads = [
        "=SUM(A1:A10)",
        "@HYPERLINK('http://attacker.com','Click')",
        "-2+3*cmd|' /C calc'!A0",
        "+cmd|' /C calc'!A0"
    ]
    for text in payloads:
        client.post("/feedback", json={"text": text})

    # Export CSV
    export_res = client.get("/feedback/export")
    assert export_res.status_code == 200
    assert export_res.headers["content-type"].startswith("text/csv")
    csv_text = export_res.text

    # Verify formula prefixes are neutralized with leading quote
    assert "'=SUM(A1:A10)" in csv_text
    assert "'@HYPERLINK('http://attacker.com','Click')" in csv_text
    assert "'-2+3*cmd|' /C calc'!A0" in csv_text
    assert "'+cmd|' /C calc'!A0" in csv_text


def test_csv_field_exceeding_max_text_length_rejected_at_row_level(client):
    """
    Verifies that rows with text exceeding MAX_FEEDBACK_TEXT_LENGTH (10,000 chars)
    are rejected as row-level errors, while valid rows in the same CSV succeed.
    """
    valid_text = "Standard customer feedback under limit."
    oversized_text = "A" * 12000  # 12,000 chars > 10,000 limit
    csv_content = (
        f'text,category,rating\n'
        f'"{valid_text}",Support,5\n'
        f'"{oversized_text}",UX,1\n'
        f'"Another valid review",Product,4\n'
    )
    res = client.post(
        "/feedback/import",
        files={"file": ("field_length_test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    init_job = res.json()
    status_res = client.get(f"/feedback/import/{init_job['id']}")
    assert status_res.status_code == 200
    job = status_res.json()
    assert job["status"] == "COMPLETED"
    assert job["total_rows"] == 3
    assert job["successful_rows"] == 2
    assert job["failed_rows"] == 1
    errors = json.loads(job["error_summary"])
    assert any("Row 2" in e and "Text exceeds maximum allowed length" in e for e in errors)


def test_csv_field_exceeding_1mb_fails_gracefully(client):
    """
    Verifies that a CSV containing a single field exceeding the 1 MB C-parser limit
    (e.g. runaway unclosed quote or massive single token) fails gracefully with
    status FAILED, without memory explosion or crashing the process.
    """
    # 1.2 MB single unclosed quote / runaway field
    huge_field = "text,category,rating\n\"" + ("Z" * (1200 * 1024))
    res = client.post(
        "/feedback/import",
        files={"file": ("runaway_field.csv", io.BytesIO(huge_field.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    init_job = res.json()
    status_res = client.get(f"/feedback/import/{init_job['id']}")
    assert status_res.status_code == 200
    job = status_res.json()
    assert job["status"] == "FAILED"
    assert job["successful_rows"] == 0
    errors = json.loads(job["error_summary"])
    assert any("exceeds maximum allowed size" in e.lower() or "parsing error" in e.lower() for e in errors)


def test_valid_upload_returns_accepted_and_job_id(client):
    """
    ASYNC IMPORT TEST 1:
    Verifies that a valid CSV upload returns HTTP 202 Accepted, a valid UUID job ID,
    and initial status 'PENDING'.
    """
    csv_data = "text,category,rating\n\"Service was helpful and clear.\",Support,5\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("sample.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    assert "id" in job
    assert len(job["id"]) == 36  # UUID length
    assert job["filename"] == "sample.csv"
    assert job["status"] in ("PENDING", "PROCESSING", "COMPLETED")


def test_response_returned_without_awaiting_inference(client, monkeypatch):
    """
    ASYNC IMPORT TEST 2:
    Verifies that the HTTP response is returned without blocking on the sentiment inference
    pipeline, confirming that execution was transferred to background processing.
    """
    task_scheduled = []

    def mock_add_task(self, func, *args, **kwargs):
        task_scheduled.append((func, args, kwargs))

    # Spy on FastAPI BackgroundTasks.add_task
    from fastapi import BackgroundTasks
    monkeypatch.setattr(BackgroundTasks, "add_task", mock_add_task)

    csv_data = "text,category,rating\n\"Instant response test.\",General,4\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("fast_resp.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    assert job["status"] == "PENDING"
    # Verify background task was queued with serializable job_id and file_path
    assert len(task_scheduled) == 1
    func, args, kwargs = task_scheduled[0]
    assert func == ImportService.process_csv_file_background
    assert kwargs.get("job_id") == job["id"]
    assert os.path.exists(kwargs.get("file_path"))

    # Clean up test temp file
    if os.path.exists(kwargs.get("file_path")):
        os.unlink(kwargs.get("file_path"))


def test_background_task_opens_and_closes_own_session(monkeypatch):
    """
    ASYNC IMPORT TEST 3 & 4:
    Verifies that the background worker opens its own dedicated database session
    using SessionLocal and reliably closes it in finally, without depending on a request session.
    """
    import db.session as session_module
    from services.import_service import ImportService
    from db.models import ImportJob
    import uuid

    session_created = []
    session_closed = []

    real_sessionmaker = session_module.SessionLocal

    def spy_sessionmaker(*args, **kwargs):
        sess = real_sessionmaker(*args, **kwargs)
        real_close = sess.close

        def spy_close():
            session_closed.append(True)
            return real_close()

        sess.close = spy_close
        session_created.append(sess)
        return sess

    monkeypatch.setattr(session_module, "SessionLocal", spy_sessionmaker)

    # Pre-create an ImportJob in the DB
    setup_db = real_sessionmaker()
    job_id = str(uuid.uuid4())
    job = ImportJob(
        id=job_id,
        filename="session_test.csv",
        status="PENDING",
        total_rows=0,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0
    )
    setup_db.add(job)
    setup_db.commit()
    setup_db.close()

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        f.write("text,category,rating\n\"Session lifecycle test.\",Support,5\n")
        temp_path = f.name

    try:
        # Fast mock predict
        monkeypatch.setattr(
            "services.import_service.sentiment_service.predict",
            lambda text: {
                "text": text,
                "sentiment": "positive",
                "confidence": 0.9,
                "scores": {"positive": 0.9, "neutral": 0.05, "negative": 0.05}
            }
        )

        ImportService.process_csv_file_background(job_id=job_id, file_path=temp_path)

        # Assert task-owned session was created and closed
        assert len(session_created) >= 1
        assert len(session_closed) >= 1
        # Assert temp file was deleted
        assert not os.path.exists(temp_path)

        # Verify job was reloaded and completed
        verify_db = real_sessionmaker()
        reloaded_job = verify_db.query(ImportJob).filter(ImportJob.id == job_id).first()
        assert reloaded_job is not None
        assert reloaded_job.status == "COMPLETED"
        assert reloaded_job.successful_rows == 1
        verify_db.close()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_temporary_file_ownership_transfer_and_cleanup(client):
    """
    ASYNC IMPORT TEST 5 & 6:
    Verifies that temporary files are deleted after both successful and failed processing,
    and that route failures delete the temp file immediately.
    """
    # Case A: Success cleanup
    csv_data = "text,category,rating\n\"Success cleanup test.\",General,5\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("cleanup_success.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202

    # Case B: Rejection cleanup (e.g. empty file)
    temp_dir = tempfile.gettempdir()
    before_temps = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    res_empty = client.post(
        "/feedback/import",
        files={"file": ("empty_reject.csv", io.BytesIO(b""), "text/csv")}
    )
    assert res_empty.status_code == 400

    after_temps = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert len(after_temps - before_temps) == 0, "Orphan temporary file left after route rejection!"


def test_background_exception_marks_job_as_failed(monkeypatch):
    """
    ASYNC IMPORT TEST 7:
    Verifies that an unhandled exception inside the background task worker
    marks the job as FAILED and records the error without leaving orphan files.
    """
    from db.session import SessionLocal
    from db.models import ImportJob
    import uuid

    db = SessionLocal()
    job_id = str(uuid.uuid4())
    job = ImportJob(
        id=job_id,
        filename="exception_test.csv",
        status="PENDING",
        total_rows=1,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0
    )
    db.add(job)
    db.commit()
    db.close()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        f.write("text,category,rating\n\"Error row\",Bug,1\n")
        temp_path = f.name

    try:
        def failing_process(*args, **kwargs):
            raise RuntimeError("Simulated background worker crash")

        monkeypatch.setattr(ImportService, "process_csv_file", failing_process)

        # Run background task
        ImportService.process_csv_file_background(job_id=job_id, file_path=temp_path)

        # Verify job marked as FAILED
        check_db = SessionLocal()
        failed_job = check_db.query(ImportJob).filter(ImportJob.id == job_id).first()
        assert failed_job.status == "FAILED"
        assert failed_job.successful_rows == 0
        errors = json.loads(failed_job.error_summary)
        assert any("Background task execution failed" in e for e in errors)
        check_db.close()

        # Verify temp file was unlinked
        assert not os.path.exists(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_polling_endpoint_reports_all_states(client):
    """
    ASYNC IMPORT TEST 10:
    Verifies that the polling endpoint GET /feedback/import/{id}
    accurately reports PENDING, PROCESSING, COMPLETED, and FAILED states.
    """
    from db.session import SessionLocal
    from db.models import ImportJob
    import uuid

    db = SessionLocal()
    states = ["PENDING", "PROCESSING", "COMPLETED", "FAILED"]
    job_ids = {}

    for state in states:
        jid = str(uuid.uuid4())
        job = ImportJob(
            id=jid,
            filename=f"test_{state.lower()}.csv",
            status=state,
            total_rows=10,
            processed_rows=5 if state == "PROCESSING" else 10,
            successful_rows=10 if state == "COMPLETED" else 0,
            failed_rows=10 if state == "FAILED" else 0,
            error_summary=json.dumps(["Sample failure"]) if state == "FAILED" else None
        )
        db.add(job)
        job_ids[state] = jid
    db.commit()
    db.close()

    for state, jid in job_ids.items():
        res = client.get(f"/feedback/import/{jid}")
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == jid
        assert data["status"] == state
        if state == "PROCESSING":
            assert data["processed_rows"] == 5
            assert data["total_rows"] == 10
        elif state == "COMPLETED":
            assert data["successful_rows"] == 10
        elif state == "FAILED":
            assert data["failed_rows"] == 10
            assert "Sample failure" in data["error_summary"]


def test_streaming_byte_count_enforcement_without_content_length(client):
    """
    ASYNC IMPORT TEST 12:
    Verifies that a streaming upload exceeding 100 MB cannot bypass size limits
    even when Content-Length header is missing or inaccurate.
    """
    oversized = settings.MAX_IMPORT_FILE_SIZE_BYTES + 1024
    stream = FastCSVStream(oversized)

    # Use a custom generator to simulate streaming without standard Content-Length
    res = client.post(
        "/feedback/import",
        files={"file": ("stream_oversized.csv", stream, "text/csv")}
    )
    assert res.status_code == 413
    assert "Uploaded file exceeds maximum allowed size" in res.json()["detail"]


def test_polling_visibility_from_separate_session_during_processing(monkeypatch):
    """
    REGRESSION TEST:
    Verifies progress and job status visibility through an independent database session
    while background ingestion is actively in progress.
    Confirms:
    1. status='PROCESSING' and total_rows are committed and visible to concurrent sessions.
    2. Atomic transaction isolation prevents uncommitted feedback rows from leaking into other sessions.
    3. status='COMPLETED' and all feedback rows become visible only after final commit.
    """
    from db.session import SessionLocal
    from db.models import ImportJob, Feedback
    import uuid

    # Setup isolated test database session
    db_setup = SessionLocal()
    job_id = str(uuid.uuid4())
    job = ImportJob(
        id=job_id,
        filename="concurrent_test.csv",
        status="PENDING",
        total_rows=0,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0
    )
    db_setup.add(job)
    db_setup.commit()
    initial_fb_count = db_setup.query(Feedback).count()
    db_setup.close()

    # Create temporary CSV file with 3 rows
    csv_text = (
        "text,category,rating\n"
        "\"First review\",UX,5\n"
        "\"Second review\",Support,4\n"
        "\"Third review\",Bug,2\n"
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        f.write(csv_text)
        temp_path = f.name

    # Separate independent polling database session
    polling_db = SessionLocal()
    mid_processing_observed = []

    def mocked_predict_with_concurrent_poll(text):
        # Query job from completely separate database session while worker transaction is open
        polled_job = polling_db.query(ImportJob).filter(ImportJob.id == job_id).first()
        current_fb_count = polling_db.query(Feedback).count()
        mid_processing_observed.append({
            "status": polled_job.status if polled_job else None,
            "total_rows": polled_job.total_rows if polled_job else None,
            "fb_count": current_fb_count
        })
        return {
            "text": text,
            "sentiment": "positive",
            "confidence": 0.95,
            "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}
        }

    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        mocked_predict_with_concurrent_poll
    )

    try:
        # Run worker
        ImportService.process_csv_file_background(job_id=job_id, file_path=temp_path)

        # 1. Verify that during processing, concurrent session observed PROCESSING and total_rows
        assert len(mid_processing_observed) >= 1
        for obs in mid_processing_observed:
            assert obs["status"] == "PROCESSING", "Job status must be visible as PROCESSING during ingestion"
            assert obs["total_rows"] == 3, "total_rows must be committed and visible during ingestion"
            assert obs["fb_count"] == initial_fb_count, "Uncommitted feedback records must not leak to other sessions!"

        # 2. Verify that after worker completes, final commit makes COMPLETED and records visible
        polling_db.expire_all()
        final_job = polling_db.query(ImportJob).filter(ImportJob.id == job_id).first()
        assert final_job.status == "COMPLETED"
        assert final_job.total_rows == 3
        assert final_job.successful_rows == 3

        final_fb_count = polling_db.query(Feedback).count()
        assert final_fb_count == initial_fb_count + 3
    finally:
        polling_db.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_invalid_header_failure_paths(client):
    """
    Verifies all invalid header validation failure paths:
    1. Whitespace-only file (StopIteration on header read)
    2. Malformed quote in header
    3. Missing text column
    In all cases: returns HTTP 202 with FAILED ImportJob, persists error detail,
    and unlinks temporary file without leaking resources.
    """
    temp_dir = tempfile.gettempdir()

    # Path 1: Whitespace-only file
    before_files = set(os.listdir(temp_dir))
    res1 = client.post(
        "/feedback/import",
        files={"file": ("whitespace.csv", io.BytesIO(b"   \n\n\r\n  \n"), "text/csv")}
    )
    assert res1.status_code == 202
    job1 = res1.json()
    assert job1["status"] == "FAILED"
    errors1 = json.loads(job1["error_summary"])
    assert any("must contain a 'text'" in e or "empty" in e.lower() for e in errors1)

    after_files1 = set(os.listdir(temp_dir))
    assert len(after_files1 - before_files) == 0, "Temporary file leaked on whitespace header failure!"

    # Path 2: Missing text column
    res2 = client.post(
        "/feedback/import",
        files={"file": ("no_text_col.csv", io.BytesIO(b"user,score,date\nalice,5,2026-01-01\n"), "text/csv")}
    )
    assert res2.status_code == 202
    job2 = res2.json()
    assert job2["status"] == "FAILED"
    errors2 = json.loads(job2["error_summary"])
    assert any("must contain a 'text'" in e for e in errors2)

    # Path 3: Verify GET /feedback/import/{id} retrieves the durable FAILED job
    poll_res = client.get(f"/feedback/import/{job2['id']}")
    assert poll_res.status_code == 200
    polled_job = poll_res.json()
    assert polled_job["status"] == "FAILED"
    assert "must contain a 'text'" in polled_job["error_summary"]


# ============================================================================
# TARGETED RELIABILITY REGRESSION TESTS: CSV TEMPORARY-FILE CLEANUP & LIFECYCLE
# ============================================================================


def test_upload_file_read_exception_cleans_up_temp_file(client):
    """
    Simulates a network abort or socket failure during file.read(CHUNK_SIZE).
    Asserts that the temporary file handle is closed and the file is unlinked on disk,
    leaving no orphaned temporary files behind.
    """
    from starlette.datastructures import UploadFile

    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def spy_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=spy_temp_file):
        with patch.object(UploadFile, "read", side_effect=IOError("Simulated network abort")):
            with pytest.raises(IOError, match="Simulated network abort"):
                client.post(
                    "/feedback/import",
                    files={"file": ("stream_error.csv", io.BytesIO(b"text\nHello"), "text/csv")}
                )

    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.closed, "Temporary file handle must be closed!"
    assert not os.path.exists(tf.name), f"Temp file {tf.name} should have been unlinked!"
    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert not (after_csv - before_csv), "No orphan temporary CSV files should remain!"


def test_temp_file_write_exception_cleans_up_temp_file(client):
    """
    Simulates a disk full / write error during temp_file.write(chunk).
    Asserts that the file handle is closed and the file is unlinked without orphan files.
    """
    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def faulty_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)

        def mock_write(data):
            raise OSError("Disk full: no space left on device")

        tf.write = mock_write
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=faulty_temp_file):
        with pytest.raises(OSError, match="Disk full"):
            client.post(
                "/feedback/import",
                files={"file": ("disk_full.csv", io.BytesIO(b"text\nHello"), "text/csv")}
            )

    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.closed, "File handle must be closed!"
    assert not os.path.exists(tf.name), "Temporary file must be deleted from disk!"
    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert not (after_csv - before_csv), "No orphan temporary CSV files should remain!"


def test_temp_file_flush_exception_cleans_up_temp_file(client):
    """
    Simulates a failure during temp_file.flush().
    Asserts that the file handle is closed and the file is unlinked without orphan files.
    """
    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def faulty_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)

        def mock_flush():
            raise OSError("I/O error during flush")

        tf.flush = mock_flush
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=faulty_temp_file):
        with pytest.raises(OSError, match="I/O error during flush"):
            client.post(
                "/feedback/import",
                files={"file": ("flush_fail.csv", io.BytesIO(b"text\nHello"), "text/csv")}
            )

    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.closed, "File handle must be closed!"
    assert not os.path.exists(tf.name), "Temporary file must be deleted from disk!"
    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert not (after_csv - before_csv), "No orphan temporary CSV files should remain!"


def test_handle_closed_strictly_before_deletion_attempt(client):
    """
    Verifies that on Windows, the temporary file handle is strictly closed before
    os.unlink() is attempted, avoiding WinError 32 sharing violations.
    """
    from starlette.datastructures import UploadFile

    captured_state_at_unlink = {}
    real_unlink = os.unlink
    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def spy_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)
        return tf

    def spy_unlink(path):
        for tf in created_temp_files:
            if tf.name == path:
                captured_state_at_unlink[path] = {
                    "closed_before_unlink": tf.closed
                }
        return real_unlink(path)

    with patch("tempfile.NamedTemporaryFile", side_effect=spy_temp_file):
        with patch("os.unlink", side_effect=spy_unlink):
            with patch.object(UploadFile, "read", side_effect=IOError("Simulated interruption")):
                with pytest.raises(IOError):
                    client.post(
                        "/feedback/import",
                        files={"file": ("order_check.csv", io.BytesIO(b"text\nTest"), "text/csv")}
                    )

    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.name in captured_state_at_unlink, "os.unlink should have been called on the temp file"
    assert captured_state_at_unlink[tf.name]["closed_before_unlink"] is True, (
        "temp_file handle MUST be closed before os.unlink is called!"
    )
    assert not os.path.exists(tf.name)


def test_upload_rejection_oversize_cleans_up_temp_file(client, monkeypatch):
    """
    Verifies that when streaming exceeds MAX_IMPORT_FILE_SIZE_BYTES during chunked read:
    - HTTP 413 is returned.
    - The temporary file handle is closed.
    - The temporary file is unlinked on disk without leaking.
    Also verifies fast boundary rejection (file.size > limit) leaves no temp files.
    """
    from starlette.datastructures import UploadFile

    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def spy_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)
        return tf

    # 1. Test streaming 413: simulate chunked transfer where file.size is None
    monkeypatch.setattr(settings, "MAX_IMPORT_FILE_SIZE_BYTES", 50)

    orig_upload_init = UploadFile.__init__

    def forced_none_size_init(self, *args, **kwargs):
        kwargs["size"] = None
        orig_upload_init(self, *args, **kwargs)

    oversize_payload = b"text,category\n" + (b"A" * 200)

    with patch.object(UploadFile, "__init__", forced_none_size_init):
        with patch("tempfile.NamedTemporaryFile", side_effect=spy_temp_file):
            res = client.post(
                "/feedback/import",
                files={"file": ("streaming_oversize.csv", io.BytesIO(oversize_payload), "text/csv")}
            )

    assert res.status_code == 413
    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.closed, "File handle must be closed on streaming 413 rejection!"
    assert not os.path.exists(tf.name), "Temporary file must be deleted on streaming 413 rejection!"

    # 2. Test fast-boundary 413: file.size is known before streaming
    res_fast = client.post(
        "/feedback/import",
        files={"file": ("fast_oversize.csv", io.BytesIO(b"text\n" + (b"B" * 200)), "text/csv")}
    )
    assert res_fast.status_code == 413

    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert not (after_csv - before_csv), "No orphan temporary CSV files should remain!"


def test_upload_empty_csv_cleans_up_temp_file(client):
    """
    Verifies that when an empty CSV is uploaded, HTTP 400 is returned,
    and the temporary file is unlinked without leaking.
    """
    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def spy_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=spy_temp_file):
        res = client.post(
            "/feedback/import",
            files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")}
        )

    assert res.status_code == 400
    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.closed, "File handle must be closed on empty file rejection!"
    assert not os.path.exists(tf.name), "Temporary file must be deleted on empty file rejection!"
    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert not (after_csv - before_csv), "No orphan temporary CSV files should remain!"


def test_upload_invalid_header_cleans_up_temp_file(client):
    """
    Verifies that when a CSV has invalid headers (e.g. missing text column),
    a FAILED ImportJob is returned (HTTP 202), ownership is NOT transferred to background tasks,
    and the temporary file is closed and deleted.
    """
    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def spy_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=spy_temp_file):
        res = client.post(
            "/feedback/import",
            files={"file": ("invalid_hdr.csv", io.BytesIO(b"wrong_col,category\nfoo,bar\n"), "text/csv")}
        )

    assert res.status_code == 202
    data = res.json()
    assert data["status"] == "FAILED"
    assert len(created_temp_files) == 1
    tf = created_temp_files[0]
    assert tf.closed, "File handle must be closed on header failure!"
    assert not os.path.exists(tf.name), "Temporary file must be unlinked on header failure!"
    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    assert not (after_csv - before_csv), "No orphan temporary CSV files should remain!"


def test_background_task_exception_cleans_up_temp_file():
    """
    Verifies that if the background worker encounters an unhandled exception during processing,
    the job is transitioned to FAILED and the temporary file is unlinked in finally:.
    """
    from db.session import SessionLocal
    from db.models import ImportJob

    db = SessionLocal()
    try:
        job = ImportService.create_import_job(db, "bg_error.csv")
        job_id = job.id
    finally:
        db.close()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_path = temp_file.name
    temp_file.write(b"text,category\nGreat service,General\n")
    temp_file.flush()
    temp_file.close()

    assert os.path.exists(temp_path)

    with patch.object(ImportService, "process_csv_file", side_effect=RuntimeError("Worker crash")):
        ImportService.process_csv_file_background(job_id=job_id, file_path=temp_path)

    assert not os.path.exists(temp_path), "Background task must clean up temp file even on error!"

    verify_db = SessionLocal()
    try:
        failed_job = verify_db.query(ImportJob).filter(ImportJob.id == job_id).first()
        assert failed_job.status == "FAILED"
        assert "Worker crash" in failed_job.error_summary
    finally:
        verify_db.close()


def test_successful_upload_transfers_ownership_and_cleans_up(client):
    """
    Verifies that on successful upload, ownership is transferred:
    - Route returns HTTP 202 with PENDING job.
    - Route does NOT delete the temporary file before background task runs.
    - Background task processes the file and unlinks it when finished.
    """
    created_temp_files = []
    real_named_temp = tempfile.NamedTemporaryFile

    def spy_temp_file(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)
        created_temp_files.append(tf)
        return tf

    file_existed_at_worker_start = []
    orig_bg_worker = ImportService.process_csv_file_background

    def spy_bg_worker(job_id, file_path):
        file_existed_at_worker_start.append(os.path.exists(file_path))
        return orig_bg_worker(job_id, file_path)

    with patch("tempfile.NamedTemporaryFile", side_effect=spy_temp_file):
        with patch.object(ImportService, "process_csv_file_background", side_effect=spy_bg_worker):
            res = client.post(
                "/feedback/import",
                files={"file": ("valid_transfer.csv", io.BytesIO(b"text\nExcellent app!\n"), "text/csv")}
            )

    assert res.status_code == 202
    assert len(created_temp_files) == 1
    tf = created_temp_files[0]

    assert len(file_existed_at_worker_start) == 1
    assert file_existed_at_worker_start[0] is True, "File must exist when background worker begins!"
    assert not os.path.exists(tf.name), "Worker must unlink file after processing!"


def test_no_orphan_temp_files_across_all_failure_paths(client):
    """
    Runs consecutive requests triggering various failure scenarios in sequence:
    1. Non-CSV extension
    2. Empty file
    3. Invalid CSV header
    4. Oversized upload (413)
    5. I/O read error
    6. Write failure
    7. Flush failure
    Asserts that across all 7 scenarios, exactly 0 orphaned CSV files remain in the temp directory.
    """
    from starlette.datastructures import UploadFile

    temp_dir = tempfile.gettempdir()
    before_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))

    # 1. Non-CSV extension
    client.post("/feedback/import", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")})

    # 2. Empty file
    client.post("/feedback/import", files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")})

    # 3. Invalid header
    client.post("/feedback/import", files={"file": ("bad_hdr.csv", io.BytesIO(b"bad_column\nval\n"), "text/csv")})

    # 4. Oversized upload
    oversize = b"text\n" + (b"A" * (settings.MAX_IMPORT_FILE_SIZE_BYTES + 512))
    client.post("/feedback/import", files={"file": ("large.csv", io.BytesIO(oversize), "text/csv")})

    # 5. I/O read error
    with patch.object(UploadFile, "read", side_effect=IOError("Read failure")):
        with pytest.raises(IOError):
            client.post("/feedback/import", files={"file": ("read_err.csv", io.BytesIO(b"text\nhi"), "text/csv")})

    # 6. Write failure
    real_named_temp = tempfile.NamedTemporaryFile

    def write_fail_temp(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)

        def bad_write(chunk):
            raise OSError("Write failed")

        tf.write = bad_write
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=write_fail_temp):
        with pytest.raises(OSError):
            client.post("/feedback/import", files={"file": ("write_err.csv", io.BytesIO(b"text\nhi"), "text/csv")})

    # 7. Flush failure
    def flush_fail_temp(*args, **kwargs):
        tf = real_named_temp(*args, **kwargs)

        def bad_flush():
            raise OSError("Flush failed")

        tf.flush = bad_flush
        return tf

    with patch("tempfile.NamedTemporaryFile", side_effect=flush_fail_temp):
        with pytest.raises(OSError):
            client.post("/feedback/import", files={"file": ("flush_err.csv", io.BytesIO(b"text\nhi"), "text/csv")})

    after_csv = set(f for f in os.listdir(temp_dir) if f.endswith(".csv"))
    orphan_files = after_csv - before_csv
    assert len(orphan_files) == 0, f"Detected orphaned temporary CSV files: {orphan_files}"


def test_cleanup_temp_file_helper_robustness():
    """
    Direct unit test for _cleanup_temp_file:
    1. Handles None temp_file and None temp_path safely without error.
    2. Handles already-closed temp_file without error.
    3. Handles non-existent path without error.
    4. Closes open file handle and unlinks existing file.
    5. Logs warning if closing raises error, but still attempts unlink.
    """
    from api.routes.imports import _cleanup_temp_file

    # 1. None inputs
    _cleanup_temp_file(None, None)

    # 2. Already closed
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tf_path = tf.name
    tf.close()
    assert os.path.exists(tf_path)
    _cleanup_temp_file(tf, tf_path)
    assert not os.path.exists(tf_path)

    # 3. Non-existent path
    _cleanup_temp_file(None, "C:/path/does/not/exist/foo.csv")

    # 4. Open file
    tf2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tf2_path = tf2.name
    assert not tf2.closed
    _cleanup_temp_file(tf2, tf2_path)
    assert tf2.closed
    assert not os.path.exists(tf2_path)

    # 5. Close raises error, unlink still succeeds
    mock_file = MagicMock()
    mock_file.closed = False
    mock_file.close.side_effect = RuntimeError("Close error")
    tf3 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tf3_path = tf3.name
    tf3.close()
    _cleanup_temp_file(mock_file, tf3_path)
    assert not os.path.exists(tf3_path)


# ============================================================================
# COMPREHENSIVE CSV INGESTION REGRESSION SUITE (PHASE 11)
# ============================================================================


def test_csv_header_aliases_and_case_insensitivity(client, monkeypatch):
    """
    Verifies all primary and extended header aliases with various casing:
    - text, review, feedback
    - TEXT, Review, FEEDBACK
    - review_text, Review Text, feedback_text
    - surrounding whitespace '  review  '
    """
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "positive",
            "confidence": 0.95,
            "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02}
        }
    )

    alias_headers = [
        "text,category,rating\n\"Valid text column\",UX,5\n",
        "review,category,rating\n\"Valid review column\",UX,5\n",
        "feedback,category,rating\n\"Valid feedback column\",UX,5\n",
        "TEXT,CATEGORY,RATING\n\"Valid uppercase TEXT\",UX,5\n",
        "Review,Category,Rating\n\"Valid capitalized Review\",UX,5\n",
        "  review  ,  category  ,  rating  \n\"Valid review with whitespace\",UX,5\n",
        "review_text,category,rating\n\"Valid review_text alias\",UX,5\n",
        "Review Text,Category,Rating\n\"Valid spaced Review Text alias\",UX,5\n",
        "feedback_text,category,rating\n\"Valid feedback_text alias\",UX,5\n",
    ]

    for idx, csv_data in enumerate(alias_headers):
        res = client.post(
            "/feedback/import",
            files={"file": (f"alias_test_{idx}.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
        )
        assert res.status_code == 202, f"Failed for header: {csv_data.splitlines()[0]}"
        job = res.json()
        assert job["status"] in ("PENDING", "PROCESSING", "COMPLETED")

        # Verify job completes successfully
        poll = client.get(f"/feedback/import/{job['id']}").json()
        assert poll["status"] == "COMPLETED", f"Job failed for header {csv_data.splitlines()[0]}: {poll.get('error_summary')}"
        assert poll["successful_rows"] == 1


def test_csv_delimiters_semicolon_tab_pipe(client, monkeypatch):
    """
    Verifies supported non-comma delimiters:
    1. Semicolon ';' (European standard)
    2. Tab '\\t' (TSV standard)
    3. Pipe '|' (Pipe-delimited standard)
    """
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "neutral",
            "confidence": 0.88,
            "scores": {"positive": 0.05, "neutral": 0.88, "negative": 0.07}
        }
    )

    # 1. Semicolon
    semi_csv = "text;category;rating\n\"European semicolon separated review\";Support;4\n"
    res_semi = client.post(
        "/feedback/import",
        files={"file": ("semicolon.csv", io.BytesIO(semi_csv.encode("utf-8")), "text/csv")}
    )
    assert res_semi.status_code == 202
    job_semi = client.get(f"/feedback/import/{res_semi.json()['id']}").json()
    assert job_semi["status"] == "COMPLETED"
    assert job_semi["successful_rows"] == 1

    # 2. Tab
    tab_csv = "review\tcategory\trating\nTab separated customer review\tUX\t5\n"
    res_tab = client.post(
        "/feedback/import",
        files={"file": ("tab.csv", io.BytesIO(tab_csv.encode("utf-8")), "text/csv")}
    )
    assert res_tab.status_code == 202
    job_tab = client.get(f"/feedback/import/{res_tab.json()['id']}").json()
    assert job_tab["status"] == "COMPLETED"
    assert job_tab["successful_rows"] == 1

    # 3. Pipe
    pipe_csv = "feedback|category|rating\nPipe separated customer feedback|General|3\n"
    res_pipe = client.post(
        "/feedback/import",
        files={"file": ("pipe.csv", io.BytesIO(pipe_csv.encode("utf-8")), "text/csv")}
    )
    assert res_pipe.status_code == 202
    job_pipe = client.get(f"/feedback/import/{res_pipe.json()['id']}").json()
    assert job_pipe["status"] == "COMPLETED"
    assert job_pipe["successful_rows"] == 1


def test_csv_quoted_commas_and_multiline_fields(client, monkeypatch):
    """
    Verifies RFC 4180 parsing for:
    - Feedback containing commas inside quoted strings
    - Feedback containing newline characters inside quoted strings
    - Multiline rows correctly counted as 1 record rather than multiple physical lines
    """
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "positive",
            "confidence": 0.99,
            "scores": {"positive": 0.99, "neutral": 0.01, "negative": 0.00}
        }
    )

    multiline_csv = (
        'text,category,rating\n'
        '"Great product, highly recommended, will buy again!",E-Commerce,5\n'
        '"First paragraph of long review.\nSecond paragraph with detailed feedback.\nConclusion: 5 stars.",Support,5\n'
    )
    res = client.post(
        "/feedback/import",
        files={"file": ("multiline.csv", io.BytesIO(multiline_csv.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = client.get(f"/feedback/import/{res.json()['id']}").json()
    assert job["status"] == "COMPLETED"
    assert job["total_rows"] == 2
    assert job["successful_rows"] == 2


def test_csv_empty_rows_and_duplicate_headers(client, monkeypatch):
    """
    Verifies that:
    1. Empty rows at start, middle, and end of CSV are cleanly ignored
    2. Empty header cells are ignored
    3. Duplicate headers take first occurrence without crashing
    """
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "neutral",
            "confidence": 0.85,
            "scores": {"positive": 0.1, "neutral": 0.85, "negative": 0.05}
        }
    )

    messy_csv = (
        "\n"
        "text,,category,text,rating\n"
        "\n"
        '"First review",,UX,"duplicate text cell",5\n'
        "   \n"
        '"Second review",,Bug,"another dup",2\n'
        "\n\n"
    )
    res = client.post(
        "/feedback/import",
        files={"file": ("messy.csv", io.BytesIO(messy_csv.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = client.get(f"/feedback/import/{res.json()['id']}").json()
    assert job["status"] == "COMPLETED"
    assert job["total_rows"] == 2
    assert job["successful_rows"] == 2


def test_unsupported_utf16_encoding_rejected(client):
    """
    Verifies that UTF-16 encoded CSV files are rejected with the documented
    actionable encoding error message.
    """
    utf16_content = "text,category,rating\n\"UTF-16 test\",Test,5\n".encode("utf-16")
    res = client.post(
        "/feedback/import",
        files={"file": ("utf16_file.csv", io.BytesIO(utf16_content), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    assert job["status"] == "FAILED"
    assert "CSV encoding is not supported" in job["error_summary"]
    assert "UTF-8" in job["error_summary"]


def test_no_records_persisted_on_header_validation_failure(client):
    """
    Verifies that when a CSV fails header pre-validation, no Feedback records
    are inserted or persisted in the database.
    """
    # Get current feedback count
    init_res = client.get("/feedback")
    init_count = init_res.json()["total"]

    bad_csv = "user,score,timestamp\nalice,5,2026-01-01\n"
    res = client.post(
        "/feedback/import",
        files={"file": ("missing_col.csv", io.BytesIO(bad_csv.encode("utf-8")), "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    assert job["status"] == "FAILED"
    assert job["total_rows"] == 0
    assert job["successful_rows"] == 0

    # Verify count did not increase
    after_res = client.get("/feedback")
    assert after_res.json()["total"] == init_count


def test_file_between_5mb_and_100mb_accepted(client, monkeypatch):
    """
    Verifies that a CSV file larger than the old 5 MB limit (e.g. 10 MB)
    is accepted and processed successfully under the 100 MB limit.
    """
    monkeypatch.setattr(
        "services.import_service.sentiment_service.predict",
        lambda text: {
            "text": text[:50],
            "sentiment": "positive",
            "confidence": 0.98,
            "scores": {"positive": 0.98, "neutral": 0.01, "negative": 0.01}
        }
    )

    # 10 MB streaming CSV file
    stream = FastCSVStream(10 * 1024 * 1024, num_rows=100)
    res = client.post(
        "/feedback/import",
        files={"file": ("ten_mb.csv", stream, "text/csv")}
    )
    assert res.status_code == 202
    job = res.json()
    assert job["status"] in ("PENDING", "PROCESSING", "COMPLETED")

    status_res = client.get(f"/feedback/import/{job['id']}").json()
    assert status_res["status"] == "COMPLETED"
    assert status_res["successful_rows"] == 100



# ---------------------------------------------------------------------------
# SentimentText dataset compatibility tests
# ---------------------------------------------------------------------------

def _mock_predict(text):
    """Deterministic sentiment mock for SentimentText tests."""
    return {
        "text": text[:50],
        "sentiment": "positive" if "great" in text.lower() or "excellent" in text.lower() or "enjoyed" in text.lower() else "negative",
        "confidence": 0.95,
        "scores": {"positive": 0.90, "neutral": 0.05, "negative": 0.05},
    }


class TestSentimentTextDatasetCompatibility:
    """Tests for ItemID,Sentiment,SentimentText dataset support."""

    def test_sentimenttext_header_recognized(self, client, monkeypatch):
        """SentimentText (camelCase) is accepted as the text column."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "SentimentText\nGreat product!\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("st.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        assert res.status_code == 202
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_sentimenttext_lowercase(self, client, monkeypatch):
        """sentimenttext (all lowercase) resolves to the text column."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "sentimenttext\nGreat service.\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("st2.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_sentimenttext_uppercase(self, client, monkeypatch):
        """SENTIMENTTEXT (all caps) resolves to the text column."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "SENTIMENTTEXT\nExcellent!\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("st3.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_sentiment_text_with_underscore(self, client, monkeypatch):
        """Sentiment_Text (underscore variant) resolves to the text column."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "Sentiment_Text\nDecent experience.\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("st4.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_sentiment_text_with_space(self, client, monkeypatch):
        """'Sentiment Text' (space-separated) resolves via slug normalization."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "Sentiment Text\nPretty good.\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("st5.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_sentiment_column_alone_not_treated_as_text(self, client):
        """A CSV with only 'Sentiment' (no text column) must fail header validation."""
        csv_data = "Sentiment,ItemID\n0,1\n1,2\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("no_text.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        assert res.status_code == 202
        job = res.json()
        assert job["status"] == "FAILED"
        assert "text" in job["error_summary"].lower() or "column" in job["error_summary"].lower()

    def test_full_itemid_sentiment_sentimenttext_dataset(self, client, monkeypatch):
        """
        End-to-end integration test with the exact ItemID,Sentiment,SentimentText
        dataset format. Verifies:
        - SentimentText values are extracted as feedback text
        - ItemID and Sentiment columns are safely ignored
        - All 3 rows are imported successfully
        """
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = (
            'ItemID,Sentiment,SentimentText\n'
            '1,0,"This product is disappointing."\n'
            '2,1,"Excellent experience and great service."\n'
            '3,1,"I really enjoyed using this application."\n'
        )
        res = client.post(
            "/feedback/import",
            files={"file": ("dataset.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        assert res.status_code == 202
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["total_rows"] == 3
        assert status["successful_rows"] == 3
        assert status["failed_rows"] == 0

    def test_duplicate_itemid_values_imported_independently(self, client, monkeypatch):
        """Duplicate ItemID values do not cause conflicts (ItemID is ignored)."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = (
            "ItemID,SentimentText\n"
            "1,First feedback.\n"
            "1,Second feedback with same ID.\n"
        )
        res = client.post(
            "/feedback/import",
            files={"file": ("dup_id.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 2

    def test_existing_text_format_still_works(self, client, monkeypatch):
        """Standard text,category,source,rating format continues to work."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "text,category,source,rating\nGreat stuff!,product,web,5\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("std.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_existing_review_format_still_works(self, client, monkeypatch):
        """Standard review header continues to work."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "review\nExcellent product!\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("rev.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1

    def test_existing_feedback_format_still_works(self, client, monkeypatch):
        """Standard feedback header continues to work."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        csv_data = "feedback\nI enjoyed the experience.\n"
        res = client.post(
            "/feedback/import",
            files={"file": ("fb.csv", io.BytesIO(csv_data.encode()), "text/csv")},
        )
        job = res.json()
        status = client.get(f"/feedback/import/{job['id']}").json()
        assert status["status"] == "COMPLETED"
        assert status["successful_rows"] == 1



# ---------------------------------------------------------------------------
# Advanced CSV Row Count Validation & Large Dataset Tests
# ---------------------------------------------------------------------------

class TestRowCountValidationAndLargeDatasets:
    """Tests for exact row count boundaries, large datasets, and accurate statistics."""

    def test_4999_rows_accepted(self, client, monkeypatch):
        """Verifies that 4,999 data records are accepted."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        header = "text,category,source,rating\n"
        data_rows = "".join(f'"Feedback row {i}",Product,Web,5\n' for i in range(4999))
        csv_bytes = (header + data_rows).encode("utf-8")

        res = client.post(
            "/feedback/import",
            files={"file": ("exact_4999.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 4999
        assert status_res["successful_rows"] == 4999
        assert status_res["failed_rows"] == 0

    def test_exact_5000_rows_accepted(self, client, monkeypatch):
        """Verifies that exactly 5,000 data records are accepted."""
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)

        # 5,000 data rows
        header = "text,category,source,rating\n"
        data_rows = "".join(f'"Feedback row {i}",Product,Web,5\n' for i in range(5000))
        csv_bytes = (header + data_rows).encode("utf-8")

        res = client.post(
            "/feedback/import",
            files={"file": ("exact_5000.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 5000
        assert status_res["successful_rows"] == 5000
        assert status_res["failed_rows"] == 0

    def test_5001_rows_accepted(self, client, monkeypatch):
        """
        Verifies that 5,001 data rows (formerly the rejection boundary)
        are accepted cleanly under the 1,050,000 limit.
        """
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)
        header = "text,category,source,rating\n"
        data_rows = "".join(f'"Feedback row {i}",Product,Web,5\n' for i in range(5001))
        csv_bytes = (header + data_rows).encode("utf-8")

        res = client.post(
            "/feedback/import",
            files={"file": ("exact_5001.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 5001
        assert status_res["successful_rows"] == 5001
        assert status_res["failed_rows"] == 0

    def test_100000_rows_accepted_streaming(self, client, monkeypatch):
        """
        Verifies that 100,000 records stream through preflight counting,
        rewinding, and chunk processing cleanly.
        """
        def _mock_fast_chunk(db, job, chunk, start_row_idx, text_col_idx, category_idx, source_idx, rating_idx, errors):
            job.successful_rows += len(chunk)
            job.processed_rows += len(chunk)

        monkeypatch.setattr(ImportService, "_process_row_chunk", _mock_fast_chunk)

        class SyntheticCSVStream(io.RawIOBase):
            def __init__(self, num_rows):
                self.num_rows = num_rows
                self.header = b"text,category,source,rating\n"
                self.row_template = b'"Streaming feedback",Product,Web,5\n'
                self.header_sent = False
                self.rows_sent = 0
                self.buffer = b""

            def readable(self):
                return True

            def readinto(self, b):
                if not self.header_sent:
                    self.buffer += self.header
                    self.header_sent = True
                while len(self.buffer) < len(b) and self.rows_sent < self.num_rows:
                    batch = min(10000, self.num_rows - self.rows_sent)
                    self.buffer += self.row_template * batch
                    self.rows_sent += batch
                if not self.buffer:
                    return 0
                n = min(len(b), len(self.buffer))
                b[:n] = self.buffer[:n]
                self.buffer = self.buffer[n:]
                return n

        stream = SyntheticCSVStream(num_rows=100000)
        res = client.post(
            "/feedback/import",
            files={"file": ("stream_100k.csv", stream, "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 100000
        assert status_res["successful_rows"] == 100000
        assert status_res["failed_rows"] == 0

    def test_1000000_rows_accepted_streaming(self, client, monkeypatch):
        """
        Verifies that 1,000,000 records stream through preflight counting,
        rewinding, and chunk processing cleanly.
        """
        def _mock_fast_chunk(db, job, chunk, start_row_idx, text_col_idx, category_idx, source_idx, rating_idx, errors):
            job.successful_rows += len(chunk)
            job.processed_rows += len(chunk)

        monkeypatch.setattr(ImportService, "_process_row_chunk", _mock_fast_chunk)

        class SyntheticCSVStream(io.RawIOBase):
            def __init__(self, num_rows):
                self.num_rows = num_rows
                self.header = b"text,category,source,rating\n"
                self.row_template = b'"Streaming feedback",Product,Web,5\n'
                self.header_sent = False
                self.rows_sent = 0
                self.buffer = b""

            def readable(self):
                return True

            def readinto(self, b):
                if not self.header_sent:
                    self.buffer += self.header
                    self.header_sent = True
                while len(self.buffer) < len(b) and self.rows_sent < self.num_rows:
                    batch = min(20000, self.num_rows - self.rows_sent)
                    self.buffer += self.row_template * batch
                    self.rows_sent += batch
                if not self.buffer:
                    return 0
                n = min(len(b), len(self.buffer))
                b[:n] = self.buffer[:n]
                self.buffer = self.buffer[n:]
                return n

        stream = SyntheticCSVStream(num_rows=1000000)
        res = client.post(
            "/feedback/import",
            files={"file": ("stream_1m.csv", stream, "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 1000000
        assert status_res["successful_rows"] == 1000000
        assert status_res["failed_rows"] == 0

    def test_1048575_rows_accepted_safely(self, client, monkeypatch):
        """
        Verifies that a dataset with 1,048,575 rows (the production issue dataset)
        is now safely accepted and processed under the new 1,050,000 row capacity limit.
        """
        def _mock_fast_chunk(db, job, chunk, start_row_idx, text_col_idx, category_idx, source_idx, rating_idx, errors):
            job.successful_rows += len(chunk)
            job.processed_rows += len(chunk)

        monkeypatch.setattr(ImportService, "_process_row_chunk", _mock_fast_chunk)

        class MillionRowCSVStream(io.RawIOBase):
            def __init__(self, num_rows=1048575):
                self.num_rows = num_rows
                self.header = b"ItemID,Sentiment,SentimentText\n"
                self.row_template = b"1,0,Feedback\n"
                self.header_sent = False
                self.rows_sent = 0
                self.buffer = b""

            def readable(self):
                return True

            def readinto(self, b):
                if not self.header_sent:
                    self.buffer += self.header
                    self.header_sent = True

                while len(self.buffer) < len(b) and self.rows_sent < self.num_rows:
                    batch = min(20000, self.num_rows - self.rows_sent)
                    self.buffer += self.row_template * batch
                    self.rows_sent += batch

                if not self.buffer:
                    return 0

                n = min(len(b), len(self.buffer))
                b[:n] = self.buffer[:n]
                self.buffer = self.buffer[n:]
                return n

        stream = MillionRowCSVStream(num_rows=1048575)
        res = client.post(
            "/feedback/import",
            files={"file": ("million_rows.csv", stream, "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 1048575
        assert status_res["successful_rows"] == 1048575
        assert status_res["failed_rows"] == 0

    def test_exact_1050000_rows_accepted_boundary(self, client, monkeypatch):
        """
        BOUNDARY MAXIMUM TEST:
        Verifies that exactly 1,050,000 data records (boundary maximum)
        are accepted and processed successfully.
        """
        def _mock_fast_chunk(db, job, chunk, start_row_idx, text_col_idx, category_idx, source_idx, rating_idx, errors):
            job.successful_rows += len(chunk)
            job.processed_rows += len(chunk)

        monkeypatch.setattr(ImportService, "_process_row_chunk", _mock_fast_chunk)

        class BoundaryCSVStream(io.RawIOBase):
            def __init__(self, num_rows=1050000):
                self.num_rows = num_rows
                self.header = b"text,category,source,rating\n"
                self.row_template = b'"Boundary text",Product,Web,5\n'
                self.header_sent = False
                self.rows_sent = 0
                self.buffer = b""

            def readable(self):
                return True

            def readinto(self, b):
                if not self.header_sent:
                    self.buffer += self.header
                    self.header_sent = True

                while len(self.buffer) < len(b) and self.rows_sent < self.num_rows:
                    batch = min(20000, self.num_rows - self.rows_sent)
                    self.buffer += self.row_template * batch
                    self.rows_sent += batch

                if not self.buffer:
                    return 0

                n = min(len(b), len(self.buffer))
                b[:n] = self.buffer[:n]
                self.buffer = self.buffer[n:]
                return n

        stream = BoundaryCSVStream(num_rows=1050000)
        res = client.post(
            "/feedback/import",
            files={"file": ("exact_1050000.csv", stream, "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 1050000
        assert status_res["successful_rows"] == 1050000
        assert status_res["failed_rows"] == 0

    def test_1050001_rows_rejected_safely(self, client):
        """
        BOUNDARY EXCEEDED TEST (1,050,001 data rows = limit + 1):
        Verifies that a dataset with 1,050,001 data rows is rejected BEFORE ingestion,
        total_rows is stored accurately as 1,050,001, and zero feedback records are persisted.
        """
        from db.session import SessionLocal
        from db.models import Feedback

        db = SessionLocal()
        initial_feedback_count = db.query(Feedback).count()
        db.close()

        class ExceededCSVStream(io.RawIOBase):
            def __init__(self, num_rows=1050001):
                self.num_rows = num_rows
                self.header = b"text,category,source,rating\n"
                self.row_template = b'"Too many rows",Product,Web,5\n'
                self.header_sent = False
                self.rows_sent = 0
                self.buffer = b""

            def readable(self):
                return True

            def readinto(self, b):
                if not self.header_sent:
                    self.buffer += self.header
                    self.header_sent = True

                while len(self.buffer) < len(b) and self.rows_sent < self.num_rows:
                    batch = min(20000, self.num_rows - self.rows_sent)
                    self.buffer += self.row_template * batch
                    self.rows_sent += batch

                if not self.buffer:
                    return 0

                n = min(len(b), len(self.buffer))
                b[:n] = self.buffer[:n]
                self.buffer = self.buffer[n:]
                return n

        stream = ExceededCSVStream(num_rows=1050001)
        res = client.post(
            "/feedback/import",
            files={"file": ("too_many_1050001.csv", stream, "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "FAILED"
        assert status_res["total_rows"] == 1050001, f"Expected total_rows=1050001, got {status_res['total_rows']}"
        assert status_res["successful_rows"] == 0
        assert status_res["failed_rows"] == 0
        errors = json.loads(status_res["error_summary"])
        assert any("1050001 > 1050000" in e for e in errors)
        assert any("Maximum batch limit is 1,050,000 records" in e for e in errors)

        # Confirm zero records were persisted in DB
        db2 = SessionLocal()
        assert db2.query(Feedback).count() == initial_feedback_count
        db2.close()

    def test_logical_row_counting_with_quoted_multiline_feedback(self, client, monkeypatch):
        """
        Verifies that quoted multiline feedback fields are counted as single logical records,
        NOT multiple physical lines.
        """
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)

        # 3 logical rows, with row 1 containing 4 lines, row 2 containing 2 lines
        csv_content = (
            'text,category,source,rating\n'
            '"Line 1\nLine 2\nLine 3\nLine 4",UX,Web,5\n'
            '"Second record\nwith newline",Support,Email,4\n'
            '"Third record normal",Product,Web,5\n'
        )
        res = client.post(
            "/feedback/import",
            files={"file": ("multiline_logical.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 3
        assert status_res["successful_rows"] == 3
        assert status_res["failed_rows"] == 0

    def test_empty_rows_and_trailing_newlines_excluded(self, client, monkeypatch):
        """
        Verifies that empty rows, whitespace rows, and trailing newlines
        are excluded from total_rows and data processing.
        """
        monkeypatch.setattr("services.import_service.sentiment_service.predict", _mock_predict)

        csv_content = (
            "text,category,source,rating\n"
            "\n"
            "   \n"
            "Valid review 1,Product,Web,5\n"
            ",,,\n"
            "Valid review 2,Support,Email,4\n"
            "\n"
            "\n"
        )
        res = client.post(
            "/feedback/import",
            files={"file": ("sparse_empty_rows.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
        )
        assert res.status_code == 202
        job_init = res.json()

        status_res = client.get(f"/feedback/import/{job_init['id']}").json()
        assert status_res["status"] == "COMPLETED"
        assert status_res["total_rows"] == 2
        assert status_res["successful_rows"] == 2
