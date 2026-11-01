"""
Sentiment Pulse - Large-Scale CSV Import Performance & Memory Benchmark Test
Tests end-to-end processing of high-volume CSV datasets up to 1,050,000 records.
Verifies:
1. Flat, bounded memory consumption (Peak RAM < 25 MB)
2. Chunk-based database execution (DB_INSERT_CHUNK_SIZE)
3. Accurate total, processed, and successful row counting
4. Job status lifecycle transitions (PENDING -> PROCESSING -> COMPLETED)
5. Zero memory leaks across chunk iterations
"""

import io
import os
import time
import uuid
import tempfile
import tracemalloc
import pytest
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from core.config import settings
from db.session import SessionLocal
from db.models import ImportJob, Feedback
from services.import_service import ImportService


def _fast_predict(text: str):
    """Fast mock inference function for benchmarking storage and streaming pipelines."""
    return {
        "text": text[:50],
        "sentiment": "positive",
        "confidence": 0.96,
        "scores": {"positive": 0.96, "neutral": 0.02, "negative": 0.02}
    }


class GeneratorCSVFile:
    """
    Disk-buffered streaming CSV generator producing valid feedback rows
    without ever keeping the entire dataset in heap memory.
    """
    @staticmethod
    def create_temp_csv(num_rows: int) -> str:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8", newline="")
        writer = io.StringIO()
        writer.write("text,category,source,rating\n")
        tmp.write(writer.getvalue())

        batch_size = 10000
        rows_written = 0
        while rows_written < num_rows:
            take = min(batch_size, num_rows - rows_written)
            chunk_str = "".join(
                f'"Benchmark feedback message {rows_written + i}",Performance,Web,5\n'
                for i in range(take)
            )
            tmp.write(chunk_str)
            rows_written += take

        tmp.flush()
        tmp.close()
        return tmp.name


@pytest.mark.performance
def test_large_scale_csv_import_bounded_memory_and_throughput(monkeypatch):
    """
    PERFORMANCE & MEMORY BENCHMARK:
    Processes 50,000 real persisted records through the full ImportService pipeline.
    Measures memory usage with tracemalloc to guarantee memory remains strictly bounded.
    """
    monkeypatch.setattr("services.import_service.sentiment_service.predict", _fast_predict)

    num_records = 50000
    temp_path = GeneratorCSVFile.create_temp_csv(num_records)
    db = SessionLocal()

    try:
        initial_count = db.query(Feedback).count()
        job_id = str(uuid.uuid4())
        job = ImportJob(
            id=job_id,
            filename="large_scale_benchmark.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc)
        )
        db.add(job)
        db.commit()

        # Start tracking heap memory
        tracemalloc.start()
        t0 = time.perf_counter()

        # Execute ingestion pipeline
        completed_job = ImportService.process_csv_file(db, job, temp_path)

        duration = time.perf_counter() - t0
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        throughput = num_records / duration if duration > 0 else 0

        print(f"\n--- Ingestion Performance Benchmark ---")
        print(f"Records Processed: {num_records:,}")
        print(f"Elapsed Time:      {duration:.2f} seconds")
        print(f"Throughput:        {throughput:,.1f} records/second")
        print(f"Peak Heap Memory:  {peak_mb:.2f} MB")
        print(f"----------------------------------------")

        # 1. Verify Job Status & Metrics
        assert completed_job.status == "COMPLETED"
        assert completed_job.total_rows == num_records
        assert completed_job.successful_rows == num_records
        assert completed_job.failed_rows == 0
        assert completed_job.completed_at is not None

        # 2. Verify Database Persistence
        final_count = db.query(Feedback).count()
        assert final_count == initial_count + num_records

        # 3. Verify Memory Remains Flat and Bounded (< 25 MB)
        assert peak_mb < 25.0, f"Memory leak detected: Peak RAM was {peak_mb:.2f} MB (expected < 25 MB)"

    finally:
        # Cleanup benchmark data to keep database clean
        try:
            db.query(Feedback).filter(Feedback.category == "Performance").delete()
            db.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@pytest.mark.performance
def test_synthetic_million_row_streaming_throughput(monkeypatch):
    """
    STREAMING SCALE BENCHMARK (1,050,000 RECORDS):
    Verifies that a CSV file containing the full 1,050,000 records
    passes the preflight streaming pass and chunk iterator cleanly.
    """
    chunk_call_count = [0]
    total_chunk_rows = [0]

    def _spy_chunk(db, job, chunk, start_row_idx, text_col_idx, category_idx, source_idx, rating_idx, errors):
        chunk_call_count[0] += 1
        total_chunk_rows[0] += len(chunk)
        job.successful_rows += len(chunk)
        job.processed_rows += len(chunk)

    monkeypatch.setattr(ImportService, "_process_row_chunk", _spy_chunk)

    target_rows = 1050000
    temp_path = GeneratorCSVFile.create_temp_csv(target_rows)
    db = SessionLocal()

    try:
        job_id = str(uuid.uuid4())
        job = ImportJob(
            id=job_id,
            filename="full_million_benchmark.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc)
        )
        db.add(job)
        db.commit()

        tracemalloc.start()
        t0 = time.perf_counter()

        completed_job = ImportService.process_csv_file(db, job, temp_path)

        duration = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        throughput = target_rows / duration if duration > 0 else 0

        print(f"\n--- 1,050,000 Records Streaming Benchmark ---")
        print(f"Total Rows:       {target_rows:,}")
        print(f"Streaming Time:   {duration:.2f} seconds")
        print(f"Throughput:       {throughput:,.1f} rows/second")
        print(f"Chunk Calls:      {chunk_call_count[0]:,}")
        print(f"Peak Heap Memory: {peak_mb:.2f} MB")
        print(f"---------------------------------------------")

        assert completed_job.status == "COMPLETED"
        assert completed_job.total_rows == target_rows
        assert completed_job.successful_rows == target_rows
        assert total_chunk_rows[0] == target_rows
        # Verify chunk size adhered to (2000 per chunk -> 525 chunks for 1,050,000 rows)
        expected_chunks = (target_rows + settings.DB_INSERT_CHUNK_SIZE - 1) // settings.DB_INSERT_CHUNK_SIZE
        assert chunk_call_count[0] == expected_chunks
        assert peak_mb < 20.0, f"Peak memory {peak_mb:.2f} MB exceeded 20 MB budget"

    finally:
        try:
            db.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@pytest.mark.performance
def test_real_db_persistence_chunking_and_rollback(monkeypatch):
    """
    REAL PERSISTENCE, CHUNKING, AND ROLLBACK TEST:
    Verifies that:
    1. Chunks of size DB_INSERT_CHUNK_SIZE (2,000) are flushed and expunged.
    2. Simulated database failure mid-import cleanly rolls back all partial records.
    3. No orphan records leak into the database.
    4. Subsequent retry succeeds without duplicate records.
    """
    monkeypatch.setattr("services.import_service.sentiment_service.predict", _fast_predict)

    num_records = 10000
    temp_path = GeneratorCSVFile.create_temp_csv(num_records)
    db = SessionLocal()
    job_id = str(uuid.uuid4())

    try:
        initial_count = db.query(Feedback).count()

        job = ImportJob(
            id=job_id,
            filename="rollback_test.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()

        # Step 1: Simulate DB flush failure on chunk 3
        real_flush = db.flush
        chunk_counter = [0]

        def _failing_flush(*args, **kwargs):
            chunk_counter[0] += 1
            if chunk_counter[0] >= 3:
                raise RuntimeError("Simulated Database I/O Error during chunk 3")
            return real_flush(*args, **kwargs)

        db.flush = _failing_flush

        try:
            ImportService.process_csv_file(db, job, temp_path)
        except RuntimeError:
            db.rollback()

        # Restore real flush
        db.flush = real_flush

        # Verify rollback: 0 orphan records should be committed
        count_after_failure = db.query(Feedback).count()
        assert count_after_failure == initial_count, f"Rollback leaked records: {count_after_failure} vs {initial_count}"

        # Step 2: Clean retry with same file to verify duplicate prevention and accurate persistence
        retry_job_id = str(uuid.uuid4())
        retry_job = ImportJob(
            id=retry_job_id,
            filename="retry_test.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc),
        )
        db.add(retry_job)
        db.commit()

        completed_job = ImportService.process_csv_file(db, retry_job, temp_path)
        final_count = db.query(Feedback).count()

        assert completed_job.status == "COMPLETED"
        assert completed_job.successful_rows == num_records
        assert final_count == initial_count + num_records

        # Verify session identity map is kept bounded (expunged Feedback records)
        feedback_in_identity_map = [
            obj for obj in db.identity_map.values() if isinstance(obj, Feedback)
        ]
        assert len(feedback_in_identity_map) == 0, "Feedback records not expunged from identity map!"

    finally:
        try:
            db.query(Feedback).filter(Feedback.category == "Performance").delete()
            db.query(ImportJob).filter(ImportJob.id.in_([job_id, retry_job_id])).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@pytest.mark.performance
def test_million_row_real_database_persistence(monkeypatch):
    """
    OPT-IN FULL 1,050,000 REAL DATABASE PERSISTENCE BENCHMARK:
    Persists the complete 1,050,000 records into the configured database (PostgreSQL or SQLite).
    Enabled via RUN_FULL_MILLION_DB_PERSISTENCE=1.
    """
    if os.getenv("RUN_FULL_MILLION_DB_PERSISTENCE", "0") != "1":
        pytest.skip(
            "Full 1,050,000 real DB persistence test skipped by default to keep unit tests fast. "
            "Run with RUN_FULL_MILLION_DB_PERSISTENCE=1 or scripts/benchmark_db_persistence.py --rows 1050000."
        )

    monkeypatch.setattr("services.import_service.sentiment_service.predict", _fast_predict)

    num_records = 1050000
    temp_path = GeneratorCSVFile.create_temp_csv(num_records)
    db = SessionLocal()
    job_id = str(uuid.uuid4())

    try:
        initial_count = db.query(Feedback).count()
        job = ImportJob(
            id=job_id,
            filename="full_million_real_persistence.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()

        tracemalloc.start()
        t0 = time.perf_counter()

        completed_job = ImportService.process_csv_file(db, job, temp_path)

        duration = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        throughput = num_records / duration if duration > 0 else 0

        final_count = db.query(Feedback).count()
        assert completed_job.status == "COMPLETED"
        assert completed_job.successful_rows == num_records
        assert final_count == initial_count + num_records
        assert peak_mb < 25.0

    finally:
        try:
            db.query(Feedback).filter(Feedback.category == "Performance").delete()
            db.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()

        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

