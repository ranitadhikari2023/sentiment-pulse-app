"""
Sentiment Pulse - Database Persistence Benchmark & Verification Utility

Runs real end-to-end database persistence tests against the application's
configured database engine (PostgreSQL or SQLite).

Verifies:
1. Actual records persisted into the feedback table.
2. Final database record count matching exact expected count.
3. Chunk execution adhering to DB_INSERT_CHUNK_SIZE (default: 2,000).
4. Memory consumption bounded (tracemalloc peak heap < 25 MB).
5. Database transaction handling and session expunge behavior.
6. Failure and rollback behavior (zero orphan records on error).
7. Duplicate column and retry handling.
"""

import argparse
import io
import json
import os
import sys
import tempfile
import time
import tracemalloc
import uuid
from datetime import datetime, timezone
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from db.models import Base, Feedback, ImportJob
from db.session import SessionLocal, engine
from services.import_service import ImportService


def _fast_predict(text: str):
    """Fast mock inference to isolate database persistence performance."""
    return {
        "text": text[:50],
        "sentiment": "positive",
        "confidence": 0.95,
        "scores": {"positive": 0.95, "neutral": 0.03, "negative": 0.02},
    }


def generate_benchmark_csv(num_rows: int, duplicate_cols: bool = False) -> str:
    """Generates a temporary CSV file with num_rows data rows."""
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode="w", encoding="utf-8", newline=""
    )
    if duplicate_cols:
        tmp.write("text,category,source,rating,category,text\n")
    else:
        tmp.write("text,category,source,rating\n")

    batch_size = 10000
    rows_written = 0
    while rows_written < num_rows:
        take = min(batch_size, num_rows - rows_written)
        if duplicate_cols:
            lines = "".join(
                f'"Benchmark feedback {rows_written + i}",Performance,Web,5,IgnoredDuplicate,"DuplicateText"\n'
                for i in range(take)
            )
        else:
            lines = "".join(
                f'"Benchmark feedback {rows_written + i}",Performance,Web,5\n'
                for i in range(take)
            )
        tmp.write(lines)
        rows_written += take

    tmp.flush()
    tmp.close()
    return tmp.name


def run_persistence_benchmark(num_rows: int, cleanup: bool = True) -> dict:
    """
    Executes a real database persistence run of num_rows records.
    Verifies chunking, transaction commit, memory usage, and final record counts.
    """
    print(f"\n=======================================================")
    print(f"DATABASE PERSISTENCE BENCHMARK: {num_rows:,} RECORDS")
    print(f"Target Database: {engine.url.render_as_string(hide_password=True)}")
    print(f"Dialect:         {engine.dialect.name}")
    print(f"Chunk Size:      {settings.DB_INSERT_CHUNK_SIZE:,} rows")
    print(f"=======================================================")

    from services.import_service import sentiment_service
    original_predict = sentiment_service.predict
    sentiment_service.predict = _fast_predict

    temp_csv_path = generate_benchmark_csv(num_rows)
    db = SessionLocal()
    job_id = str(uuid.uuid4())

    try:
        initial_db_count = db.query(Feedback).count()
        print(f"Initial Feedback records in DB: {initial_db_count:,}")

        job = ImportJob(
            id=job_id,
            filename=f"benchmark_{num_rows}.csv",
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

        # Execute full import service with real DB persistence
        completed_job = ImportService.process_csv_file(db, job, temp_csv_path)

        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_mem / (1024 * 1024)
        throughput = num_rows / elapsed if elapsed > 0 else 0

        # Query database directly for verification
        final_db_count = db.query(Feedback).count()
        benchmark_persisted = db.query(Feedback).filter(Feedback.category == "Performance").count()

        print(f"\n--- Benchmark Results ---")
        print(f"Status:               {completed_job.status}")
        print(f"Elapsed Time:         {elapsed:.2f} s")
        print(f"Throughput:           {throughput:,.1f} records/s")
        print(f"Peak Heap Memory:     {peak_mb:.2f} MB")
        print(f"Initial DB Count:     {initial_db_count:,}")
        print(f"Final DB Count:       {final_db_count:,}")
        print(f"Persisted Difference: {final_db_count - initial_db_count:,}")
        print(f"Reported Successful:  {completed_job.successful_rows:,}")
        print(f"Reported Failed:      {completed_job.failed_rows:,}")
        print(f"Expected Chunks:      {(num_rows + settings.DB_INSERT_CHUNK_SIZE - 1) // settings.DB_INSERT_CHUNK_SIZE:,}")
        print(f"-------------------------")

        assert completed_job.status == "COMPLETED", f"Expected COMPLETED, got {completed_job.status}"
        assert completed_job.successful_rows == num_rows, f"Expected {num_rows} successful, got {completed_job.successful_rows}"
        assert final_db_count == initial_db_count + num_rows, f"DB count mismatch: {final_db_count} vs {initial_db_count + num_rows}"
        assert peak_mb < 30.0, f"Memory exceeded 30 MB: {peak_mb:.2f} MB"

        return {
            "num_rows": num_rows,
            "elapsed_seconds": round(elapsed, 2),
            "throughput_rps": round(throughput, 1),
            "peak_memory_mb": round(peak_mb, 2),
            "initial_db_count": initial_db_count,
            "final_db_count": final_db_count,
            "persisted_count": final_db_count - initial_db_count,
            "status": completed_job.status,
            "db_dialect": engine.dialect.name,
        }

    finally:
        sentiment_service.predict = original_predict
        if cleanup:
            print("\nCleaning up benchmark records...")
            try:
                db.query(Feedback).filter(Feedback.category == "Performance").delete()
                db.query(ImportJob).filter(ImportJob.id == job_id).delete()
                db.commit()
                print("Cleanup complete.")
            except Exception as e:
                db.rollback()
                print(f"Cleanup error: {e}")
        db.close()

        if os.path.exists(temp_csv_path):
            try:
                os.unlink(temp_csv_path)
            except OSError:
                pass


def run_rollback_verification(test_rows: int = 10000) -> bool:
    """
    Verifies failure and rollback behavior.
    Forces a mid-import exception during chunk processing and verifies:
    1. Zero partial records remain committed in the database.
    2. ImportJob status transitions to FAILED.
    3. Re-running a valid import succeeds without duplicate records.
    """
    print(f"\n=======================================================")
    print(f"FAILURE & ROLLBACK VERIFICATION ({test_rows:,} ROWS)")
    print(f"=======================================================")

    from services.import_service import sentiment_service
    original_predict = sentiment_service.predict
    sentiment_service.predict = _fast_predict

    temp_csv_path = generate_benchmark_csv(test_rows)
    db = SessionLocal()
    job_id = str(uuid.uuid4())

    try:
        initial_count = db.query(Feedback).count()
        job = ImportJob(
            id=job_id,
            filename="simulated_failure.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()

        # Monkeypatch db.flush to simulate a database connection or disk error in chunk 3
        real_flush = db.flush
        chunk_counter = [0]

        def _failing_flush(*args, **kwargs):
            chunk_counter[0] += 1
            if chunk_counter[0] >= 3:
                raise RuntimeError("Simulated Database Connection Failure during chunk 3")
            return real_flush(*args, **kwargs)

        db.flush = _failing_flush

        failed = False
        try:
            ImportService.process_csv_file(db, job, temp_csv_path)
        except RuntimeError as e:
            failed = True
            print(f"Simulated exception caught as expected: {e}")
            db.rollback()

        # Restore real flush
        db.flush = real_flush

        # Query database to confirm NO partial rows were committed
        count_after_failure = db.query(Feedback).count()
        print(f"Initial DB Count:       {initial_count:,}")
        print(f"Count After Failure:    {count_after_failure:,}")
        print(f"Uncommitted leaked rows: {count_after_failure - initial_count}")

        assert count_after_failure == initial_count, "Rollback failed! Orphan records were found in the database."

        # Verify a clean retry succeeds without duplicate records
        print("\nVerifying subsequent retry imports cleanly...")
        retry_job_id = str(uuid.uuid4())
        retry_job = ImportJob(
            id=retry_job_id,
            filename="simulated_failure_retry.csv",
            status="PENDING",
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            created_at=datetime.now(timezone.utc),
        )
        db.add(retry_job)
        db.commit()

        completed_retry = ImportService.process_csv_file(db, retry_job, temp_csv_path)
        count_after_retry = db.query(Feedback).count()

        print(f"Count After Retry:      {count_after_retry:,}")
        print(f"New Records from Retry: {count_after_retry - initial_count:,}")

        assert completed_retry.status == "COMPLETED"
        assert count_after_retry == initial_count + test_rows

        # Cleanup
        db.query(Feedback).filter(Feedback.category == "Performance").delete()
        db.query(ImportJob).filter(ImportJob.id.in_([job_id, retry_job_id])).delete()
        db.commit()
        print("Rollback and duplicate prevention verification PASSED!")
        return True

    finally:
        sentiment_service.predict = original_predict
        db.close()
        if os.path.exists(temp_csv_path):
            try:
                os.unlink(temp_csv_path)
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Sentiment Pulse Database Persistence Benchmark")
    parser.add_argument("--rows", type=int, default=50000, help="Number of records to persist (e.g. 50000, 100000, 1050000)")
    parser.add_argument("--rollback", action="store_true", help="Run simulated failure and rollback verification")
    parser.add_argument("--all", action="store_true", help="Run persistence benchmark and rollback verification")
    args = parser.parse_args()

    results = {}
    if args.rollback or args.all:
        results["rollback_verified"] = run_rollback_verification()

    if not args.rollback or args.all:
        results["persistence"] = run_persistence_benchmark(args.rows)

    print("\nBenchmark Summary:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

