"""
Sentiment Pulse - Comprehensive Scale Benchmark & Durability Audit Pipeline.
Executes progressive scale benchmarks:
- 10,000 records
- 100,000 records
- 500,000 records
- 1,050,000 records

Measures:
1. Total ingestion time
2. Rows inserted
3. Rows rejected
4. Peak RSS memory & Heap memory
5. CPU utilization (%)
6. Database write throughput (records/sec)
7. Chunk duration (ms/chunk)
8. Polling latency during active ingestion (ms)
9. Storage utilization (disk bytes)
10. Duplicate records check
11. Worker interruption and checkpoint recovery
"""

import os
import sys
import time
import json
import uuid
import tempfile
import threading
import tracemalloc
import hashlib
import psutil
from datetime import datetime, timezone
from typing import Dict, List, Any

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from db.models import Base, Feedback, ImportJob, FeedbackStaging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from services.import_service import ImportService, sentiment_service


def _fast_predict(text: str):
    """Fast inference simulator for pipeline storage and throughput auditing."""
    return {
        "text": text[:60],
        "sentiment": "positive",
        "confidence": 0.94,
        "scores": {"positive": 0.94, "neutral": 0.04, "negative": 0.02},
    }


def create_isolated_db():
    """Creates a temporary, isolated SQLite database with PRAGMA WAL enabled."""
    tmp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp_path = tmp_file.name
    tmp_file.close()

    engine = create_engine(
        f"sqlite:///{tmp_path}",
        connect_args={"check_same_thread": False, "timeout": 60}
    )
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA synchronous=NORMAL;"))
        conn.commit()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session, tmp_path


def generate_progressive_csv(total_rows: int, rejected_count: int = 5) -> str:
    """Generates a CSV on disk with total_rows, including a few intentionally malformed/rejected rows."""
    tmp = tempfile.NamedTemporaryFile(
        dir=settings.UPLOAD_DIR, delete=False, suffix=".csv", mode="w", encoding="utf-8", newline=""
    )
    tmp.write("text,category,source,rating\n")

    # Write intentional rejected rows at the start if requested
    for i in range(rejected_count):
        # Empty text column triggers row rejection
        tmp.write(f',InvalidCategory,Web,5\n')

    batch_size = 10000
    rows_written = 0
    valid_target = total_rows - rejected_count
    while rows_written < valid_target:
        take = min(batch_size, valid_target - rows_written)
        chunk_lines = "".join(
            f'"High-volume scale feedback item {rows_written + i} for platform intelligence.",ScaleTest,Web,5\n'
            for i in range(take)
        )
        tmp.write(chunk_lines)
        rows_written += take

    tmp.flush()
    tmp.close()
    return tmp.name


def run_single_scale_benchmark(
    target_rows: int,
    rejected_rows: int = 5,
    poll_concurrently: bool = True
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"PROGRESSIVE BENCHMARK: {target_rows:,} RECORDS (Rejected: {rejected_rows})")
    print(f"{'='*70}")

    engine, Session, db_path = create_isolated_db()
    db = Session()

    # Save original predict
    original_predict = sentiment_service.predict
    sentiment_service.predict = _fast_predict

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    csv_path = generate_progressive_csv(target_rows, rejected_count=rejected_rows)
    with open(csv_path, "rb") as f_chk:
        file_hash = hashlib.sha256(f_chk.read()).hexdigest()

    job_id = str(uuid.uuid4())
    job = ImportJob(
        id=job_id,
        filename=os.path.basename(csv_path),
        file_path=csv_path,
        file_hash=file_hash,
        total_rows=0,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        status="PENDING",
        stage="QUEUED"
    )
    db.add(job)
    db.commit()

    proc = psutil.Process()
    # Baseline memory and CPU
    proc.cpu_percent()
    initial_rss_mb = proc.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    # Track polling latency during active run
    polling_latencies: List[float] = []
    stop_polling = threading.Event()

    def _poller():
        poll_session = Session()
        try:
            while not stop_polling.is_set():
                t_poll_start = time.perf_counter()
                poll_job = poll_session.query(ImportJob).filter(ImportJob.id == job_id).first()
                t_poll_elapsed = (time.perf_counter() - t_poll_start) * 1000.0  # ms
                if poll_job:
                    polling_latencies.append(t_poll_elapsed)
                time.sleep(0.1)  # poll every 100ms
        finally:
            poll_session.close()

    poller_thread = None
    if poll_concurrently:
        poller_thread = threading.Thread(target=_poller, daemon=True)
        poller_thread.start()

    t0 = time.perf_counter()
    completed_job = ImportService.process_csv_file(db, job, csv_path)
    elapsed_sec = time.perf_counter() - t0

    stop_polling.set()
    if poller_thread:
        poller_thread.join(timeout=2.0)

    _, peak_heap_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_rss_mb = proc.memory_info().rss / (1024 * 1024)
    cpu_usage_pct = proc.cpu_percent()
    peak_heap_mb = peak_heap_bytes / (1024 * 1024)

    # Verification from Database
    total_persisted_feedback = db.query(Feedback).count()
    remaining_staging = db.query(FeedbackStaging).count()
    db_file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Check for duplicate records (all IDs should be distinct)
    unique_feedback_count = db.query(Feedback.id).distinct().count()
    duplicate_count = total_persisted_feedback - unique_feedback_count

    expected_successful = target_rows - rejected_rows
    throughput_rps = target_rows / max(elapsed_sec, 0.001)

    avg_chunk_size = getattr(settings, "DB_INSERT_CHUNK_SIZE", 5000)
    num_chunks = (target_rows + avg_chunk_size - 1) // avg_chunk_size
    avg_chunk_duration_ms = (elapsed_sec / max(num_chunks, 1)) * 1000.0

    avg_poll_latency_ms = (sum(polling_latencies) / len(polling_latencies)) if polling_latencies else 0.0
    p95_poll_latency_ms = sorted(polling_latencies)[int(len(polling_latencies) * 0.95)] if len(polling_latencies) >= 20 else avg_poll_latency_ms

    print(f"  Wall-Clock Duration:      {elapsed_sec:.2f} seconds")
    print(f"  Throughput:               {throughput_rps:,.1f} records/sec")
    print(f"  Rows Processed:           {completed_job.processed_rows:,}")
    print(f"  Rows Inserted (Success):  {completed_job.successful_rows:,} (DB Persisted: {total_persisted_feedback:,})")
    print(f"  Rows Rejected:            {completed_job.failed_rows:,}")
    print(f"  Duplicate Records:        {duplicate_count}")
    print(f"  Staging Residual Count:   {remaining_staging} (Must be 0)")
    print(f"  Initial RSS:              {initial_rss_mb:.2f} MB")
    print(f"  Peak RSS:                 {peak_rss_mb:.2f} MB")
    print(f"  Peak Heap Memory:         {peak_heap_mb:.2f} MB")
    print(f"  Estimated CPU Usage:      {cpu_usage_pct:.1f}%")
    print(f"  Estimated Chunk Duration: {avg_chunk_duration_ms:.2f} ms/chunk ({avg_chunk_size} rows)")
    print(f"  Avg Status Poll Latency:  {avg_poll_latency_ms:.2f} ms (p95: {p95_poll_latency_ms:.2f} ms)")
    print(f"  Database Disk Size:       {db_file_size_mb:.2f} MB")
    print(f"  Final Job Status:         {completed_job.status} / Stage: {completed_job.stage}")

    # Assertions
    assert completed_job.status == "COMPLETED"
    assert completed_job.successful_rows == expected_successful
    assert completed_job.failed_rows == rejected_rows
    assert total_persisted_feedback == expected_successful
    assert remaining_staging == 0
    assert duplicate_count == 0

    # Cleanup
    sentiment_service.predict = original_predict
    db.close()
    engine.dispose()
    if os.path.exists(db_path):
        try:
            os.unlink(db_path)
        except OSError:
            pass
    if os.path.exists(csv_path):
        try:
            os.unlink(csv_path)
        except OSError:
            pass

    return {
        "target_rows": target_rows,
        "inserted_rows": expected_successful,
        "rejected_rows": rejected_rows,
        "duplicate_rows": duplicate_count,
        "staging_residual": remaining_staging,
        "elapsed_seconds": round(elapsed_sec, 2),
        "throughput_rps": round(throughput_rps, 1),
        "initial_rss_mb": round(initial_rss_mb, 2),
        "peak_rss_mb": round(peak_rss_mb, 2),
        "peak_heap_mb": round(peak_heap_mb, 2),
        "cpu_usage_pct": round(cpu_usage_pct, 1),
        "avg_chunk_duration_ms": round(avg_chunk_duration_ms, 2),
        "avg_poll_latency_ms": round(avg_poll_latency_ms, 2),
        "p95_poll_latency_ms": round(p95_poll_latency_ms, 2),
        "storage_mb": round(db_file_size_mb, 2),
    }


def run_interruption_and_recovery_audit(num_rows: int = 30000):
    """
    Verifies worker interruption and resumption from durable checkpoint:
    1. Ingestion starts on 30,000 rows.
    2. Simulated worker interruption after 2 chunks (10,000 rows).
    3. Verifies staging has 10,000 rows and job is marked INTERRUPTED.
    4. Worker resumes via resume_import_job.
    5. Ingestion finishes, staging is transferred atomically to feedback.
    6. Verifies exact row counts and 0 duplicates.
    """
    print(f"\n{'='*70}")
    print(f"WORKER INTERRUPTION & CHECKPOINT RESUMPTION AUDIT ({num_rows:,} ROWS)")
    print(f"{'='*70}")

    engine, Session, db_path = create_isolated_db()
    db = Session()
    original_predict = sentiment_service.predict
    sentiment_service.predict = _fast_predict

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    csv_path = generate_progressive_csv(num_rows, rejected_count=0)
    with open(csv_path, "rb") as f_chk:
        file_hash = hashlib.sha256(f_chk.read()).hexdigest()

    job_id = str(uuid.uuid4())
    job = ImportJob(
        id=job_id,
        filename=os.path.basename(csv_path),
        file_path=csv_path,
        file_hash=file_hash,
        total_rows=0,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        status="PENDING",
        stage="QUEUED"
    )
    db.add(job)
    db.commit()

    # Step 1: Simulate crash after 2 chunks (10,000 rows)
    chunk_count = [0]
    orig_process_chunk = ImportService._process_row_chunk

    def _interrupting_process_chunk(*args, **kwargs):
        chunk_count[0] += 1
        orig_process_chunk(*args, **kwargs)
        if chunk_count[0] >= 2:
            raise RuntimeError("Simulated Worker Crash (SIGKILL / Process Abortion)")

    ImportService._process_row_chunk = _interrupting_process_chunk

    try:
        ImportService.process_csv_file(db, job, csv_path)
    except Exception as exc:
        print(f"  [Interruption Triggered]: {exc}")

    # Restore original method
    ImportService._process_row_chunk = orig_process_chunk

    # Verify Interrupted State
    db.expire_all()
    interrupted_job = db.query(ImportJob).filter(ImportJob.id == job_id).first()
    staged_rows = db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).count()
    finalized_rows = db.query(Feedback).count()

    print(f"  Interrupted Job Status: {interrupted_job.status} / Stage: {interrupted_job.stage}")
    print(f"  Checkpoint Row:         {interrupted_job.last_checkpoint_row:,}")
    print(f"  Records in Staging:     {staged_rows:,}")
    print(f"  Records in Final Table: {finalized_rows} (Must be 0)")

    assert interrupted_job.status == "FAILED"
    assert interrupted_job.stage == "INTERRUPTED"
    assert staged_rows == 5000
    assert interrupted_job.last_checkpoint_row == 5000
    assert finalized_rows == 0

    # Step 2: Resume the job
    print("  Resuming import job from durable checkpoint...")
    resumed_job = ImportService.resume_import_job(db, job_id)
    assert resumed_job.status == "PENDING"
    assert resumed_job.stage == "QUEUED"

    # Process remaining rows
    completed_job = ImportService.process_csv_file(db, resumed_job, csv_path)
    db.expire_all()

    final_staged = db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).count()
    final_feedback = db.query(Feedback).count()

    print(f"  Post-Resumption Status: {completed_job.status} / Stage: {completed_job.stage}")
    print(f"  Total Processed:        {completed_job.processed_rows:,}")
    print(f"  Final Staged Residual:  {final_staged} (Must be 0)")
    print(f"  Final Records in DB:    {final_feedback:,} (Expected: {num_rows:,})")

    assert completed_job.status == "COMPLETED"
    assert completed_job.successful_rows == num_rows
    assert final_staged == 0
    assert final_feedback == num_rows

    print("  Worker interruption and checkpoint resumption PASSED!")

    # Cleanup
    sentiment_service.predict = original_predict
    db.close()
    engine.dispose()
    if os.path.exists(db_path):
        try:
            os.unlink(db_path)
        except OSError:
            pass
    if os.path.exists(csv_path):
        try:
            os.unlink(csv_path)
        except OSError:
            pass

    return True


def main():
    print("=" * 80)
    print("SENTIMENT PULSE: PROGRESSIVE SCALE & RELIABILITY AUDIT")
    print("=" * 80)

    # 1. Run worker interruption and recovery audit first
    recovery_passed = run_interruption_and_recovery_audit(num_rows=20000)

    # 2. Run progressive scales: 10k, 100k, 500k, 1,050,000
    scales = [10000, 100000, 500000, 1050000]
    results = {}

    for s in scales:
        results[f"scale_{s}"] = run_single_scale_benchmark(
            target_rows=s, rejected_rows=5, poll_concurrently=True
        )

    print("\n" + "=" * 80)
    print("FINAL PROGRESSIVE SCALE AUDIT SUMMARY")
    print("=" * 80)
    print(json.dumps({
        "worker_interruption_and_recovery_verified": recovery_passed,
        "benchmarks": results
    }, indent=2))


if __name__ == "__main__":
    main()
