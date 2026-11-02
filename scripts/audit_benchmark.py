"""
Sentiment Pulse - Post-Implementation Performance Verification & Reliability Audit
Measures:
1. Pipeline A: CSV parsing and validation only.
2. Pipeline B: Database insertion without inference (Core bulk insert + finalization).
3. Pipeline C: Real batched ONNX inference (various batch sizes).
4. Pipeline D: Complete end-to-end CSV ingestion using real production ONNX inference.
5. OS-level Peak RSS, Private Bytes, and Python tracemalloc.
6. Clean isolated database per benchmark scale to measure true database file growth.
"""

import os
import sys
import time
import json
import uuid
import tempfile
import tracemalloc
from datetime import datetime, timezone
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import create_engine, text, insert, select, delete
from sqlalchemy.orm import sessionmaker

from core.config import settings, BASE_DIR
from db.models import Base, Feedback, ImportJob, FeedbackStaging
from services.import_service import ImportService
from services.sentiment_service import sentiment_service
from scripts.measure_memory import get_os_memory_info


def create_temp_csv(num_rows: int) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8", newline="")
    tmp.write("text,category,source,rating\n")
    batch_size = 10000
    rows_written = 0
    sample_texts = [
        "The application is fast, intuitive, and extremely helpful for our customer support team.",
        "Constant crashing on startup after the recent upgrade, highly disappointed with the quality.",
        "Decent user interface but lacks advanced analytics and export options for enterprise users.",
        "Exceptional responsiveness and outstanding accuracy in sentiment classification.",
        "Worst experience ever, customer service never answered our priority support tickets."
    ]
    while rows_written < num_rows:
        take = min(batch_size, num_rows - rows_written)
        lines = "".join(
            f'"{sample_texts[(rows_written + i) % len(sample_texts)]} - ID {rows_written + i}",Performance,web,{1 + (i % 5)}\n'
            for i in range(take)
        )
        tmp.write(lines)
        rows_written += take
    tmp.flush()
    tmp.close()
    return tmp.name


def create_isolated_db():
    """Creates a clean, isolated temporary SQLite database with WAL mode."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp_path = tmp.name
    tmp.close()
    db_url = f"sqlite:///{tmp_path}"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text("PRAGMA busy_timeout=60000"))
        conn.commit()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session, tmp_path


def benchmark_pipeline_a_parsing(csv_path: str, num_rows: int) -> Dict[str, Any]:
    """A. CSV parsing and validation only."""
    import csv
    t0 = time.perf_counter()
    encoding = ImportService._detect_file_encoding(csv_path)
    delimiter = ImportService.detect_csv_delimiter(csv_path, encoding)
    total_parsed = 0
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        text_idx, cat_idx, src_idx, rate_idx, _ = ImportService.map_header_row(header)
        for row in reader:
            if row and any(c.strip() for c in row):
                total_parsed += 1
    t1 = time.perf_counter()
    duration = t1 - t0
    rps = total_parsed / duration if duration > 0 else 0
    return {
        "pipeline": "A_parsing_and_validation_only",
        "rows": total_parsed,
        "duration_sec": round(duration, 3),
        "throughput_rps": round(rps, 1),
    }


def benchmark_pipeline_b_db_insertion(num_rows: int) -> Dict[str, Any]:
    """B. Database insertion without inference (Core bulk insert + set-based finalization)."""
    engine, Session, db_path = create_isolated_db()
    db = Session()
    job_id = str(uuid.uuid4())
    initial_db_size = os.path.getsize(db_path) / (1024 * 1024)

    job = ImportJob(
        id=job_id,
        filename=f"bench_db_{num_rows}.csv",
        status="PENDING",
        stage="INSERTING",
        total_rows=num_rows,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0,
        created_at=datetime.now(timezone.utc)
    )
    db.add(job)
    db.commit()

    chunk_size = settings.DB_INSERT_CHUNK_SIZE
    now_utc = datetime.now(timezone.utc)
    
    t0 = time.perf_counter()
    inserted_so_far = 0
    while inserted_so_far < num_rows:
        take = min(chunk_size, num_rows - inserted_so_far)
        batch = [
            {
                "import_job_id": job_id,
                "user_id": None,
                "text": f"Pre-validated feedback text index {inserted_so_far + i}",
                "clean_text": f"Pre-validated feedback text index {inserted_so_far + i}",
                "sentiment": "positive" if i % 2 == 0 else "neutral",
                "confidence": 0.95,
                "positive_score": 0.95,
                "neutral_score": 0.03,
                "negative_score": 0.02,
                "category": "Performance",
                "source": "web",
                "rating": 5,
                "model_version": "3.0.0",
                "created_at": now_utc,
                "updated_at": now_utc,
            }
            for i in range(take)
        ]
        db.execute(insert(FeedbackStaging), batch)
        db.commit()
        inserted_so_far += take

    # Finalization transfer
    t_finalize_0 = time.perf_counter()
    transfer_stmt = insert(Feedback).from_select(
        [
            Feedback.user_id, Feedback.text, Feedback.clean_text, Feedback.sentiment,
            Feedback.confidence, Feedback.positive_score, Feedback.neutral_score,
            Feedback.negative_score, Feedback.category, Feedback.source,
            Feedback.rating, Feedback.model_version, Feedback.created_at, Feedback.updated_at
        ],
        select(
            FeedbackStaging.user_id, FeedbackStaging.text, FeedbackStaging.clean_text,
            FeedbackStaging.sentiment, FeedbackStaging.confidence, FeedbackStaging.positive_score,
            FeedbackStaging.neutral_score, FeedbackStaging.negative_score, FeedbackStaging.category,
            FeedbackStaging.source, FeedbackStaging.rating, FeedbackStaging.model_version,
            FeedbackStaging.created_at, FeedbackStaging.updated_at
        ).where(FeedbackStaging.import_job_id == job_id)
    )
    db.execute(transfer_stmt)
    db.execute(delete(FeedbackStaging).where(FeedbackStaging.import_job_id == job_id))
    job.status = "COMPLETED"
    job.stage = "COMPLETED"
    job.successful_rows = num_rows
    job.processed_rows = num_rows
    db.commit()
    t1 = time.perf_counter()

    final_db_size = os.path.getsize(db_path) / (1024 * 1024)
    duration = t1 - t0
    finalization_time = t1 - t_finalize_0
    rps = num_rows / duration if duration > 0 else 0

    fb_count = db.query(Feedback).count()
    staging_count = db.query(FeedbackStaging).count()
    db.close()
    engine.dispose()
    try:
        os.unlink(db_path)
    except OSError:
        pass

    return {
        "pipeline": "B_database_insertion_without_inference",
        "rows": num_rows,
        "total_duration_sec": round(duration, 3),
        "finalization_sec": round(finalization_time, 3),
        "throughput_rps": round(rps, 1),
        "initial_db_size_mb": round(initial_db_size, 2),
        "final_db_size_mb": round(final_db_size, 2),
        "db_growth_mb": round(final_db_size - initial_db_size, 2),
        "persisted_records": fb_count,
        "remaining_staging": staging_count
    }


def benchmark_pipeline_c_real_onnx(num_samples: int = 2000, batch_sizes: List[int] = None) -> Dict[str, Any]:
    """C. Real batched ONNX inference."""
    if batch_sizes is None:
        batch_sizes = [32, 64, 128, 256, 500, 1000]

    sample_texts = [
        "The customer service was exceptional and the issue was resolved within five minutes.",
        "Terrible product quality, the plastic hinge cracked after three days of normal usage.",
        "Average platform, user interface is somewhat confusing but core reporting works.",
        "Outstanding performance and very reliable notifications for our engineering team.",
        "Worst software upgrade ever, completely broke our existing integrations and workflows."
    ] * (num_samples // 5 + 1)
    texts = sample_texts[:num_samples]

    results_by_batch = {}
    for bs in batch_sizes:
        batch = texts[:bs]
        # Warmup
        _ = sentiment_service.analyzer.predict_batch(batch[:min(8, len(batch))])
        
        t0 = time.perf_counter()
        preds = sentiment_service.analyzer.predict_batch(batch)
        t1 = time.perf_counter()
        
        dur = t1 - t0
        rps = len(batch) / dur if dur > 0 else 0
        latency_per_item_ms = (dur / len(batch)) * 1000
        results_by_batch[f"batch_{bs}"] = {
            "batch_size": bs,
            "duration_sec": round(dur, 4),
            "throughput_rps": round(rps, 1),
            "latency_per_item_ms": round(latency_per_item_ms, 2)
        }

    # Also measure single-item inference for comparison
    t0_single = time.perf_counter()
    single_count = min(100, num_samples)
    for i in range(single_count):
        _ = sentiment_service.analyzer.predict(texts[i])
    t1_single = time.perf_counter()
    single_dur = t1_single - t0_single
    single_rps = single_count / single_dur if single_dur > 0 else 0
    single_latency_ms = (single_dur / single_count) * 1000

    return {
        "pipeline": "C_real_batched_onnx_inference",
        "single_item_unbatched": {
            "evaluated_items": single_count,
            "latency_per_item_ms": round(single_latency_ms, 2),
            "throughput_rps": round(single_rps, 1)
        },
        "batched_inference": results_by_batch
    }


def benchmark_pipeline_d_e2e_real_onnx(num_rows: int) -> Dict[str, Any]:
    """D. Complete end-to-end CSV ingestion using actual production ONNX inference."""
    engine, Session, db_path = create_isolated_db()
    db = Session()
    csv_path = create_temp_csv(num_rows)
    job_id = str(uuid.uuid4())
    initial_db_size = os.path.getsize(db_path) / (1024 * 1024)

    job = ImportJob(
        id=job_id,
        filename=f"bench_e2e_real_{num_rows}.csv",
        status="PENDING",
        stage="QUEUED",
        total_rows=0,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0,
        created_at=datetime.now(timezone.utc)
    )
    db.add(job)
    db.commit()

    rss_0, peak_0, private_0 = get_os_memory_info()
    tracemalloc.start()
    t0 = time.perf_counter()

    completed_job = ImportService.process_csv_file(db, job, csv_path)

    t1 = time.perf_counter()
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_1, peak_rss_1, private_1 = get_os_memory_info()

    duration = t1 - t0
    final_db_size = os.path.getsize(db_path) / (1024 * 1024)
    rps = num_rows / duration if duration > 0 else 0

    fb_count = db.query(Feedback).count()
    staging_count = db.query(FeedbackStaging).count()
    db.close()
    engine.dispose()
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
        "pipeline": "D_e2e_with_real_onnx_inference",
        "rows": num_rows,
        "total_duration_sec": round(duration, 2),
        "throughput_rps": round(rps, 1),
        "tracemalloc_peak_mb": round(tracemalloc_peak / (1024 * 1024), 2),
        "os_rss_start_mb": rss_0,
        "os_rss_final_mb": rss_1,
        "os_peak_rss_mb": peak_rss_1,
        "os_private_bytes_mb": private_1,
        "initial_db_size_mb": round(initial_db_size, 2),
        "final_db_size_mb": round(final_db_size, 2),
        "db_growth_mb": round(final_db_size - initial_db_size, 2),
        "persisted_records": fb_count,
        "remaining_staging": staging_count,
        "status": completed_job.status,
        "stage": completed_job.stage
    }


def main():
    print("=" * 70)
    print("POST-IMPLEMENTATION PERFORMANCE VERIFICATION & RELIABILITY AUDIT")
    print("=" * 70)

    # 1. Warm up & test Real ONNX Inference
    print("\n[Audit 1/4] Measuring Real Batched ONNX Inference (Pipeline C)...")
    onnx_results = benchmark_pipeline_c_real_onnx(num_samples=1000)
    print(json.dumps(onnx_results, indent=2))

    # 2. Measure CSV Parsing & Validation (Pipeline A)
    print("\n[Audit 2/4] Measuring CSV Parsing & Validation (Pipeline A)...")
    csv_100k = create_temp_csv(100000)
    parsing_results = benchmark_pipeline_a_parsing(csv_100k, 100000)
    os.unlink(csv_100k)
    print(json.dumps(parsing_results, indent=2))

    # 3. Measure Database Bulk Insertion without Inference (Pipeline B) across isolated DBs
    print("\n[Audit 3/4] Measuring Database Bulk Insertion & Finalization on Isolated DBs (Pipeline B)...")
    db_results = {}
    for count in [1000, 10000, 50000, 100000]:
        res = benchmark_pipeline_b_db_insertion(count)
        db_results[f"{count}_rows"] = res
        print(f"  {count:,} rows: {res['total_duration_sec']}s ({res['throughput_rps']:,.1f} rps), growth: {res['db_growth_mb']} MB")

    # 4. Measure End-to-End Ingestion with Real Production ONNX Inference (Pipeline D)
    print("\n[Audit 4/4] Measuring Complete E2E Ingestion with Real ONNX Inference (Pipeline D)...")
    e2e_results = {}
    for count in [100, 500, 1000, 2000]:
        res = benchmark_pipeline_d_e2e_real_onnx(count)
        e2e_results[f"{count}_rows"] = res
        print(f"  {count:,} rows: {res['total_duration_sec']}s ({res['throughput_rps']:,.1f} rps), Peak RSS: {res['os_peak_rss_mb']} MB, Python Heap: {res['tracemalloc_peak_mb']} MB")

    audit_summary = {
        "pipeline_a_parsing": parsing_results,
        "pipeline_b_db_storage": db_results,
        "pipeline_c_onnx_inference": onnx_results,
        "pipeline_d_e2e_real_onnx": e2e_results
    }

    out_path = os.path.join(BASE_DIR, "audit_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"\nAudit benchmarks completed. Results written to {out_path}")


if __name__ == "__main__":
    main()

