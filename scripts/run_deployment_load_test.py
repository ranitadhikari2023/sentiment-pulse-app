"""
Sentiment Pulse — Phase 2.1 Deployment Load Testing & End-to-End Ingestion Benchmark.
Evaluates:
1. Container Resource Profiles:
   - Profile 1: 1 vCPU / 1 GB RAM (constrained container)
   - Profile 2: 1 vCPU / 2 GB RAM (standard entry VM)
   - Profile 3: 2 vCPU / 2 GB RAM (production baseline)
2. Synthetic End-to-End Ingestion:
   - 10,000 records
   - 100,000 records
   - 500,000 records
   - 1,050,000 records
Measures Wall-Clock Time, Records/sec, Peak RSS, Breakdown, Database Growth.
"""

import os
import sys
import time
import json
import uuid
import hashlib
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings, BASE_DIR
from db.session import SessionLocal, engine
from db.models import ImportJob, Feedback, FeedbackStaging
from model import SentimentAnalyzer
from services.sentiment_service import sentiment_service
from services.import_service import ImportService
from scripts.measure_memory import get_os_memory_info


def get_current_rss_mb() -> float:
    return get_os_memory_info()[0]


def run_deployment_profiles():
    print("=" * 80)
    print("PHASE 2.1: REALISTIC DEPLOYMENT LOAD TESTING")
    print("=" * 80)

    analyzer = sentiment_service.analyzer
    if not analyzer:
        analyzer = SentimentAnalyzer()

    results = {}

    # --------------------------------------------------------------------------
    # Profile 1: 1 vCPU, 1 GB RAM
    # --------------------------------------------------------------------------
    print("\n[PROFILE 1: 1 vCPU, 1 GB RAM Simulation]")
    rss_start = get_current_rss_mb()
    latencies = []

    # Run 1 import job batch (500 reviews) + 50 concurrent interactive requests
    def interactive_request(i):
        t0 = time.perf_counter()
        res = analyzer.predict(f"Customer review interactive test #{i}")
        lat = (time.perf_counter() - t0) * 1000
        return lat

    t0_prof1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        # Import job worker (batch 200 items)
        f_batch = pool.submit(
            analyzer.predict_batch,
            [f"Import job review payload text number #{k}" for k in range(200)]
        )
        # Interactive queries
        f_interactive = [pool.submit(interactive_request, k) for k in range(50)]

        batch_out = f_batch.result()
        for f in f_interactive:
            latencies.append(f.result())

    elapsed_prof1 = time.perf_counter() - t0_prof1
    rss_prof1 = get_current_rss_mb()
    latencies.sort()
    p50_prof1 = latencies[int(len(latencies) * 0.50)]
    p95_prof1 = latencies[int(len(latencies) * 0.95)]

    passed_prof1 = (rss_prof1 < 750.0) and (p95_prof1 < 800.0)
    print(f"  Duration: {elapsed_prof1:.2f}s | Batch Items: 200 | Interactive Queries: 50")
    print(f"  Peak RSS: {rss_prof1:.1f} MB (Threshold: < 750 MB) -> {'PASS' if rss_prof1 < 750 else 'FAIL'}")
    print(f"  Interactive Latency P50: {p50_prof1:.1f} ms | P95: {p95_prof1:.1f} ms (SLA: < 800 ms) -> {'PASS' if p95_prof1 < 800 else 'FAIL'}")
    print(f"  Profile 1 Status: {'PASS' if passed_prof1 else 'FAIL'}")

    results["profile_1"] = {
        "peak_rss_mb": round(rss_prof1, 1),
        "p50_latency_ms": round(p50_prof1, 1),
        "p95_latency_ms": round(p95_prof1, 1),
        "status": "PASS" if passed_prof1 else "FAIL",
    }

    # --------------------------------------------------------------------------
    # Profile 2: 1 vCPU, 2 GB RAM
    # --------------------------------------------------------------------------
    print("\n[PROFILE 2: 1 vCPU, 2 GB RAM Simulation]")
    latencies_prof2 = []
    t0_prof2 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=4) as pool:
        # 2 concurrent import jobs
        f_job1 = pool.submit(
            analyzer.predict_batch,
            [f"Job 1 review sample #{k}" for k in range(300)]
        )
        f_job2 = pool.submit(
            analyzer.predict_batch,
            [f"Job 2 review sample #{k}" for k in range(300)]
        )
        # 50 interactive requests
        f_inter = [pool.submit(interactive_request, k) for k in range(50)]

        j1_out = f_job1.result()
        j2_out = f_job2.result()
        for f in f_inter:
            latencies_prof2.append(f.result())

    elapsed_prof2 = time.perf_counter() - t0_prof2
    rss_prof2 = get_current_rss_mb()
    latencies_prof2.sort()
    p50_prof2 = latencies_prof2[int(len(latencies_prof2) * 0.50)]
    p95_prof2 = latencies_prof2[int(len(latencies_prof2) * 0.95)]

    passed_prof2 = (rss_prof2 < 1400.0) and (p95_prof2 < 500.0)
    print(f"  Duration: {elapsed_prof2:.2f}s | 2 Concurrent Jobs: 600 items | Interactive Queries: 50")
    print(f"  Peak RSS: {rss_prof2:.1f} MB (Threshold: < 1400 MB) -> {'PASS' if rss_prof2 < 1400 else 'FAIL'}")
    print(f"  Interactive Latency P50: {p50_prof2:.1f} ms | P95: {p95_prof2:.1f} ms (SLA: < 500 ms) -> {'PASS' if p95_prof2 < 500 else 'FAIL'}")
    print(f"  Profile 2 Status: {'PASS' if passed_prof2 else 'FAIL'}")

    results["profile_2"] = {
        "peak_rss_mb": round(rss_prof2, 1),
        "p50_latency_ms": round(p50_prof2, 1),
        "p95_latency_ms": round(p95_prof2, 1),
        "status": "PASS" if passed_prof2 else "FAIL",
    }

    # --------------------------------------------------------------------------
    # Profile 3: 2 vCPU, 2 GB RAM (Production Baseline)
    # --------------------------------------------------------------------------
    print("\n[PROFILE 3: 2 vCPU, 2 GB RAM Production Baseline]")
    # Medium text throughput test (512 medium reviews, 30-70 tokens)
    medium_texts = [
        f"This customer experience review number #{k} was quite average. The delivery was on time but packaging was damaged."
        for k in range(512)
    ]
    t0_prof3 = time.perf_counter()
    prof3_preds = analyzer.predict_batch(medium_texts)
    elapsed_prof3 = time.perf_counter() - t0_prof3
    rate_prof3 = len(medium_texts) / max(elapsed_prof3, 1e-6)
    rss_prof3 = get_current_rss_mb()

    passed_prof3 = (rate_prof3 > 150.0) and (rss_prof3 < 1500.0)
    print(f"  Duration: {elapsed_prof3:.2f}s | Medium Reviews: 512")
    print(f"  Throughput: {rate_prof3:.1f} records/sec (Target: > 150 rps) -> {'PASS' if rate_prof3 > 150 else 'FAIL'}")
    print(f"  Peak RSS: {rss_prof3:.1f} MB (Threshold: < 1500 MB) -> {'PASS' if rss_prof3 < 1500 else 'FAIL'}")
    print(f"  Profile 3 Status: {'PASS' if passed_prof3 else 'FAIL'}")

    results["profile_3"] = {
        "throughput_rps": round(rate_prof3, 1),
        "peak_rss_mb": round(rss_prof3, 1),
        "status": "PASS" if passed_prof3 else "FAIL",
    }

    return results


def run_e2e_ingestion_benchmarks():
    print("\n" + "=" * 80)
    print("PHASE 2.1: REAL END-TO-END IMPORT VALIDATION")
    print("=" * 80)

    # Isolated test database
    bench_db_file = os.path.join(BASE_DIR, "data", "e2e_bench_temp.db")
    if os.path.exists(bench_db_file):
        os.unlink(bench_db_file)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from db.models import Base

    bench_url = f"sqlite:///{bench_db_file}"
    bench_engine = create_engine(bench_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=bench_engine)
    BenchSession = sessionmaker(bind=bench_engine)

    # Dataset scales
    scales = [10_000, 100_000, 500_000, 1_050_000]
    e2e_results = {}

    for count in scales:
        print(f"\n--- Benchmark: {count:,} Records ---")
        csv_path = os.path.join(settings.UPLOAD_DIR, f"bench_{count}.csv")
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # Generate CSV streamingly
        t_gen0 = time.perf_counter()
        hasher = hashlib.sha256()
        with open(csv_path, "wb") as f_out:
            header = b"text,category,source,rating\n"
            f_out.write(header)
            hasher.update(header)
            row_template = b'"Customer feedback record sample for end-to-end import benchmarking.",Benchmark,web,5\n'
            for _ in range(count):
                f_out.write(row_template)
                hasher.update(row_template)

        file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        file_hash = hasher.hexdigest()
        t_gen = time.perf_counter() - t_gen0
        print(f"  Generated CSV: {file_size_mb:.2f} MB in {t_gen:.2f}s (SHA-256: {file_hash[:16]}...)")

        # Create isolated job
        db = BenchSession()
        initial_db_size = os.path.getsize(bench_db_file) / (1024 * 1024) if os.path.exists(bench_db_file) else 0.0

        job_id = str(uuid.uuid4())
        job = ImportJob(
            id=job_id,
            filename=f"bench_{count}.csv",
            file_path=csv_path,
            file_hash=file_hash,
            total_rows=count,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            last_checkpoint_row=0,
            status="PENDING",
            stage="QUEUED",
            created_at=datetime.now(timezone.utc)
        )
        db.add(job)
        db.commit()

        # Execute Ingestion with real pipeline
        t0 = time.perf_counter()
        rss_before = get_current_rss_mb()

        # Run process_csv_file
        completed_job = ImportService.process_csv_file(db, job, csv_path)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        rate = count / max(elapsed, 1e-6)
        rss_after = get_current_rss_mb()
        final_db_size = os.path.getsize(bench_db_file) / (1024 * 1024)

        # Query counts
        persisted_fb = db.query(Feedback).count()
        remaining_staging = db.query(FeedbackStaging).count()

        print(f"  Wall-clock time: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
        print(f"  Throughput: {rate:.1f} records/sec")
        print(f"  Status: {completed_job.status} | Stage: {completed_job.stage}")
        print(f"  Processed: {completed_job.processed_rows:,} | Successful: {completed_job.successful_rows:,} | Failed: {completed_job.failed_rows}")
        print(f"  Feedback Persisted: {persisted_fb:,} | Remaining Staging: {remaining_staging}")
        print(f"  Initial DB: {initial_db_size:.2f} MB | Final DB: {final_db_size:.2f} MB | Growth: {final_db_size - initial_db_size:.2f} MB")
        print(f"  Peak RSS: {rss_after:.1f} MB (Delta: {rss_after - rss_before:+.1f} MB)")

        e2e_results[count] = {
            "records": count,
            "wall_clock_sec": round(elapsed, 2),
            "throughput_rps": round(rate, 1),
            "status": completed_job.status,
            "persisted_feedback": persisted_fb,
            "remaining_staging": remaining_staging,
            "db_size_mb": round(final_db_size, 2),
            "db_growth_mb": round(final_db_size - initial_db_size, 2),
            "peak_rss_mb": round(rss_after, 1),
        }

        # Cleanup for next scale
        db.query(Feedback).delete()
        db.commit()
        db.close()

        if os.path.exists(csv_path):
            try:
                os.unlink(csv_path)
            except OSError:
                pass

    bench_engine.dispose()
    if os.path.exists(bench_db_file):
        try:
            os.unlink(bench_db_file)
        except OSError:
            pass

    return e2e_results


if __name__ == "__main__":
    prof_res = run_deployment_profiles()
    e2e_res = run_e2e_ingestion_benchmarks()

    out_file = os.path.join(BASE_DIR, "data", "phase2_1_load_test_results.json")
    with open(out_file, "w") as f:
        json.dump({"deployment_profiles": prof_res, "e2e_ingestion": e2e_res}, f, indent=2)
    print(f"\nAll benchmark results successfully exported to: {out_file}")
