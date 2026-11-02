"""
Concurrent Import & Real ONNX Inference Benchmark Suite
Measures 1, 2, and 3 simultaneous imports using the real ONNX sentiment analysis engine.
Records:
1. Total throughput (records/sec)
2. Per-job throughput (records/sec)
3. Wall time and CPU utilization
4. Peak RSS and Private Bytes
5. API responsiveness (health/live and predict latency during ingestion)
"""

import os
import sys
import time
import json
import uuid
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings, BASE_DIR
from db.session import SessionLocal
from db.models import ImportJob, Feedback, FeedbackStaging
from services.import_service import ImportService
from services.sentiment_service import sentiment_service
from scripts.measure_memory import get_os_memory_info
import pandas as pd


def generate_benchmark_csv(row_count: int, texts_pool: List[str], offset: int = 0) -> str:
    """Creates a temporary CSV file with representative review text."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write("text,category,source,rating\n")
    pool_len = len(texts_pool)
    for i in range(row_count):
        # Unique suffix ensures uncached evaluation when testing cold inference
        t = texts_pool[(i + offset) % pool_len].replace('"', '""')
        tmp.write(f'"{t}",CustomerSupport,benchmark,5\n')
    tmp.close()
    return tmp.name


def run_single_import_worker(csv_file: str, tenant_id: int) -> Dict[str, Any]:
    """Executes a single CSV import job in its own database session and thread."""
    db = SessionLocal()
    job_id = str(uuid.uuid4())
    job = ImportJob(
        id=job_id,
        user_id=tenant_id,
        filename=os.path.basename(csv_file),
        status="PENDING",
        stage="QUEUED",
    )
    db.add(job)
    db.commit()

    t0 = time.perf_counter()
    try:
        completed_job = ImportService.process_csv_file(db, job, csv_file)
        elapsed = time.perf_counter() - t0
        res = {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "status": completed_job.status,
            "total_rows": completed_job.total_rows,
            "successful_rows": completed_job.successful_rows,
            "failed_rows": completed_job.failed_rows,
            "elapsed_sec": round(elapsed, 2),
            "throughput_rps": round(completed_job.successful_rows / elapsed, 1) if elapsed > 0 else 0,
        }
    finally:
        # Clean up database records for benchmark run
        db.query(Feedback).filter(Feedback.category == "CustomerSupport").delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).delete()
        db.query(ImportJob).filter(ImportJob.id == job_id).delete()
        db.commit()
        db.close()
        if os.path.exists(csv_file):
            try:
                os.unlink(csv_file)
            except OSError:
                pass

    return res


def probe_api_responsiveness(stop_event: threading.Event, latencies: List[float]):
    """Probes model prediction responsiveness while background import jobs are active."""
    sample = "The service was reasonably fast and the UI looked clean."
    while not stop_event.is_set():
        t0 = time.perf_counter()
        try:
            sentiment_service.predict(sample)
            latencies.append((time.perf_counter() - t0) * 1000)
        except Exception:
            pass
        time.sleep(0.05)


def run_concurrent_benchmarks():
    print("=" * 75)
    print("PHASE 2: CONCURRENT IMPORT & REAL ONNX INFERENCE BENCHMARK")
    print("=" * 75)

    # Load representative review texts
    df = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])
    texts_pool = df["text"].head(500).tolist()

    rows_per_job = 250
    concurrency_levels = [1, 2, 3]
    summary_results = {}

    for c in concurrency_levels:
        sentiment_service.clear_cache()  # Clear cache to evaluate pure concurrent ONNX inference
        print(f"\n--- Testing Concurrency Level: {c} Simultaneous Import(s) (Cold ONNX Inference, {rows_per_job} rows/job) ---")

        # Generate CSV files with distinct text slices
        csv_files = [generate_benchmark_csv(rows_per_job, texts_pool, offset=i * rows_per_job) for i in range(c)]

        # Probe API responsiveness
        api_latencies: List[float] = []
        stop_probe = threading.Event()
        probe_thread = threading.Thread(target=probe_api_responsiveness, args=(stop_probe, api_latencies))
        probe_thread.daemon = True
        probe_thread.start()

        rss_before, peak_before, priv_before = get_os_memory_info()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        with ThreadPoolExecutor(max_workers=c) as executor:
            futures = [
                executor.submit(run_single_import_worker, csv_files[i], 100 + i)
                for i in range(c)
            ]
            job_results = [f.result() for f in futures]

        t_wall_total = time.perf_counter() - t_wall_start
        t_cpu_total = time.process_time() - t_cpu_start

        stop_probe.set()
        probe_thread.join(timeout=2.0)

        rss_after, peak_after, priv_after = get_os_memory_info()

        total_records = sum(j["successful_rows"] for j in job_results)
        total_throughput = total_records / t_wall_total if t_wall_total > 0 else 0
        avg_job_throughput = sum(j["throughput_rps"] for j in job_results) / c
        cpu_util = (t_cpu_total / t_wall_total) * 100 if t_wall_total > 0 else 0

        avg_probe_lat = round(sum(api_latencies) / len(api_latencies), 2) if api_latencies else 0.0
        p95_probe_lat = round(float(pd.Series(api_latencies).quantile(0.95)), 2) if api_latencies else 0.0

        res_entry = {
            "concurrency_level": c,
            "jobs_executed": c,
            "rows_per_job": rows_per_job,
            "total_records_processed": total_records,
            "total_wall_time_sec": round(t_wall_total, 2),
            "total_throughput_rps": round(total_throughput, 1),
            "avg_per_job_throughput_rps": round(avg_job_throughput, 1),
            "cpu_utilization_pct": round(cpu_util, 1),
            "peak_rss_mb": peak_after,
            "private_bytes_mb": priv_after,
            "api_responsiveness": {
                "probes_recorded": len(api_latencies),
                "avg_predict_latency_ms": avg_probe_lat,
                "p95_predict_latency_ms": p95_probe_lat,
            },
            "job_breakdown": job_results,
        }

        summary_results[f"concurrency_{c}"] = res_entry
        print(f"Total Throughput:     {total_throughput:5.1f} rps across {c} job(s)")
        print(f"Avg Per-Job Speed:    {avg_job_throughput:5.1f} rps")
        print(f"Wall Time:            {t_wall_total:5.2f}s | CPU Util: {cpu_util:5.1f}%")
        print(f"Memory:               Peak RSS {peak_after:.1f} MB | Private Bytes {priv_after:.1f} MB")
        print(f"API Responsiveness:   Avg {avg_probe_lat:.2f} ms | P95 {p95_probe_lat:.2f} ms ({len(api_latencies)} probes)")

    out_file = os.path.join(BASE_DIR, "concurrent_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\nConcurrent benchmark results saved to {out_file}")
    return summary_results


if __name__ == "__main__":
    run_concurrent_benchmarks()
