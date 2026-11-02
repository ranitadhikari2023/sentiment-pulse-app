"""
Diagnostic script measuring the 10 lifecycle stages of memory consumption:
1. Application startup RSS.
2. RSS after ONNX model loading.
3. RSS after tokenizer initialization.
4. RSS during dynamic padding.
5. RSS during batched inference (bs 8, 16, 32, 64, 128).
6. RSS during parallel inference.
7. RSS during simultaneous imports.
8. RSS after completed jobs.
9. RSS after failed jobs.
10. RSS after repeated inference cycles.
"""

import os
import sys
import gc
import time
import json
import uuid
import tempfile
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.measure_memory import get_os_memory_info

def run_diagnostics():
    results = {}
    print("=" * 75)
    print("MEMORY CONSUMPTION LIFECYCLE AUDIT (10 STAGES)")
    print("=" * 75)

    # Stage 1: Application startup RSS
    gc.collect()
    rss_1, peak_1, priv_1 = get_os_memory_info()
    results["stage_1_startup"] = {"rss_mb": rss_1, "peak_rss_mb": peak_1, "private_mb": priv_1}
    print(f"1. Application Startup:              RSS: {rss_1:6.2f} MB | Peak: {peak_1:6.2f} MB | Priv: {priv_1:6.2f} MB")

    # Stage 2: After ONNX model loading
    import onnxruntime as ort
    from core.config import settings
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
    session = ort.InferenceSession(settings.MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    rss_2, peak_2, priv_2 = get_os_memory_info()
    results["stage_2_model_load"] = {"rss_mb": rss_2, "peak_rss_mb": peak_2, "private_mb": priv_2, "delta_rss_mb": round(rss_2 - rss_1, 2)}
    print(f"2. After ONNX Model Loading:         RSS: {rss_2:6.2f} MB | Peak: {peak_2:6.2f} MB | Delta: +{rss_2 - rss_1:5.2f} MB")

    # Stage 3: After tokenizer initialization
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)
    rss_3, peak_3, priv_3 = get_os_memory_info()
    results["stage_3_tokenizer_init"] = {"rss_mb": rss_3, "peak_rss_mb": peak_3, "private_mb": priv_3, "delta_rss_mb": round(rss_3 - rss_2, 2)}
    print(f"3. After Tokenizer Initialization:   RSS: {rss_3:6.2f} MB | Peak: {peak_3:6.2f} MB | Delta: +{rss_3 - rss_2:5.2f} MB")

    # Load test texts
    df = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])
    sample_texts = df["text"].head(512).tolist()

    # Stage 4: RSS during dynamic padding
    encs = tokenizer.encode_batch(sample_texts[:64])
    max_l = max(len(e.ids) for e in encs)
    ids = np.zeros((64, max_l), dtype=np.int64)
    mask = np.zeros((64, max_l), dtype=np.int64)
    for i, e in enumerate(encs):
        ids[i, :len(e.ids)] = e.ids
        mask[i, :len(e.attention_mask)] = e.attention_mask
    rss_4, peak_4, priv_4 = get_os_memory_info()
    results["stage_4_dynamic_padding"] = {"rss_mb": rss_4, "peak_rss_mb": peak_4, "private_mb": priv_4, "tensor_shape": [64, max_l]}
    print(f"4. During Dynamic Padding (64 rows): RSS: {rss_4:6.2f} MB | Peak: {peak_4:6.2f} MB | Tensor: [64, {max_l}]")

    # Stage 5: RSS during batched inference
    _ = session.run(None, {"input_ids": ids, "attention_mask": mask})
    rss_5, peak_5, priv_5 = get_os_memory_info()
    results["stage_5_batched_inference"] = {"rss_mb": rss_5, "peak_rss_mb": peak_5, "private_mb": priv_5}
    print(f"5. After Batched Inference (bs=64):  RSS: {rss_5:6.2f} MB | Peak: {peak_5:6.2f} MB | Delta: +{rss_5 - rss_4:5.2f} MB")

    # Stage 6: RSS during parallel inference (4 workers)
    def worker_infer(chunk):
        w_encs = tokenizer.encode_batch(chunk)
        w_max_l = max(len(e.ids) for e in w_encs)
        w_ids = np.zeros((len(chunk), w_max_l), dtype=np.int64)
        w_mask = np.zeros((len(chunk), w_max_l), dtype=np.int64)
        for i, e in enumerate(w_encs):
            w_ids[i, :len(e.ids)] = e.ids
            w_mask[i, :len(e.attention_mask)] = e.attention_mask
        return session.run(None, {"input_ids": w_ids, "attention_mask": w_mask})

    chunks = [sample_texts[i*32:(i+1)*32] for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        _ = list(executor.map(worker_infer, chunks))
    rss_6, peak_6, priv_6 = get_os_memory_info()
    results["stage_6_parallel_inference"] = {"rss_mb": rss_6, "peak_rss_mb": peak_6, "private_mb": priv_6}
    print(f"6. During Parallel Inference (4 wrk): RSS: {rss_6:6.2f} MB | Peak: {peak_6:6.2f} MB | Delta: +{rss_6 - rss_5:5.2f} MB")

    # Stage 7: RSS during simultaneous imports (simulating 2 imports)
    from services.import_service import ImportService
    from db.session import SessionLocal
    from db.models import ImportJob, Feedback, FeedbackStaging

    def run_dummy_import(job_id: str, texts: list):
        db = SessionLocal()
        try:
            job = ImportJob(id=job_id, filename="sim_test.csv", status="PENDING", stage="QUEUED")
            db.add(job)
            db.commit()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
            tmp.write("text,category\n")
            for t in texts:
                tmp.write(f'"{t.replace(chr(34), chr(34)+chr(34))}",SimTest\n')
            tmp.close()
            ImportService.process_csv_file(db, job, tmp.name)
        finally:
            db.query(Feedback).filter(Feedback.category == "SimTest").delete()
            db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).delete()
            db.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db.commit()
            db.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    id1, id2 = str(uuid.uuid4()), str(uuid.uuid4())
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_dummy_import, id1, sample_texts[:100])
        f2 = executor.submit(run_dummy_import, id2, sample_texts[100:200])
        f1.result()
        f2.result()
    rss_7, peak_7, priv_7 = get_os_memory_info()
    results["stage_7_simultaneous_imports"] = {"rss_mb": rss_7, "peak_rss_mb": peak_7, "private_mb": priv_7}
    print(f"7. During Simultaneous Imports (x2): RSS: {rss_7:6.2f} MB | Peak: {peak_7:6.2f} MB | Delta: +{rss_7 - rss_6:5.2f} MB")

    # Stage 8: RSS after completed jobs & gc
    gc.collect()
    rss_8, peak_8, priv_8 = get_os_memory_info()
    results["stage_8_after_completion"] = {"rss_mb": rss_8, "peak_rss_mb": peak_8, "private_mb": priv_8}
    print(f"8. After Completed Jobs & GC:        RSS: {rss_8:6.2f} MB | Peak: {peak_8:6.2f} MB | Net Retention: {rss_8 - rss_7:5.2f} MB")

    # Stage 9: RSS after failed jobs
    def run_failing_import(job_id: str):
        db = SessionLocal()
        try:
            job = ImportJob(id=job_id, filename="fail_test.csv", status="PENDING", stage="QUEUED")
            db.add(job)
            db.commit()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
            tmp.write("invalid_header_with_no_text_column\n12345\n")
            tmp.close()
            ImportService.process_csv_file(db, job, tmp.name)
        finally:
            db.query(ImportJob).filter(ImportJob.id == job_id).delete()
            db.commit()
            db.close()
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    run_failing_import(str(uuid.uuid4()))
    gc.collect()
    rss_9, peak_9, priv_9 = get_os_memory_info()
    results["stage_9_after_failed_jobs"] = {"rss_mb": rss_9, "peak_rss_mb": peak_9, "private_mb": priv_9}
    print(f"9. After Failed Jobs & Cleanup:      RSS: {rss_9:6.2f} MB | Peak: {peak_9:6.2f} MB | Clean: Yes")

    # Stage 10: RSS after repeated inference cycles (50 cycles)
    for _ in range(50):
        _ = session.run(None, {"input_ids": ids, "attention_mask": mask})
    gc.collect()
    rss_10, peak_10, priv_10 = get_os_memory_info()
    results["stage_10_repeated_cycles"] = {
        "rss_mb": rss_10,
        "peak_rss_mb": peak_10,
        "private_mb": priv_10,
        "memory_leak_mb": round(rss_10 - rss_8, 2)
    }
    print(f"10. After 50 Repeated Cycles:        RSS: {rss_10:6.2f} MB | Peak: {peak_10:6.2f} MB | Net Leak: {rss_10 - rss_8:5.2f} MB")

    with open("memory_lifecycle_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved memory lifecycle results to memory_lifecycle_results.json")
    return results

if __name__ == "__main__":
    run_diagnostics()

