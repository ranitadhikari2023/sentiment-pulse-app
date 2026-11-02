"""
Re-evaluates optimal batch size across Short, Medium, and Long reviews
with Dynamic Padding, Length Bucketing, and Optimized INT8 ONNX Model.
Measures:
- Real ONNX throughput (records/sec)
- Average batch latency (ms)
- P50 and P95 latency (ms)
- Peak RSS (MB) & Private Bytes (MB)
- CPU utilization (%)
- Memory per active worker
- Prediction consistency across batch sizes
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.measure_memory import get_os_memory_info
from core.config import settings

def run_evaluation():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(settings.MODEL_PATH, opts, providers=["CPUExecutionProvider"])

    tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)

    # 1. Prepare Short, Medium, and Long text pools
    df = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])
    all_texts = df["text"].tolist()

    short_texts = []   # < 30 tokens
    medium_texts = []  # 30 - 80 tokens
    long_texts = []    # > 100 tokens

    # Scan more texts to find at least 512 of each category
    encs_sample = tokenizer.encode_batch(all_texts[:10000])
    for t, e in zip(all_texts[:10000], encs_sample):
        length = len(e.ids)
        if length <= 25 and len(short_texts) < 512:
            short_texts.append(t)
        elif 30 <= length <= 70 and len(medium_texts) < 512:
            medium_texts.append(t)
        elif length >= 120 and len(long_texts) < 512:
            long_texts.append(t)

    # If short reviews are slightly under 512 in the sample, repeat them to reach 512
    while len(short_texts) < 512:
        short_texts.extend(short_texts[:512 - len(short_texts)])

    print(f"Dataset pools prepared:")
    print(f"  Short Reviews (<25 tokens):    {len(short_texts)} items (Avg tokens: {np.mean([len(tokenizer.encode(t).ids) for t in short_texts[:20]]):.1f})")
    print(f"  Medium Reviews (30-70 tokens): {len(medium_texts)} items (Avg tokens: {np.mean([len(tokenizer.encode(t).ids) for t in medium_texts[:20]]):.1f})")
    print(f"  Long Reviews (>120 tokens):    {len(long_texts)} items (Avg tokens: {np.mean([len(tokenizer.encode(t).ids) for t in long_texts[:20]]):.1f})")

    review_types = {
        "short": short_texts,
        "medium": medium_texts,
        "long": long_texts,
    }

    batch_sizes = [8, 16, 32, 64, 128]
    full_report = {}

    for r_type, texts_pool in review_types.items():
        print(f"\n=======================================================")
        print(f"BENCHMARKING REVIEW TYPE: {r_type.upper()}")
        print(f"=======================================================")
        full_report[r_type] = {}

        # Establish baseline predictions with batch size 8 for consistency check
        encs_all = tokenizer.encode_batch(texts_pool)
        baseline_preds = []

        for bs in batch_sizes:
            # Length bucketing within batch
            sorted_indices = sorted(range(len(texts_pool)), key=lambda idx: len(encs_all[idx].ids))
            total_batches = len(texts_pool) // bs
            eval_indices = sorted_indices[: total_batches * bs]

            batch_latencies = []
            preds_this_bs = [None] * len(eval_indices)

            t_start = time.perf_counter()
            t_cpu_start = time.process_time()

            for b in range(total_batches):
                chunk_idx = eval_indices[b * bs : (b + 1) * bs]
                chunk_encs = [encs_all[i] for i in chunk_idx]
                max_l = max(len(e.ids) for e in chunk_encs)

                ids = np.zeros((bs, max_l), dtype=np.int64)
                mask = np.zeros((bs, max_l), dtype=np.int64)
                for j, e in enumerate(chunk_encs):
                    ids[j, :len(e.ids)] = e.ids
                    mask[j, :len(e.attention_mask)] = e.attention_mask

                t0 = time.perf_counter()
                logits = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
                t1 = time.perf_counter()
                batch_latencies.append((t1 - t0) * 1000)

                preds = np.argmax(logits, axis=-1)
                for j, p in enumerate(preds):
                    preds_this_bs[b * bs + j] = int(p)

            t_total = time.perf_counter() - t_start
            t_cpu_total = time.process_time() - t_cpu_start

            rps = len(eval_indices) / t_total if t_total > 0 else 0
            cpu_util = (t_cpu_total / t_total) * 100 if t_total > 0 else 0
            avg_lat = np.mean(batch_latencies)
            p50_lat = np.percentile(batch_latencies, 50)
            p95_lat = np.percentile(batch_latencies, 95)
            rss, peak_rss, priv = get_os_memory_info()

            if bs == 8:
                baseline_preds = list(preds_this_bs)
                consistency_pct = 100.0
            else:
                # Compare against baseline predictions for same evaluated subset
                overlap_len = min(len(baseline_preds), len(preds_this_bs))
                matches = sum(1 for i in range(overlap_len) if baseline_preds[i] == preds_this_bs[i])
                consistency_pct = (matches / overlap_len) * 100.0 if overlap_len > 0 else 100.0

            entry = {
                "batch_size": bs,
                "records_evaluated": len(eval_indices),
                "throughput_rps": round(rps, 1),
                "avg_batch_latency_ms": round(avg_lat, 2),
                "p50_latency_ms": round(p50_lat, 2),
                "p95_latency_ms": round(p95_lat, 2),
                "cpu_utilization_pct": round(cpu_util, 1),
                "peak_rss_mb": peak_rss,
                "private_bytes_mb": priv,
                "prediction_consistency_pct": round(consistency_pct, 2)
            }
            full_report[r_type][f"batch_{bs}"] = entry
            print(f"BS {bs:3d} | RPS: {rps:6.1f} | Avg Lat: {avg_lat:7.2f}ms | P50: {p50_lat:7.2f}ms | P95: {p95_lat:7.2f}ms | Peak RSS: {peak_rss:6.1f}MB | CPU: {cpu_util:5.1f}% | Consist: {consistency_pct:.1f}%")

    out_file = "batch_size_reevaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved batch size re-evaluation results to {out_file}")
    return full_report

if __name__ == "__main__":
    run_evaluation()
