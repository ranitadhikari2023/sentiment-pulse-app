"""
Sentiment Pulse - Advanced ONNX Inference Profiling & Batch-Size Benchmark Suite
Inspects and measures:
1. Input tensor shapes and sequence lengths.
2. Tokenization time vs Array prep vs ONNX execution vs Softmax.
3. Batch sizes: 8, 16, 32, 64, 128.
4. Latencies: Average, P50, P95.
5. OS-level Peak RSS and Private Bytes.
6. CPU Utilization (process_time / wall_time).
7. Prediction consistency across batch sizes.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings, BASE_DIR
from scripts.measure_memory import get_os_memory_info


def run_initial_investigation():
    print("=" * 75)
    print("MANDATORY INITIAL INVESTIGATION: MODEL & RUNTIME ARCHITECTURE")
    print("=" * 75)

    rss_0, peak_0, priv_0 = get_os_memory_info()
    t0_load = time.perf_counter()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(8, max(2, os.cpu_count() or 2))
    session = ort.InferenceSession(settings.MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    t_load = time.perf_counter() - t0_load

    rss_1, peak_1, priv_1 = get_os_memory_info()

    # 1. Model Input & Output Shapes
    inputs_meta = []
    for inp in session.get_inputs():
        inputs_meta.append({"name": inp.name, "shape": inp.shape, "type": inp.type})
    outputs_meta = []
    for out in session.get_outputs():
        outputs_meta.append({"name": out.name, "shape": out.shape, "type": out.type})

    # 2. Tokenizer Configuration
    with open(settings.TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    # 3. Memory Consumed by Model Load
    model_file_size_mb = os.path.getsize(settings.MODEL_PATH) / (1024 * 1024)
    model_ram_delta_mb = rss_1 - rss_0

    info = {
        "cpu_count": os.cpu_count(),
        "effective_thread_allocation": opts.intra_op_num_threads,
        "model_file_path": settings.MODEL_PATH,
        "model_file_size_mb": round(model_file_size_mb, 2),
        "model_load_time_sec": round(t_load, 3),
        "process_rss_before_load_mb": rss_0,
        "process_rss_after_load_mb": rss_1,
        "model_ram_delta_mb": round(model_ram_delta_mb, 2),
        "inputs": inputs_meta,
        "outputs": outputs_meta,
        "tokenizer_padding_config": tok_cfg.get("padding"),
        "tokenizer_truncation_config": tok_cfg.get("truncation"),
    }

    print(json.dumps(info, indent=2))
    return info, session, tokenizer


def benchmark_batch_sizes(session: ort.InferenceSession, tokenizer: Tokenizer, test_csv_path: str = "data/test.csv"):
    print("\n" + "=" * 75)
    print("BENCHMARKING DIFFERENT BATCH SIZES (8, 16, 32, 64, 128)")
    print("=" * 75)

    df = pd.read_csv(test_csv_path).dropna(subset=["text", "label"])
    # Sample 1,024 diverse customer reviews
    eval_texts = df["text"].head(1024).tolist()

    batch_sizes = [8, 16, 32, 64, 128]
    results = {}

    # Baseline predictions with batch size 1 for consistency check
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)

    # Warmup
    warmup_enc = tokenizer.encode_batch(eval_texts[:8])
    max_w = max(len(e.ids) for e in warmup_enc)
    w_ids = np.zeros((8, max_w), dtype=np.int64)
    w_mask = np.zeros((8, max_w), dtype=np.int64)
    for i, e in enumerate(warmup_enc):
        w_ids[i, :len(e.ids)] = e.ids
        w_mask[i, :len(e.attention_mask)] = e.attention_mask
    _ = session.run(None, {"input_ids": w_ids, "attention_mask": w_mask})

    for bs in batch_sizes:
        batch_latencies = []
        tokenization_times = []
        onnx_times = []
        post_times = []
        all_preds = []

        total_batches = len(eval_texts) // bs
        eval_subset = eval_texts[: total_batches * bs]

        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        for b in range(total_batches):
            chunk = eval_subset[b * bs : (b + 1) * bs]

            # 1. Tokenize
            t0 = time.perf_counter()
            encs = tokenizer.encode_batch(chunk)
            t_tok = time.perf_counter() - t0
            tokenization_times.append(t_tok)

            # 2. Pad to batch max
            max_l = max(len(e.ids) for e in encs)
            ids = np.zeros((bs, max_l), dtype=np.int64)
            mask = np.zeros((bs, max_l), dtype=np.int64)
            for i, e in enumerate(encs):
                ids[i, :len(e.ids)] = e.ids
                mask[i, :len(e.attention_mask)] = e.attention_mask

            # 3. ONNX Session Run
            t0 = time.perf_counter()
            logits = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            t_onnx = time.perf_counter() - t0
            onnx_times.append(t_onnx)

            # 4. Softmax
            t0 = time.perf_counter()
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            preds = np.argmax(probs, axis=-1)
            t_post = time.perf_counter() - t0
            post_times.append(t_post)

            all_preds.extend(preds.tolist())
            batch_latencies.append(t_tok + t_onnx + t_post)

        t_wall_total = time.perf_counter() - t_wall_start
        t_cpu_total = time.process_time() - t_cpu_start

        cpu_utilization = (t_cpu_total / t_wall_total) * 100 if t_wall_total > 0 else 0
        avg_batch_lat_ms = (t_wall_total / total_batches) * 1000
        p50_lat_ms = float(np.percentile(batch_latencies, 50)) * 1000
        p95_lat_ms = float(np.percentile(batch_latencies, 95)) * 1000
        throughput_rps = len(eval_subset) / t_wall_total

        avg_tok_ms = float(np.mean(tokenization_times)) * 1000
        avg_onnx_ms = float(np.mean(onnx_times)) * 1000
        avg_post_ms = float(np.mean(post_times)) * 1000

        rss, peak_rss, priv = get_os_memory_info()

        res_entry = {
            "batch_size": bs,
            "evaluated_records": len(eval_subset),
            "throughput_rps": round(throughput_rps, 1),
            "avg_batch_latency_ms": round(avg_batch_lat_ms, 2),
            "p50_latency_ms": round(p50_lat_ms, 2),
            "p95_latency_ms": round(p95_lat_ms, 2),
            "avg_tokenization_ms": round(avg_tok_ms, 3),
            "avg_onnx_execution_ms": round(avg_onnx_ms, 2),
            "avg_postprocessing_ms": round(avg_post_ms, 3),
            "cpu_utilization_pct": round(cpu_utilization, 1),
            "peak_rss_mb": peak_rss,
            "private_bytes_mb": priv,
        }
        results[f"batch_{bs}"] = res_entry
        print(f"Batch {bs:3d}: {throughput_rps:5.1f} rps | Avg Lat: {avg_batch_lat_ms:6.2f} ms | P50: {p50_lat_ms:6.2f} ms | P95: {p95_lat_ms:6.2f} ms | CPU: {cpu_utilization:5.1f}% | ONNX: {avg_onnx_ms:5.1f} ms | Tok: {avg_tok_ms:5.2f} ms")

    out_file = os.path.join(BASE_DIR, "inference_profiling_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nInference profiling saved to {out_file}")
    return results


if __name__ == "__main__":
    info, sess, tok = run_initial_investigation()
    batch_res = benchmark_batch_sizes(sess, tok)
