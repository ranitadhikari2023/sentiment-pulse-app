"""
Benchmark the impact of:
1. Dynamic Padding alone (unbucketed)
2. Length-Sorted Bucketing
3. Multi-Threaded Worker Execution (ONNX session thread pool)
4. Duplicate Prediction Caching
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
from core.config import settings
from scripts.measure_memory import get_os_memory_info

def run():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(8, max(2, os.cpu_count() or 2))
    session = ort.InferenceSession(settings.MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)

    df = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])
    eval_texts = df["text"].head(1024).tolist()

    # 1. Baseline: Unbucketed Sequential Batch 32
    print("--- 1. Unbucketed Sequential Batch 32 ---")
    t0 = time.perf_counter()
    bs = 32
    for i in range(0, len(eval_texts), bs):
        chunk = eval_texts[i : i + bs]
        encs = tokenizer.encode_batch(chunk)
        max_l = max(len(e.ids) for e in encs)
        ids = np.zeros((len(chunk), max_l), dtype=np.int64)
        mask = np.zeros((len(chunk), max_l), dtype=np.int64)
        for j, e in enumerate(encs):
            ids[j, :len(e.ids)] = e.ids
            mask[j, :len(e.attention_mask)] = e.attention_mask
        _ = session.run(None, {"input_ids": ids, "attention_mask": mask})
    t_unbucketed = time.perf_counter() - t0
    rps_unbucketed = len(eval_texts) / t_unbucketed
    print(f"Unbucketed Batch 32: {rps_unbucketed:.2f} rps ({t_unbucketed:.2f}s)")

    # 2. Length-Sorted Bucketed Sequential Batch 32
    print("--- 2. Length-Sorted Bucketed Sequential Batch 32 ---")
    t0 = time.perf_counter()
    # Pre-encode to get lengths
    encs_all = tokenizer.encode_batch(eval_texts)
    sorted_indices = sorted(range(len(eval_texts)), key=lambda idx: len(encs_all[idx].ids))
    for i in range(0, len(eval_texts), bs):
        batch_idx = sorted_indices[i : i + bs]
        batch_encs = [encs_all[idx] for idx in batch_idx]
        max_l = max(len(e.ids) for e in batch_encs)
        ids = np.zeros((len(batch_idx), max_l), dtype=np.int64)
        mask = np.zeros((len(batch_idx), max_l), dtype=np.int64)
        for j, e in enumerate(batch_encs):
            ids[j, :len(e.ids)] = e.ids
            mask[j, :len(e.attention_mask)] = e.attention_mask
        _ = session.run(None, {"input_ids": ids, "attention_mask": mask})
    t_bucketed = time.perf_counter() - t0
    rps_bucketed = len(eval_texts) / t_bucketed
    print(f"Bucketed Batch 32: {rps_bucketed:.2f} rps ({t_bucketed:.2f}s, speedup: {rps_bucketed/rps_unbucketed:.2f}x)")

    # 3. Length-Sorted Bucketed + Multi-Threaded Parallel Execution (4 workers)
    print("--- 3. Length-Sorted Bucketed + Parallel Threads (4 workers) ---")
    def run_sub_batch(sub_encs):
        max_l = max(len(e.ids) for e in sub_encs)
        sub_bs = len(sub_encs)
        ids = np.zeros((sub_bs, max_l), dtype=np.int64)
        mask = np.zeros((sub_bs, max_l), dtype=np.int64)
        for j, e in enumerate(sub_encs):
            ids[j, :len(e.ids)] = e.ids
            mask[j, :len(e.attention_mask)] = e.attention_mask
        return session.run(None, {"input_ids": ids, "attention_mask": mask})

    t0 = time.perf_counter()
    encs_all = tokenizer.encode_batch(eval_texts)
    sorted_indices = sorted(range(len(eval_texts)), key=lambda idx: len(encs_all[idx].ids))
    batches = []
    for i in range(0, len(eval_texts), bs):
        batch_idx = sorted_indices[i : i + bs]
        batches.append([encs_all[idx] for idx in batch_idx])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run_sub_batch, batches))
    t_parallel = time.perf_counter() - t0
    rps_parallel = len(eval_texts) / t_parallel
    print(f"Bucketed + Parallel (4 workers): {rps_parallel:.2f} rps ({t_parallel:.2f}s, speedup over baseline: {rps_parallel/rps_unbucketed:.2f}x)")

    # 4. Realistic Duplicate Rate (20% duplicate text) + Cache
    print("--- 4. With 20% Duplicates + Cache Hit ---")
    cache = {}
    cached_hits = 0
    t0 = time.perf_counter()
    # Simulate realistic customer review corpus where 20% are common expressions
    sample_corpus = eval_texts[:800] + eval_texts[:200] + eval_texts[:24]
    
    uncached = []
    uncached_idx = []
    for idx, txt in enumerate(sample_corpus):
        if txt in cache:
            cached_hits += 1
        else:
            uncached.append(txt)
            uncached_idx.append(idx)

    # Process uncached with bucketed parallel
    encs_uncached = tokenizer.encode_batch(uncached)
    sorted_uncached = sorted(range(len(uncached)), key=lambda i: len(encs_uncached[i].ids))
    batches_uncached = []
    for i in range(0, len(uncached), bs):
        b_idx = sorted_uncached[i : i + bs]
        batches_uncached.append([encs_uncached[idx] for idx in b_idx])

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run_sub_batch, batches_uncached))
    t_cached = time.perf_counter() - t0
    rps_cached = len(sample_corpus) / t_cached
    print(f"With Cache & 20% duplicates: {rps_cached:.2f} rps (cached hits: {cached_hits}/{len(sample_corpus)})")

if __name__ == "__main__":
    run()

