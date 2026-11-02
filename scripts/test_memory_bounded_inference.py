"""
Test memory containment under controlled concurrency (Global Semaphore)
and compare Peak RSS with batch sizes 16, 32, 64 under repeated execution.
"""
import os
import sys
import gc
import time
import json
import threading
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.measure_memory import get_os_memory_info
from core.config import settings

def test_bounded_concurrency(batch_size=16, use_arena=True, num_threads=2):
    print(f"\n--- Testing Bounded Concurrency: BS={batch_size}, Arena={use_arena}, Threads={num_threads} ---")
    rss_start, peak_start, priv_start = get_os_memory_info()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = use_arena
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(settings.MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(settings.TOKENIZER_PATH)
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)

    df = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])
    texts = df["text"].head(512).tolist()

    # Global inference semaphore restricting active ONNX execution to 1 thread
    inference_sem = threading.Semaphore(1)

    def run_inference_worker(worker_id, worker_texts):
        encs = tokenizer.encode_batch(worker_texts)
        # Process in chunks of batch_size
        for i in range(0, len(worker_texts), batch_size):
            chunk_encs = encs[i:i+batch_size]
            max_l = max(len(e.ids) for e in chunk_encs)
            ids = np.zeros((len(chunk_encs), max_l), dtype=np.int64)
            mask = np.zeros((len(chunk_encs), max_l), dtype=np.int64)
            for j, e in enumerate(chunk_encs):
                ids[j, :len(e.ids)] = e.ids
                mask[j, :len(e.attention_mask)] = e.attention_mask

            # Acquire global inference semaphore so only 1 batch executes in ONNX at any instant
            with inference_sem:
                _ = session.run(None, {"input_ids": ids, "attention_mask": mask})

    # Simulate 4 concurrent workers (e.g. 2 concurrent imports + 2 background workers)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_inference_worker, w, texts[w*64:(w+1)*64])
            for w in range(4)
        ]
        for f in futures:
            f.result()
    elapsed = time.perf_counter() - t0

    rss_end, peak_end, priv_end = get_os_memory_info()
    rps = 256 / elapsed
    print(f"Elapsed: {elapsed:.2f}s | Throughput: {rps:.1f} rps")
    print(f"Start RSS: {rss_start:.1f} MB | End RSS: {rss_end:.1f} MB | Peak RSS: {peak_end:.1f} MB | Private: {priv_end:.1f} MB")
    return {"bs": batch_size, "arena": use_arena, "threads": num_threads, "elapsed": elapsed, "rps": rps, "peak_rss": peak_end}

if __name__ == "__main__":
    r1 = test_bounded_concurrency(batch_size=16, use_arena=True, num_threads=2)
    r2 = test_bounded_concurrency(batch_size=32, use_arena=True, num_threads=2)
    r3 = test_bounded_concurrency(batch_size=16, use_arena=False, num_threads=2)
    print("\nSummary:", [r1, r2, r3])

