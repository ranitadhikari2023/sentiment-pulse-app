"""
Phase 2.2 Independent Ground-Truth Evaluation Script.
Evaluates the INT8 quantized ONNX model against true ground-truth labels from data/test.csv.
Computes:
1. True Confusion Matrix (True Label vs Predicted Label).
2. Per-Class Precision, Recall, and F1.
3. Overall Accuracy and Macro-F1.
4. 95% Confidence Intervals for Accuracy using the Wilson score and normal approximation.
5. Statistical significance comparison with FP32 reference baseline (78.67%).
Saves results to data/phase2_2_ground_truth_evaluation.json.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import onnxruntime as ort
from tokenizers import Tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV = os.path.join(BASE_DIR, "data", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
OUTPUT_JSON = os.path.join(BASE_DIR, "data", "phase2_2_ground_truth_evaluation.json")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
CLASS_NAMES = ["negative", "neutral", "positive"]


def wilson_score_interval(p: float, n: int, confidence: float = 0.95) -> tuple[float, float]:
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
    upper = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
    return max(0.0, lower), min(1.0, upper)


def run_ground_truth_evaluation(sample_limit: int = 1500):
    print(f"Loading ground truth test dataset from {TEST_CSV}...", flush=True)
    df = pd.read_csv(TEST_CSV).dropna(subset=["text", "label"])
    if sample_limit and sample_limit < len(df):
        print(f"Sampling {sample_limit:,} representative test rows (random_state=42)...", flush=True)
        df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)

    total_samples = len(df)
    print(f"Total test samples: {total_samples:,}", flush=True)

    # Map true labels
    y_true = df["label"].map(LABEL2ID).to_numpy()

    print(f"Loading ONNX model from {MODEL_PATH}...", flush=True)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 2)
    session = ort.InferenceSession(MODEL_PATH, sess_opts, providers=["CPUExecutionProvider"])
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    batch_size = 32
    texts = df["text"].tolist()
    y_pred = []

    print(f"Running batched inference (batch size: {batch_size})...", flush=True)
    total_batches = (total_samples + batch_size - 1) // batch_size
    for b in range(total_batches):
        start_idx = b * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_texts = texts[start_idx:end_idx]
        encodings = tokenizer.encode_batch(batch_texts)

        max_len = min(256, max(len(e.ids) for e in encodings))
        input_ids = np.zeros((len(batch_texts), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(batch_texts), max_len), dtype=np.int64)

        for j, enc in enumerate(encodings):
            truncated_ids = enc.ids[:max_len]
            truncated_mask = enc.attention_mask[:max_len]
            input_ids[j, :len(truncated_ids)] = truncated_ids
            attention_mask[j, :len(truncated_mask)] = truncated_mask

        ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = session.run(None, ort_inputs)[0]
        preds = np.argmax(logits, axis=-1)
        y_pred.extend(preds.tolist())

        if (b + 1) % 10 == 0 or (b + 1) == total_batches:
            print(f"  Batch {b + 1}/{total_batches} ({end_idx}/{total_samples} samples)", flush=True)

    y_pred = np.array(y_pred)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    precisions = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    recalls = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    ci_lower, ci_upper = wilson_score_interval(acc, total_samples, confidence=0.95)

    # Statistical significance vs FP32 baseline (0.7867)
    fp32_acc = 0.7867
    diff = acc - fp32_acc
    se_diff = math.sqrt((acc * (1 - acc) + fp32_acc * (1 - fp32_acc)) / total_samples)
    z_stat = diff / se_diff if se_diff > 0 else 0.0
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_stat) / math.sqrt(2))))

    results = {
        "dataset": "yelp_review_full (data/test.csv)",
        "total_samples": total_samples,
        "model_architecture": "DistilBERT INT8 Quantized ONNX",
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "accuracy_95_ci": [round(ci_lower * 100, 2), round(ci_upper * 100, 2)],
        "macro_f1": round(macro_f1, 4),
        "macro_f1_pct": round(macro_f1 * 100, 2),
        "fp32_reference_accuracy": fp32_acc,
        "accuracy_delta_pct": round(diff * 100, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": round(p_value, 4),
        "statistically_significant_difference": bool(p_value < 0.05),
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": cm.tolist(),
            "description": "Rows represent True Labels, columns represent Predicted Labels"
        },
        "per_class_metrics": {
            cls: {
                "precision": round(float(precisions[idx]), 4),
                "recall": round(float(recalls[idx]), 4),
                "f1_score": round(float(f1s[idx]), 4),
                "support": int(np.sum(y_true == idx))
            }
            for idx, cls in enumerate(CLASS_NAMES)
        }
    }

    print("\n" + "=" * 70)
    print("PHASE 2.2 GROUND TRUTH EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy:  {acc * 100:.2f}% (95% CI: [{ci_lower * 100:.2f}%, {ci_upper * 100:.2f}%])")
    print(f"Macro-F1 Score:    {macro_f1 * 100:.2f}%")
    print(f"FP32 Baseline:     {fp32_acc * 100:.2f}% (Delta: {diff * 100:+.2f}%)")
    print(f"p-value:           {p_value:.4f} (Statistically Significant: {p_value < 0.05})")
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    print(f"{'':>10} | {'Negative':>10} | {'Neutral':>10} | {'Positive':>10}")
    print("-" * 46)
    for idx, cls in enumerate(CLASS_NAMES):
        row = cm[idx]
        print(f"{cls:>10} | {row[0]:>10} | {row[1]:>10} | {row[2]:>10}")

    print("\nPer-Class Performance:")
    for idx, cls in enumerate(CLASS_NAMES):
        print(f"  {cls:<9} -> Precision: {precisions[idx]:.4f} | Recall: {recalls[idx]:.4f} | F1: {f1s[idx]:.4f} (Support: {np.sum(y_true == idx):,})")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults successfully exported to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_ground_truth_evaluation()
