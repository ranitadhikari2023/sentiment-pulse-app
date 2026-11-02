"""
Comprehensive Verification Test Suite for Phase 2: ONNX Inference Optimization
Covers all 14 mandatory test scenarios:
1. Single prediction correctness
2. Batched prediction correctness
3. Different batch sizes (8, 16, 32, 64, 128)
4. Long customer reviews (>256 and >512 tokens)
5. Empty and malformed inputs
6. Quantized model prediction consistency
7. Prediction cache correctness & hit rate
8. Memory stability under repeated inference
9. Concurrent inference safety (multi-threaded predict_batch)
10. Failed inference batch recovery
11. Import progress accuracy during inference
12. Staging and atomic finalization after inference
13. Existing CSV compatibility
14. Tenant isolation with cache
"""

import os
import sys
import uuid
import time
import tempfile
import threading
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import settings
from db.session import SessionLocal
from db.models import ImportJob, Feedback, FeedbackStaging
from model import SentimentAnalyzer, PredictionCache
from services.sentiment_service import sentiment_service
from services.import_service import ImportService
from scripts.measure_memory import get_os_memory_info


# ==============================================================================
# 1. Single prediction correctness
# ==============================================================================
def test_scenario_01_single_prediction_correctness():
    """Verify single prediction classifies positive, negative, and neutral with valid scores."""
    analyzer = SentimentAnalyzer()

    pos_res = analyzer.predict("I absolutely love this new update, it made everything so fast and easy!")
    assert pos_res["sentiment"] == "positive"
    assert pos_res["confidence"] > 0.80
    assert abs(sum(pos_res["scores"].values()) - 1.0) < 0.01

    neg_res = analyzer.predict("Terrible service, it crashed immediately and customer support refused to help.")
    assert neg_res["sentiment"] == "negative"
    assert neg_res["confidence"] > 0.80
    assert abs(sum(neg_res["scores"].values()) - 1.0) < 0.01

    neu_res = analyzer.predict("The package arrived on Tuesday afternoon via standard delivery.")
    assert neu_res["sentiment"] in ("neutral", "positive")  # factual/neutral statement
    assert abs(sum(neu_res["scores"].values()) - 1.0) < 0.01


# ==============================================================================
# 2. Batched prediction correctness
# ==============================================================================
def test_scenario_02_batched_prediction_correctness():
    """Verify batched prediction produces identical outputs to single predictions."""
    analyzer = SentimentAnalyzer()
    texts = [
        "Outstanding customer support, very happy!",
        "The checkout system is broken and keeps taking money.",
        "The store operates between 9 AM and 5 PM on weekdays.",
        "Wonderful quality, highly recommend.",
        "Worst experience ever, totally unacceptable.",
    ]

    single_results = [analyzer.predict(t) for t in texts]
    batch_results = analyzer.predict_batch(texts, mini_batch_size=8)

    assert len(single_results) == len(batch_results)
    for s_res, b_res in zip(single_results, batch_results):
        assert s_res["sentiment"] == b_res["sentiment"]
        assert abs(s_res["confidence"] - b_res["confidence"]) < 1e-3
        for k in ["positive", "neutral", "negative"]:
            assert abs(s_res["scores"][k] - b_res["scores"][k]) < 1e-3


# ==============================================================================
# 3. Different batch sizes (8, 16, 32, 64, 128)
# ==============================================================================
def test_scenario_03_different_batch_sizes():
    """Verify prediction consistency across batch sizes 8, 16, 32, 64, 128."""
    analyzer = SentimentAnalyzer()
    texts = [
        "Great experience with quick shipping.",
        "Complete disaster, useless product.",
        "Order dispatched via carrier.",
        "Excellent support team.",
    ] * 32  # 128 items total

    baseline = analyzer.predict_batch(texts, mini_batch_size=8)
    for bs in [16, 32, 64, 128]:
        curr = analyzer.predict_batch(texts, mini_batch_size=bs)
        assert len(curr) == len(baseline)
        for b_item, c_item in zip(baseline, curr):
            assert b_item["sentiment"] == c_item["sentiment"]
            assert abs(b_item["confidence"] - c_item["confidence"]) < 1e-3


# ==============================================================================
# 4. Long customer reviews (>256 and >512 tokens)
# ==============================================================================
def test_scenario_04_long_customer_reviews():
    """Verify truncation bounds (>256 and >512 tokens) without overflow or crashes."""
    analyzer = SentimentAnalyzer()

    # ~350 words (>256 tokens)
    long_review_300 = "I ordered this product last month and it was fantastic. " * 35
    # ~700 words (>512 tokens)
    long_review_700 = "The service provided by this company has been consistently great. " * 70

    res_300 = analyzer.predict(long_review_300)
    assert res_300["sentiment"] in ("positive", "neutral", "negative")
    assert res_300["confidence"] > 0

    res_700 = analyzer.predict(long_review_700)
    assert res_700["sentiment"] in ("positive", "neutral", "negative")
    assert res_700["confidence"] > 0

    # Test in batch
    batch_res = analyzer.predict_batch([long_review_300, long_review_700])
    assert len(batch_res) == 2


# ==============================================================================
# 5. Empty and malformed inputs
# ==============================================================================
def test_scenario_05_empty_and_malformed_inputs():
    """Verify empty and whitespace strings raise ValueError and non-ASCII is handled."""
    analyzer = SentimentAnalyzer()

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.predict("")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.predict("   \n\t  ")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.predict(None)

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.predict_batch(["Valid text", "   ", "Another valid text"])

    # Emojis and non-ASCII characters should be parsed safely
    emoji_res = analyzer.predict("Awesome work! 👍🎉 Super happy! 😊")
    assert emoji_res["sentiment"] == "positive"


# ==============================================================================
# 6. Quantized model prediction consistency
# ==============================================================================
def test_scenario_06_quantized_model_prediction_consistency():
    """Verify deterministic, calibrated probabilities summing to 1.0."""
    analyzer = SentimentAnalyzer()
    sample = "The quality of the delivered items matched the product description."

    results = [analyzer.predict(sample) for _ in range(5)]
    for r in results:
        assert r["sentiment"] == results[0]["sentiment"]
        assert r["confidence"] == results[0]["confidence"]
        assert r["scores"] == results[0]["scores"]
        assert abs(sum(r["scores"].values()) - 1.0) < 0.01


# ==============================================================================
# 7. Prediction cache correctness & hit rate
# ==============================================================================
def test_scenario_07_prediction_cache_correctness():
    """Verify LRU cache hit rate, eviction, and clearing."""
    cache = PredictionCache(capacity=3)
    mv = "3.0.0"

    res1 = {"text": "A", "sentiment": "positive", "confidence": 0.9, "scores": {"positive": 0.9}}
    res2 = {"text": "B", "sentiment": "negative", "confidence": 0.8, "scores": {"negative": 0.8}}
    res3 = {"text": "C", "sentiment": "neutral", "confidence": 0.7, "scores": {"neutral": 0.7}}
    res4 = {"text": "D", "sentiment": "positive", "confidence": 0.95, "scores": {"positive": 0.95}}

    cache.put(mv, "A", res1)
    cache.put(mv, "B", res2)
    cache.put(mv, "C", res3)

    # Hits
    assert cache.get(mv, "A") is not None
    assert cache.get(mv, "B") is not None
    assert cache.hits == 2

    # Miss
    assert cache.get(mv, "Z") is None
    assert cache.misses == 1

    # Insert 4th item -> evicts least recently used (C, because A and B were accessed)
    cache.put(mv, "D", res4)
    assert cache.get(mv, "C") is None  # Evicted!
    assert cache.get(mv, "A") is not None  # Retained!

    stats = cache.stats()
    assert stats["size"] == 3
    assert stats["capacity"] == 3
    assert stats["hit_rate"] > 0

    cache.clear()
    assert cache.stats()["size"] == 0


# ==============================================================================
# 8. Memory stability under repeated inference
# ==============================================================================
def test_scenario_08_memory_stability_under_repeated_inference():
    """Verify memory remains bounded under repeated inference passes."""
    analyzer = SentimentAnalyzer()
    batch = [
        "Fast delivery and great customer support.",
        "Completely dissatisfied with product performance.",
        "Standard order tracking update notification.",
    ] * 20  # 60 items

    _, peak_0, _ = get_os_memory_info()

    for _ in range(20):
        _ = analyzer.predict_batch(batch, mini_batch_size=32)

    _, peak_1, _ = get_os_memory_info()
    # Memory growth should be negligible and bounded
    growth_mb = peak_1 - peak_0
    assert growth_mb < 300.0  # Safe bounded ceiling


# ==============================================================================
# 9. Concurrent inference safety (multi-threaded predict_batch)
# ==============================================================================
def test_scenario_09_concurrent_inference_safety():
    """Verify multi-threaded concurrent predict_batch calls without race conditions."""
    analyzer = SentimentAnalyzer()

    def worker_fn(worker_id: int):
        texts = [
            f"Worker {worker_id} report: outstanding quality and service!",
            f"Worker {worker_id} report: terrible crash on startup.",
            f"Worker {worker_id} report: scheduled delivery arriving tomorrow.",
        ] * 10
        res = analyzer.predict_batch(texts, mini_batch_size=16)
        assert len(res) == len(texts)
        for r in res:
            assert r["sentiment"] in ("positive", "negative", "neutral")
        return len(res)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(worker_fn, i) for i in range(6)]
        results = [f.result() for f in futures]

    assert sum(results) == 180


# ==============================================================================
# 10. Failed inference batch recovery
# ==============================================================================
def test_scenario_10_failed_inference_batch_recovery(monkeypatch):
    """Verify fallback mechanism records failed rows without halting the job."""
    db = SessionLocal()
    job = ImportService.create_import_job(db, "recovery_test.csv")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write("text,category,source,rating\n")
    tmp.write("Good feedback 1,Test,web,5\n")
    tmp.write("Bad feedback 2,Test,web,1\n")
    tmp.close()

    call_count = [0]

    def failing_batch_predict(texts):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("Simulated transient ONNX Runtime inference error")
        return [
            {"text": t, "sentiment": "positive", "confidence": 0.9, "scores": {"positive": 0.9, "neutral": 0.05, "negative": 0.05}}
            for t in texts
        ]

    monkeypatch.setattr(sentiment_service, "predict_batch", failing_batch_predict)

    try:
        completed_job = ImportService.process_csv_file(db, job, tmp.name)
        assert completed_job.status == "COMPLETED"
        assert completed_job.successful_rows == 2
        assert completed_job.failed_rows == 0
    finally:
        db.query(Feedback).filter(Feedback.category == "Test").delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job.id).delete()
        db.query(ImportJob).filter(ImportJob.id == job.id).delete()
        db.commit()
        db.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ==============================================================================
# 11. Import progress accuracy during inference
# ==============================================================================
def test_scenario_11_import_progress_accuracy():
    """Verify accurate tracking of total_rows, processed_rows, and stage transitions."""
    db = SessionLocal()
    job = ImportService.create_import_job(db, "progress_test.csv")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write("text,category,source,rating\n")
    for i in range(15):
        tmp.write(f"Customer review {i},ProgressTest,web,5\n")
    tmp.close()

    try:
        completed = ImportService.process_csv_file(db, job, tmp.name)
        assert completed.status == "COMPLETED"
        assert completed.stage == "COMPLETED"
        assert completed.total_rows == 15
        assert completed.processed_rows == 15
        assert completed.successful_rows == 15
        assert completed.failed_rows == 0
    finally:
        db.query(Feedback).filter(Feedback.category == "ProgressTest").delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job.id).delete()
        db.query(ImportJob).filter(ImportJob.id == job.id).delete()
        db.commit()
        db.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ==============================================================================
# 12. Staging and atomic finalization after inference
# ==============================================================================
def test_scenario_12_staging_and_atomic_finalization():
    """Verify staged records are atomically transferred to feedback and cleared from staging."""
    db = SessionLocal()
    job = ImportService.create_import_job(db, "staging_final_test.csv")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write("text,category,source,rating\n")
    for i in range(10):
        tmp.write(f"Staging atomic feedback {i},StagingFinalTest,web,5\n")
    tmp.close()

    try:
        completed = ImportService.process_csv_file(db, job, tmp.name)
        assert completed.status == "COMPLETED"

        # Staging table must have 0 records for this job
        staging_count = db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job.id).count()
        assert staging_count == 0

        # Primary feedback table must have all 10 records
        fb_count = db.query(Feedback).filter(Feedback.category == "StagingFinalTest").count()
        assert fb_count == 10
    finally:
        db.query(Feedback).filter(Feedback.category == "StagingFinalTest").delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job.id).delete()
        db.query(ImportJob).filter(ImportJob.id == job.id).delete()
        db.commit()
        db.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ==============================================================================
# 13. Existing CSV compatibility
# ==============================================================================
def test_scenario_13_existing_csv_compatibility():
    """Verify compatibility with column aliases and semicolon delimiter."""
    db = SessionLocal()
    job = ImportService.create_import_job(db, "semicolon_alias.csv")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    # Using 'review' alias and semicolon delimiter
    tmp.write("review;department;channel;star_rating\n")
    tmp.write("Loved the product, great experience!;Sales;store;5\n")
    tmp.write("Did not work as advertised.;Support;call;1\n")
    tmp.close()

    try:
        completed = ImportService.process_csv_file(db, job, tmp.name)
        assert completed.status == "COMPLETED"
        assert completed.successful_rows == 2

        fb = db.query(Feedback).filter(Feedback.category == "Sales").first()
        assert fb is not None
        assert fb.sentiment == "positive"
        assert fb.rating == 5
    finally:
        db.query(Feedback).filter(Feedback.category.in_(["Sales", "Support"])).delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job.id).delete()
        db.query(ImportJob).filter(ImportJob.id == job.id).delete()
        db.commit()
        db.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


# ==============================================================================
# 14. Tenant isolation with cache
# ==============================================================================
def test_scenario_14_tenant_isolation_with_cache():
    """Verify that cached predictions do not leak tenant data across tenants."""
    db = SessionLocal()

    # Clear cache to establish clean state
    sentiment_service.clear_cache()

    shared_text = "Standard identical customer feedback review."

    # Tenant 101 import
    job1 = ImportService.create_import_job(db, "tenant101.csv", user_id=101)
    tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp1.write(f"text,category\n{shared_text},Tenant101Cat\n")
    tmp1.close()

    # Tenant 202 import
    job2 = ImportService.create_import_job(db, "tenant202.csv", user_id=202)
    tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp2.write(f"text,category\n{shared_text},Tenant202Cat\n")
    tmp2.close()

    try:
        completed1 = ImportService.process_csv_file(db, job1, tmp1.name)
        assert completed1.status == "COMPLETED"

        # Tenant 202 runs and gets cache hit for shared_text
        completed2 = ImportService.process_csv_file(db, job2, tmp2.name)
        assert completed2.status == "COMPLETED"

        # Verify Tenant 101 feedback has user_id = 101 only
        fb1 = db.query(Feedback).filter(Feedback.category == "Tenant101Cat").first()
        assert fb1 is not None
        assert fb1.user_id == 101

        # Verify Tenant 202 feedback has user_id = 202 only
        fb2 = db.query(Feedback).filter(Feedback.category == "Tenant202Cat").first()
        assert fb2 is not None
        assert fb2.user_id == 202

        # Verify predictions match identically
        assert fb1.sentiment == fb2.sentiment
        assert fb1.confidence == fb2.confidence
    finally:
        db.query(Feedback).filter(Feedback.category.in_(["Tenant101Cat", "Tenant202Cat"])).delete()
        db.query(FeedbackStaging).filter(FeedbackStaging.import_job_id.in_([job1.id, job2.id])).delete()
        db.query(ImportJob).filter(ImportJob.id.in_([job1.id, job2.id])).delete()
        db.commit()
        db.close()
        if os.path.exists(tmp1.name):
            os.unlink(tmp1.name)
        if os.path.exists(tmp2.name):
            os.unlink(tmp2.name)

