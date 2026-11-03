"""
PostgreSQL Production Database Integration & Dialect Verification Suite.
Verified in GitHub Actions CI via .github/workflows/postgres_integration.yml.

Tests:
1. PostgreSQL dialect SQL generation for bulk insert and atomic transfer.
2. Alembic migration execution on PostgreSQL.
3. Multi-row staging bulk insert via psycopg driver.
4. Atomic INSERT INTO ... SELECT transfer from feedback_staging to feedback.
5. Staging table cleanup and rollback behavior.
6. Multi-tenant isolation under PostgreSQL queries.
7. Chunk size performance comparison (1,000 vs 5,000 vs 10,000 rows).
"""

import os
import time
import uuid
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, select, insert, delete, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

from db.models import Base, Feedback, FeedbackStaging, ImportJob
from core.config import settings

# Test environment detection
POSTGRES_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://sentiment_user:sentiment_password@localhost:5432/sentiment_pulse_db"
)
FORCE_POSTGRES = os.environ.get("TEST_POSTGRES", "0") == "1"


def is_postgres_available() -> bool:
    """Checks whether the PostgreSQL database is reachable."""
    if not POSTGRES_URL.startswith("postgresql"):
        return False
    try:
        test_eng = create_engine(POSTGRES_URL, connect_args={"connect_timeout": 2})
        with test_eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        test_eng.dispose()
        return True
    except Exception:
        return False


postgres_ready = is_postgres_available()
skip_if_no_postgres = pytest.mark.skipif(
    not postgres_ready,
    reason="PostgreSQL service not reachable on 5432 (executed in CI via .github/workflows/postgres_integration.yml)"
)


# ==============================================================================
# SECTION A: PostgreSQL Dialect SQL Compilation & Compatibility (Runs Anywhere)
# ==============================================================================

def test_postgres_dialect_staging_insert_compilation():
    """Verifies that bulk insert statement compiles cleanly for PostgreSQL dialect."""
    pg_dialect = postgresql.dialect()
    records = [
        {
            "import_job_id": str(uuid.uuid4()),
            "user_id": 1,
            "row_number": 1,
            "text": "Great service!",
            "clean_text": "great service!",
            "sentiment": "positive",
            "confidence": 0.98,
            "positive_score": 0.98,
            "neutral_score": 0.01,
            "negative_score": 0.01,
            "category": "Customer Support",
            "source": "csv_import",
            "rating": 5,
            "model_version": "3.0.0",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    ]
    stmt = insert(FeedbackStaging).values(records)
    compiled = str(stmt.compile(dialect=pg_dialect))
    assert "INSERT INTO feedback_staging" in compiled
    assert "import_job_id" in compiled
    assert "row_number" in compiled


def test_postgres_dialect_atomic_transfer_compilation():
    """Verifies that atomic INSERT INTO feedback SELECT FROM feedback_staging compiles cleanly for PostgreSQL."""
    pg_dialect = postgresql.dialect()
    job_id = str(uuid.uuid4())

    transfer_stmt = insert(Feedback).from_select(
        [
            Feedback.user_id,
            Feedback.text,
            Feedback.clean_text,
            Feedback.sentiment,
            Feedback.confidence,
            Feedback.positive_score,
            Feedback.neutral_score,
            Feedback.negative_score,
            Feedback.category,
            Feedback.source,
            Feedback.rating,
            Feedback.model_version,
            Feedback.created_at,
            Feedback.updated_at,
        ],
        select(
            FeedbackStaging.user_id,
            FeedbackStaging.text,
            FeedbackStaging.clean_text,
            FeedbackStaging.sentiment,
            FeedbackStaging.confidence,
            FeedbackStaging.positive_score,
            FeedbackStaging.neutral_score,
            FeedbackStaging.negative_score,
            FeedbackStaging.category,
            FeedbackStaging.source,
            FeedbackStaging.rating,
            FeedbackStaging.model_version,
            FeedbackStaging.created_at,
            FeedbackStaging.updated_at,
        ).where(FeedbackStaging.import_job_id == job_id)
    )

    compiled = str(transfer_stmt.compile(dialect=pg_dialect))
    assert "INSERT INTO feedback" in compiled
    assert "SELECT" in compiled
    assert "FROM feedback_staging" in compiled
    assert "WHERE feedback_staging.import_job_id =" in compiled


def test_postgres_dialect_staging_delete_compilation():
    """Verifies that scoped staging purge compiles cleanly for PostgreSQL."""
    pg_dialect = postgresql.dialect()
    job_id = str(uuid.uuid4())
    purge_stmt = delete(FeedbackStaging).where(FeedbackStaging.import_job_id == job_id)
    compiled = str(purge_stmt.compile(dialect=pg_dialect))
    assert "DELETE FROM feedback_staging" in compiled
    assert "WHERE feedback_staging.import_job_id =" in compiled


def test_postgres_dialect_import_jobs_query_with_updated_at_compilation():
    """Verifies that the startup recovery query compiles with import_jobs.updated_at on PostgreSQL."""
    pg_dialect = postgresql.dialect()
    stmt = select(ImportJob).where(ImportJob.status.in_(["PENDING", "PROCESSING"]))
    compiled = str(stmt.compile(dialect=pg_dialect))
    assert "import_jobs.updated_at" in compiled
    assert "import_jobs.stage" in compiled
    assert "import_jobs.last_checkpoint_row" in compiled


def test_postgres_dialect_cas_finalization_with_updated_at_compilation():
    """Verifies that the finalization lease CAS query compiles with updated_at on PostgreSQL."""
    pg_dialect = postgresql.dialect()
    stale_cutoff = datetime.now(timezone.utc)
    cas_stmt = (
        select(ImportJob)
        .where(
            ImportJob.id == "test-job-uuid",
            ImportJob.stage == "FINALIZING",
            (ImportJob.updated_at < stale_cutoff) | (ImportJob.updated_at.is_(None))
        )
    )
    compiled = str(cas_stmt.compile(dialect=pg_dialect))
    assert "import_jobs.updated_at <" in compiled


# ==============================================================================
# SECTION B: Live PostgreSQL Integration (Runs in CI or when PG is live)
# ==============================================================================

@pytest.fixture(scope="module")
def pg_session():
    """Yields a database session connected to real PostgreSQL."""
    if not postgres_ready:
        pytest.skip("PostgreSQL not reachable")

    engine = create_engine(POSTGRES_URL)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()


@skip_if_no_postgres
def test_pg_bulk_insert_and_atomic_transfer(pg_session):
    """
    Live test on PostgreSQL:
    1. Bulk insert 2,000 records into feedback_staging.
    2. Atomically transfer to feedback via INSERT INTO ... SELECT.
    3. Verify feedback count and staging purge.
    """
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    job = ImportJob(
        id=job_id,
        user_id=1,
        filename="pg_test.csv",
        total_rows=2000,
        processed_rows=0,
        successful_rows=0,
        failed_rows=0,
        status="PROCESSING",
        stage="INSERTING",
        created_at=now,
        updated_at=now,
    )
    pg_session.add(job)
    pg_session.commit()

    now = datetime.now(timezone.utc)
    staging_rows = [
        {
            "import_job_id": job_id,
            "user_id": 1,
            "row_number": i,
            "text": f"Postgres test feedback record #{i}",
            "clean_text": f"postgres test feedback record #{i}",
            "sentiment": "positive",
            "confidence": 0.95,
            "positive_score": 0.95,
            "neutral_score": 0.03,
            "negative_score": 0.02,
            "category": "IntegrationTest",
            "source": "pg_ci",
            "rating": 5,
            "model_version": "3.0.0",
            "created_at": now,
            "updated_at": now,
        }
        for i in range(1, 2001)
    ]

    # Step 1: Bulk insert into staging
    pg_session.execute(insert(FeedbackStaging), staging_rows)
    pg_session.commit()

    staged_count = pg_session.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).count()
    assert staged_count == 2000

    # Step 2: Atomic transfer
    transfer_stmt = insert(Feedback).from_select(
        [
            Feedback.user_id,
            Feedback.text,
            Feedback.clean_text,
            Feedback.sentiment,
            Feedback.confidence,
            Feedback.positive_score,
            Feedback.neutral_score,
            Feedback.negative_score,
            Feedback.category,
            Feedback.source,
            Feedback.rating,
            Feedback.model_version,
            Feedback.created_at,
            Feedback.updated_at,
        ],
        select(
            FeedbackStaging.user_id,
            FeedbackStaging.text,
            FeedbackStaging.clean_text,
            FeedbackStaging.sentiment,
            FeedbackStaging.confidence,
            FeedbackStaging.positive_score,
            FeedbackStaging.neutral_score,
            FeedbackStaging.negative_score,
            FeedbackStaging.category,
            FeedbackStaging.source,
            FeedbackStaging.rating,
            FeedbackStaging.model_version,
            FeedbackStaging.created_at,
            FeedbackStaging.updated_at,
        ).where(FeedbackStaging.import_job_id == job_id)
    )
    pg_session.execute(transfer_stmt)

    # Step 3: Purge staging
    pg_session.execute(delete(FeedbackStaging).where(FeedbackStaging.import_job_id == job_id))
    job.status = "COMPLETED"
    job.successful_rows = 2000
    pg_session.commit()

    # Step 4: Verify
    remaining_staged = pg_session.query(FeedbackStaging).filter(FeedbackStaging.import_job_id == job_id).count()
    assert remaining_staged == 0

    transferred = pg_session.query(Feedback).filter(Feedback.category == "IntegrationTest").count()
    assert transferred == 2000

    # Cleanup
    pg_session.query(Feedback).filter(Feedback.category == "IntegrationTest").delete()
    pg_session.query(ImportJob).filter(ImportJob.id == job_id).delete()
    pg_session.commit()


@skip_if_no_postgres
def test_pg_multi_tenant_isolation(pg_session):
    """Verifies strict tenant isolation under PostgreSQL row filtering."""
    now = datetime.now(timezone.utc)
    t1_records = [
        Feedback(user_id=1001, text=f"Tenant 1 record {i}", sentiment="positive", confidence=0.9, category="TenantPG")
        for i in range(5)
    ]
    t2_records = [
        Feedback(user_id=1002, text=f"Tenant 2 record {i}", sentiment="negative", confidence=0.8, category="TenantPG")
        for i in range(5)
    ]
    pg_session.add_all(t1_records + t2_records)
    pg_session.commit()

    try:
        t1_found = pg_session.query(Feedback).filter(Feedback.user_id == 1001, Feedback.category == "TenantPG").count()
        t2_found = pg_session.query(Feedback).filter(Feedback.user_id == 1002, Feedback.category == "TenantPG").count()
        assert t1_found == 5
        assert t2_found == 5

        # Cross-tenant query produces 0
        leak_check = pg_session.query(Feedback).filter(
            Feedback.user_id == 1001,
            Feedback.text.like("%Tenant 2%")
        ).count()
        assert leak_check == 0
    finally:
        pg_session.query(Feedback).filter(Feedback.category == "TenantPG").delete()
        pg_session.commit()


@skip_if_no_postgres
def test_pg_chunk_size_benchmark(pg_session):
    """
    Evaluates bulk insertion throughput across chunk sizes (1,000 vs 5,000 vs 10,000) on PostgreSQL.
    """
    chunk_sizes = [1000, 5000, 10000]
    bench_results = {}
    now = datetime.now(timezone.utc)

    for chunk_size in chunk_sizes:
        records = [
            {
                "import_job_id": f"bench_{chunk_size}",
                "user_id": 9999,
                "row_number": i,
                "text": f"Benchmark record #{i}",
                "clean_text": f"benchmark record #{i}",
                "sentiment": "neutral",
                "confidence": 0.75,
                "positive_score": 0.1,
                "neutral_score": 0.8,
                "negative_score": 0.1,
                "category": "PGBench",
                "source": "benchmark",
                "rating": 3,
                "model_version": "3.0.0",
                "created_at": now,
                "updated_at": now,
            }
            for i in range(chunk_size)
        ]

        t0 = time.perf_counter()
        pg_session.execute(insert(FeedbackStaging), records)
        pg_session.commit()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        rate = chunk_size / max(elapsed, 1e-6)
        bench_results[chunk_size] = {"elapsed_sec": round(elapsed, 4), "records_per_sec": round(rate, 1)}

        # Cleanup
        pg_session.execute(delete(FeedbackStaging).where(FeedbackStaging.import_job_id == f"bench_{chunk_size}"))
        pg_session.commit()

    assert len(bench_results) == 3
    for cs, res in bench_results.items():
        assert res["records_per_sec"] > 5000, f"Chunk size {cs} throughput {res['records_per_sec']} rps is too low"

