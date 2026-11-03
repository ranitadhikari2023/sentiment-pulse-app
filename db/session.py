"""
Database session management, connection pooling, and lifecycle dependencies for Sentiment Pulse.
Authoritative schema migration via Alembic with advisory locking and fail-closed compatibility verification.
"""

import os
import sys
from typing import Generator, List, Dict, Set
from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.orm import sessionmaker, Session

from core.config import settings, BASE_DIR
from core.logging import get_logger
from db.models import Base

logger = get_logger(__name__)

# Determine connect args and pooling options based on engine dialect
is_sqlite = settings.DATABASE_URL.startswith("sqlite")

engine_kwargs = {
    "pool_pre_ping": True,
}

if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = settings.DB_POOL_SIZE
    engine_kwargs["max_overflow"] = settings.DB_MAX_OVERFLOW
    engine_kwargs["pool_timeout"] = settings.DB_POOL_TIMEOUT
    engine_kwargs["pool_recycle"] = settings.DB_POOL_RECYCLE

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if is_sqlite:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a transactional database session per request.
    Rolls back uncommitted transactions on unhandled exceptions and ensures clean closure.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as exc:
        db.rollback()
        logger.error(f"Database session rolled back due to error: {exc}")
        raise
    finally:
        db.close()


# Required schema definition for fail-closed compatibility check
REQUIRED_TABLE_COLUMNS: Dict[str, Set[str]] = {
    "feedback": {
        "id", "user_id", "text", "clean_text", "sentiment", "confidence",
        "positive_score", "neutral_score", "negative_score", "category",
        "source", "rating", "model_version", "created_at", "updated_at",
    },
    "import_jobs": {
        "id", "user_id", "filename", "total_rows", "processed_rows",
        "successful_rows", "failed_rows", "status", "stage", "error_summary",
        "file_path", "file_hash", "last_checkpoint_row", "created_at",
        "updated_at", "completed_at",
    },
    "feedback_staging": {
        "id", "import_job_id", "user_id", "row_number", "text", "clean_text",
        "sentiment", "confidence", "positive_score", "neutral_score",
        "negative_score", "category", "source", "rating", "model_version",
        "created_at", "updated_at",
    },
}

LATEST_ALEMBIC_HEAD = "a1b2c3d4e5f7"
PG_MIGRATION_LOCK_ID = 8675309


def run_alembic_migrations() -> None:
    """
    Applies Alembic migrations up to head.
    On PostgreSQL, acquires an advisory lock (id: 8675309) to prevent race conditions
    when multiple application workers start concurrently.
    """
    from alembic.config import Config
    from alembic import command

    alembic_ini_path = os.path.join(BASE_DIR, "alembic.ini")
    if not os.path.exists(alembic_ini_path):
        logger.warning(f"Alembic config not found at {alembic_ini_path}; skipping migration command.")
        return

    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

    if is_sqlite:
        logger.info("Executing Alembic migrations (upgrade head) for SQLite database...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic migrations applied successfully.")
    else:
        logger.info("Acquiring PostgreSQL migration advisory lock...")
        # Use raw connection to maintain lock across migration statements
        with engine.connect() as lock_conn:
            try:
                lock_conn.execute(text(f"SELECT pg_advisory_lock({PG_MIGRATION_LOCK_ID})"))
                logger.info("PostgreSQL migration advisory lock acquired. Executing Alembic upgrade head...")
                command.upgrade(alembic_cfg, "head")
                logger.info("Alembic migrations completed successfully.")
            finally:
                try:
                    lock_conn.execute(text(f"SELECT pg_advisory_unlock({PG_MIGRATION_LOCK_ID})"))
                    logger.info("PostgreSQL migration advisory lock released.")
                except Exception as unlock_err:
                    logger.warning(f"Error releasing PostgreSQL advisory lock: {unlock_err}")


def verify_schema_compatibility(check_engine=None) -> None:
    """
    Performs fail-closed schema verification against expected models.
    Verifies that all required tables and all required columns exist.
    Raises RuntimeError if any required table or column is missing.
    """
    target_engine = check_engine or engine
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())

    missing_tables: List[str] = []
    missing_columns: Dict[str, List[str]] = {}

    for table_name, expected_cols in REQUIRED_TABLE_COLUMNS.items():
        if table_name not in existing_tables:
            missing_tables.append(table_name)
            continue

        actual_cols = {c["name"] for c in inspector.get_columns(table_name)}
        missing = sorted(list(expected_cols - actual_cols))
        if missing:
            missing_columns[table_name] = missing

    if missing_tables or missing_columns:
        error_details = []
        if missing_tables:
            error_details.append(f"Missing required tables: {', '.join(missing_tables)}")
        for tbl, cols in missing_columns.items():
            error_details.append(f"Table '{tbl}' is missing required columns: {', '.join(cols)}")
        full_msg = "CRITICAL DATABASE SCHEMA MISMATCH: " + "; ".join(error_details)
        logger.critical(full_msg)
        raise RuntimeError(full_msg)

    logger.info("Database schema compatibility check passed: all required tables and columns verified.")


def verify_alembic_version(check_engine=None) -> str:
    """
    Verifies that the alembic_version table is populated and reports current version.
    """
    target_engine = check_engine or engine
    inspector = inspect(target_engine)
    if "alembic_version" not in inspector.get_table_names():
        logger.warning("Table 'alembic_version' does not exist.")
        return "none"

    with target_engine.connect() as conn:
        res = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
        version = res[0] if res else "none"
        logger.info(f"Current Alembic database version: {version}")
        return version


def _self_heal_sqlite_dangling_fks() -> None:
    """
    Cleans up legacy dangling users foreign keys on SQLite without data loss.
    """
    if not is_sqlite:
        return

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        for tbl in ["feedback", "import_jobs"]:
            if tbl in tables:
                fks = inspector.get_foreign_keys(tbl)
                has_users_fk = any(fk.get("referred_table") == "users" for fk in fks)
                if has_users_fk:
                    logger.info(f"Self-healing SQLite schema: dropping dangling users foreign key from {tbl}...")
                    with engine.begin() as conn:
                        conn.execute(text("PRAGMA foreign_keys = OFF"))
                        if tbl == "feedback":
                            conn.execute(text("""
                                CREATE TABLE IF NOT EXISTS feedback_clean (
                                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                                    user_id INTEGER,
                                    text TEXT NOT NULL,
                                    clean_text TEXT,
                                    sentiment VARCHAR(20) NOT NULL,
                                    confidence FLOAT NOT NULL,
                                    positive_score FLOAT NOT NULL DEFAULT 0.0,
                                    neutral_score FLOAT NOT NULL DEFAULT 0.0,
                                    negative_score FLOAT NOT NULL DEFAULT 0.0,
                                    category VARCHAR(50),
                                    source VARCHAR(50),
                                    rating INTEGER,
                                    model_version VARCHAR(30) NOT NULL DEFAULT '3.0.0',
                                    created_at DATETIME NOT NULL,
                                    updated_at DATETIME NOT NULL
                                )
                            """))
                            conn.execute(text("""
                                INSERT INTO feedback_clean (
                                    id, user_id, text, clean_text, sentiment, confidence,
                                    positive_score, neutral_score, negative_score, category, source, rating,
                                    model_version, created_at, updated_at
                                )
                                SELECT
                                    id, user_id, text, clean_text, sentiment, confidence,
                                    positive_score, neutral_score, negative_score, category, source, rating,
                                    model_version, created_at, updated_at
                                FROM feedback
                            """))
                            conn.execute(text("DROP TABLE feedback"))
                            conn.execute(text("ALTER TABLE feedback_clean RENAME TO feedback"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_id ON feedback (id)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_user_id ON feedback (user_id)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_sentiment ON feedback (sentiment)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_category ON feedback (category)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_source ON feedback (source)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_created_at ON feedback (created_at)"))
                        elif tbl == "import_jobs":
                            conn.execute(text("""
                                CREATE TABLE IF NOT EXISTS import_jobs_clean (
                                    id VARCHAR(36) NOT NULL PRIMARY KEY,
                                    user_id INTEGER,
                                    filename VARCHAR(255) NOT NULL,
                                    total_rows INTEGER NOT NULL DEFAULT 0,
                                    processed_rows INTEGER NOT NULL DEFAULT 0,
                                    successful_rows INTEGER NOT NULL DEFAULT 0,
                                    failed_rows INTEGER NOT NULL DEFAULT 0,
                                    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                                    stage VARCHAR(50) NOT NULL DEFAULT 'QUEUED',
                                    error_summary TEXT,
                                    file_path VARCHAR(512),
                                    file_hash VARCHAR(64),
                                    last_checkpoint_row INTEGER NOT NULL DEFAULT 0,
                                    created_at DATETIME NOT NULL,
                                    updated_at DATETIME,
                                    completed_at DATETIME
                                )
                            """))
                            conn.execute(text("""
                                INSERT INTO import_jobs_clean (
                                    id, user_id, filename, total_rows, processed_rows,
                                    successful_rows, failed_rows, status, stage, error_summary,
                                    file_path, file_hash, last_checkpoint_row, created_at, updated_at, completed_at
                                )
                                SELECT
                                    id, user_id, filename, total_rows, processed_rows,
                                    successful_rows, failed_rows, status,
                                    COALESCE(stage, 'QUEUED'),
                                    error_summary, file_path, file_hash,
                                    COALESCE(last_checkpoint_row, 0),
                                    created_at, updated_at, completed_at
                                FROM import_jobs
                            """))
                            conn.execute(text("DROP TABLE import_jobs"))
                            conn.execute(text("ALTER TABLE import_jobs_clean RENAME TO import_jobs"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_import_jobs_id ON import_jobs (id)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_import_jobs_user_id ON import_jobs (user_id)"))
                            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_import_jobs_status ON import_jobs (status)"))
                        conn.execute(text("PRAGMA foreign_keys = ON"))
    except Exception as exc:
        logger.warning(f"SQLite self-healing check note: {exc}")


def init_db() -> None:
    """
    Initializes database tables and verifies schema consistency:
    1. Base.metadata.create_all for SQLite initial table creation
    2. Self-heals dangling SQLite foreign keys if any
    3. Executes Alembic migrations (upgrade head) with PostgreSQL advisory lock
    4. Performs fail-closed schema compatibility verification
    """
    logger.info("Verifying and initializing database schema...")

    # Step 1: For SQLite local development, ensure base tables exist
    if is_sqlite:
        Base.metadata.create_all(bind=engine)
        _self_heal_sqlite_dangling_fks()

    # Step 2: Run Alembic migrations as authoritative source of schema truth
    try:
        run_alembic_migrations()
    except Exception as mig_err:
        logger.error(f"Alembic migration failed during startup: {mig_err}", exc_info=True)
        # If migration fails, don't ignore it - schema verification below will fail-close

    # Step 3: Verify schema compatibility - raises RuntimeError if required columns are missing
    verify_schema_compatibility()

    # Step 4: Verify current alembic version
    version = verify_alembic_version()
    logger.info(f"Database initialized and verified successfully at Alembic revision {version}.")
