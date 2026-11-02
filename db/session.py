"""
Database session management, connection pooling, and lifecycle dependencies for Sentiment Pulse.
"""

from typing import Generator
from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.orm import sessionmaker, Session

from core.config import settings
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


def _migrate_schema() -> None:
    """
    Inspects database tables and ensures schema synchronization.
    Safely reconciles nullable user_id and drops dangling foreign keys on both SQLite and PostgreSQL.
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        latest_revision = "e6a1b2c3d4e5"

        # Check and add 'stage', 'file_path', 'file_hash', 'last_checkpoint_row' to import_jobs if missing
        if "import_jobs" in tables:
            columns = [c["name"] for c in inspector.get_columns("import_jobs")]
            if "stage" not in columns:
                logger.info("Migrating schema: adding 'stage' column to import_jobs table...")
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE import_jobs ADD COLUMN stage VARCHAR(50) DEFAULT 'QUEUED' NOT NULL"))
            if "file_path" not in columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE import_jobs ADD COLUMN file_path VARCHAR(512)"))
            if "file_hash" not in columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE import_jobs ADD COLUMN file_hash VARCHAR(64)"))
            if "last_checkpoint_row" not in columns:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE import_jobs ADD COLUMN last_checkpoint_row INTEGER DEFAULT 0 NOT NULL"))

        # Check and create feedback_staging table if missing
        if "feedback_staging" not in tables:
            logger.info("Migrating schema: creating feedback_staging table...")
            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS feedback_staging (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        import_job_id VARCHAR(36) NOT NULL,
                        user_id INTEGER,
                        row_number INTEGER,
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
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_staging_import_job_id ON feedback_staging (import_job_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_staging_user_id ON feedback_staging (user_id)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_feedback_staging_row_number ON feedback_staging (row_number)"))
        else:
            staging_cols = [c["name"] for c in inspector.get_columns("feedback_staging")]
            if "row_number" not in staging_cols:
                with engine.begin() as conn:
                    conn.execute(text("ALTER TABLE feedback_staging ADD COLUMN row_number INTEGER"))

        if is_sqlite:
            # Check for dangling foreign keys referencing dropped 'users' table
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
                                    CREATE TABLE feedback_clean (
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
                                    CREATE TABLE import_jobs_clean (
                                        id VARCHAR(36) NOT NULL PRIMARY KEY,
                                        user_id INTEGER,
                                        filename VARCHAR(255) NOT NULL,
                                        total_rows INTEGER NOT NULL DEFAULT 0,
                                        processed_rows INTEGER NOT NULL DEFAULT 0,
                                        successful_rows INTEGER NOT NULL DEFAULT 0,
                                        failed_rows INTEGER NOT NULL DEFAULT 0,
                                        status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
                                        error_summary TEXT,
                                        created_at DATETIME NOT NULL,
                                        completed_at DATETIME
                                    )
                                """))
                                conn.execute(text("""
                                    INSERT INTO import_jobs_clean (
                                        id, user_id, filename, total_rows, processed_rows,
                                        successful_rows, failed_rows, status, error_summary, created_at, completed_at
                                    )
                                    SELECT
                                        id, user_id, filename, total_rows, processed_rows,
                                        successful_rows, failed_rows, status, error_summary, created_at, completed_at
                                    FROM import_jobs
                                """))
                                conn.execute(text("DROP TABLE import_jobs"))
                                conn.execute(text("ALTER TABLE import_jobs_clean RENAME TO import_jobs"))
                                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_import_jobs_id ON import_jobs (id)"))
                                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_import_jobs_user_id ON import_jobs (user_id)"))
                                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_import_jobs_status ON import_jobs (status)"))
                            conn.execute(text("PRAGMA foreign_keys = ON"))
        else:
            # PostgreSQL schema synchronization
            with engine.begin() as conn:
                for tbl in ["feedback", "import_jobs"]:
                    if tbl in tables:
                        fks = inspector.get_foreign_keys(tbl)
                        for fk in fks:
                            if fk.get("referred_table") == "users" and fk.get("name"):
                                conn.execute(text(f"ALTER TABLE {tbl} DROP CONSTRAINT IF EXISTS {fk['name']}"))
                        try:
                            conn.execute(text(f"ALTER TABLE {tbl} ALTER COLUMN user_id DROP NOT NULL"))
                        except Exception as col_err:
                            logger.debug(f"Column nullable check on {tbl}.user_id: {col_err}")

        # Synchronize Alembic migration state
        if "alembic_version" in tables:
            with engine.begin() as conn:
                res = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
                if not res:
                    conn.execute(text(f"INSERT INTO alembic_version (version_num) VALUES ('{latest_revision}')"))
                elif res[0] in ("992b8e22d756", "b5f1a9238de4", "c4e2f187a912", "d5f3a1829e01"):
                    conn.execute(text(f"UPDATE alembic_version SET version_num = '{latest_revision}'"))
        else:
            try:
                with engine.begin() as conn:
                    conn.execute(text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"))
                    conn.execute(text(f"INSERT INTO alembic_version (version_num) VALUES ('{latest_revision}')"))
            except Exception as alembic_err:
                logger.debug(f"Alembic version table initialization skipped: {alembic_err}")

        logger.info("Database schema verification completed.")
    except Exception as exc:
        logger.error(f"Error during schema verification: {exc}", exc_info=True)


def init_db() -> None:
    """
    Initializes database tables if they do not exist, and reconciles
    schema differences on existing tables.
    """
    logger.info("Verifying and initializing database schema tables...")
    Base.metadata.create_all(bind=engine)
    _migrate_schema()
    logger.info("Database tables initialized successfully.")
