"""
Core configuration module for Sentiment Pulse application.
Handles environment variables, database settings, and rate limiting.
Authentication and email verification settings have been permanently removed.
"""

import os
from typing import List, Optional
import logging
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

logger = logging.getLogger(__name__)


def _normalize_db_url(url: str) -> str:
    """Normalize Render / Heroku postgres URLs to SQLAlchemy psycopg3 dialect."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _parse_origins(origins_str: str) -> List[str]:
    """Parse comma-separated origins."""
    if not origins_str or origins_str.strip() == "*":
        return ["*"]
    return [o.strip() for o in origins_str.split(",") if o.strip()]


class Settings:
    # General App Config
    APP_NAME: str = os.getenv("APP_NAME", "Sentiment Pulse")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")

    # Database Configuration
    DATABASE_URL: str = _normalize_db_url(
        os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'sentiment.db')}")
    )
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "1800"))

    # CORS Settings
    ALLOWED_ORIGINS: List[str] = _parse_origins(os.getenv("ALLOWED_ORIGINS", "*"))

    # Rate Limiting (in-memory per client IP)
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("true", "1")

    # Application Base URL
    APP_BASE_URL: str = os.getenv(
        "APP_BASE_URL",
        "https://sentiment-pulse-app.onrender.com" if os.getenv("ENVIRONMENT") == "production" else "http://localhost:8000"
    )

    # Batch & CSV Import Limits
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "50"))
    MAX_IMPORT_FILE_SIZE_BYTES: int = int(
        os.getenv("MAX_IMPORT_FILE_SIZE_BYTES", str(100 * 1024 * 1024))
    )  # 100 MB (104,857,600 bytes)
    MAX_UPLOAD_FILE_SIZE: int = MAX_IMPORT_FILE_SIZE_BYTES
    MAX_IMPORT_ROWS: int = int(
        os.getenv("MAX_IMPORT_ROWS", "1050000")
    )
    DB_INSERT_CHUNK_SIZE: int = int(
        os.getenv("DB_INSERT_CHUNK_SIZE", os.getenv("IMPORT_CHUNK_SIZE", "2000"))
    )
    IMPORT_CHUNK_SIZE: int = DB_INSERT_CHUNK_SIZE

    _max_import_rows_formatted: Optional[str] = None

    @property
    def MAX_IMPORT_ROWS_FORMATTED(self) -> str:
        """Returns international human-readable representation, e.g. '1,050,000'."""
        if self._max_import_rows_formatted is not None:
            return self._max_import_rows_formatted
        return f"{self.MAX_IMPORT_ROWS:,}"

    @MAX_IMPORT_ROWS_FORMATTED.setter
    def MAX_IMPORT_ROWS_FORMATTED(self, value: str) -> None:
        self._max_import_rows_formatted = value

    # Model & Tokenizer Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "sentiment_model.onnx"))
    TOKENIZER_PATH: str = os.getenv("TOKENIZER_PATH", os.path.join(BASE_DIR, "tokenizer", "tokenizer.json"))
    METRICS_PATH: str = os.getenv("METRICS_PATH", os.path.join(BASE_DIR, "model_metrics.json"))


settings = Settings()
