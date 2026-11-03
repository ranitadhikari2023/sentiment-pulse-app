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


def get_available_cpu_count() -> int:
    """
    Detects the true available CPU allocation, respecting container limits:
    1. cgroups v2 /sys/fs/cgroup/cpu.max (quota and period)
    2. cgroups v1 /sys/fs/cgroup/cpu/cpu.cfs_quota_us and cpu.cfs_period_us
    3. os.sched_getaffinity(0) if available on Linux
    4. os.cpu_count() fallback
    Clamps result to at least 1 core.
    """
    # 1. Check cgroups v2
    cgroup2_path = "/sys/fs/cgroup/cpu.max"
    if os.path.exists(cgroup2_path):
        try:
            with open(cgroup2_path, "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
                if len(parts) >= 2 and parts[0] != "max":
                    quota = float(parts[0])
                    period = float(parts[1])
                    if period > 0:
                        return max(1, int(round(quota / period)))
        except Exception:
            pass

    # 2. Check cgroups v1
    quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
    period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
    if os.path.exists(quota_path) and os.path.exists(period_path):
        try:
            with open(quota_path, "r", encoding="utf-8") as f_q, open(period_path, "r", encoding="utf-8") as f_p:
                quota = float(f_q.read().strip())
                period = float(f_p.read().strip())
                if quota > 0 and period > 0:
                    return max(1, int(round(quota / period)))
        except Exception:
            pass

    # 3. Check Linux process affinity
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
            if affinity > 0:
                return affinity
        except Exception:
            pass

    # 4. Standard os.cpu_count fallback
    count = os.cpu_count()
    return max(1, count or 1)


def get_upload_dir() -> str:
    """
    Resolves upload directory with persistent storage priority:
    1. Explicit UPLOAD_DIR environment variable
    2. Render Persistent Disk mount at /var/data/uploads if /var/data exists
    3. Default local project data/uploads directory
    """
    env_dir = os.getenv("UPLOAD_DIR")
    if env_dir:
        return env_dir
    if os.path.exists("/var/data") and os.path.isdir("/var/data"):
        return "/var/data/uploads"
    return os.path.join(BASE_DIR, "data", "uploads")


KNOWN_INSECURE_SECRETS = {
    "sentiment-pulse-production-secret-v1",
    "secret",
    "default",
    "changeme",
    "your-secret-key",
    "password",
    "admin",
    "12345678",
    "1234567890",
    "temporarysecretkey",
}


class Settings:
    # General App Config
    APP_NAME: str = os.getenv("APP_NAME", "Sentiment Pulse")
    APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "sentiment-pulse-production-secret-v1")

    @classmethod
    def validate_secret_entropy(cls, key: str) -> None:
        """Validates that a secret key meets entropy and length requirements."""
        if not key or not isinstance(key, str):
            raise ValueError("SECRET_KEY is empty or not a string.")
        if len(key) < 32:
            raise ValueError(f"SECRET_KEY must be at least 32 characters long (got {len(key)}).")
        lower_key = key.lower()
        if key in KNOWN_INSECURE_SECRETS or any(ph in lower_key for ph in ["placeholder", "default", "changeme"]):
            raise ValueError("SECRET_KEY cannot be a known default, insecure placeholder, or test string.")
        # Check character diversity / entropy
        if len(set(key)) < 6:
            raise ValueError("SECRET_KEY has insufficient character entropy (too few distinct characters).")

    def validate_production_security(self) -> None:
        """
        Enforce fail-closed validation:
        If ENVIRONMENT == 'production', the application MUST refuse to start if SECRET_KEY
        is missing, weak, or set to a known default or placeholder.
        """
        if self.ENVIRONMENT == "production":
            key = os.getenv("SECRET_KEY") or self.SECRET_KEY
            if not key:
                raise RuntimeError("CRITICAL SECURITY VIOLATION: SECRET_KEY environment variable is missing in production.")
            try:
                self.validate_secret_entropy(key)
            except ValueError as ve:
                raise RuntimeError(f"CRITICAL SECURITY VIOLATION: Insecure SECRET_KEY in production: {str(ve)}")

    def get_secret_key_bytes(self, override: Optional[str] = None) -> bytes:
        """
        Returns validated secret key bytes.
        Enforces production validation without leaking secret values to logs.
        """
        if override:
            return override.encode("utf-8")
        if self.ENVIRONMENT == "production":
            self.validate_production_security()
        key = self.SECRET_KEY
        if not key:
            if self.ENVIRONMENT == "production":
                raise RuntimeError("CRITICAL SECURITY VIOLATION: SECRET_KEY is missing in production.")
            return b"insecure-dev-key-change-in-prod-minimum32chars!"
        return key.encode("utf-8")

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
    RATE_LIMIT_STATUS_POLL: int = int(os.getenv("RATE_LIMIT_STATUS_POLL", "120"))
    RATE_LIMIT_IMPORT_UPLOAD: int = int(os.getenv("RATE_LIMIT_IMPORT_UPLOAD", "30"))
    RATE_LIMIT_DEFAULT: int = int(os.getenv("RATE_LIMIT_DEFAULT", "120"))

    # Application Base URL
    APP_BASE_URL: str = os.getenv(
        "APP_BASE_URL",
        "https://sentiment-pulse-app.onrender.com" if os.getenv("ENVIRONMENT") == "production" else "http://localhost:8000"
    )

    # Batch & CSV Import Limits
    API_KEYS: dict = {}
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "50"))
    MAX_IMPORT_FILE_SIZE_BYTES: int = int(
        os.getenv("MAX_IMPORT_FILE_SIZE_BYTES", str(100 * 1024 * 1024))
    )  # 100 MB (104,857,600 bytes)
    MAX_UPLOAD_FILE_SIZE: int = MAX_IMPORT_FILE_SIZE_BYTES
    MAX_IMPORT_ROWS: int = int(
        os.getenv("MAX_IMPORT_ROWS", "1050000")
    )
    DB_INSERT_CHUNK_SIZE: int = int(
        os.getenv("DB_INSERT_CHUNK_SIZE", os.getenv("IMPORT_CHUNK_SIZE", "5000"))
    )
    IMPORT_CHUNK_SIZE: int = DB_INSERT_CHUNK_SIZE
    MAX_CONCURRENT_IMPORT_JOBS: int = int(os.getenv("MAX_CONCURRENT_IMPORT_JOBS", "10"))
    MAX_CONCURRENT_PER_TENANT: int = int(os.getenv("MAX_CONCURRENT_PER_TENANT", "5"))

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

    # Phase 2 Inference & Tokenizer Optimizations
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))
    INFERENCE_BATCH_SIZE: int = int(os.getenv("INFERENCE_BATCH_SIZE", "16"))
    INFERENCE_NUM_THREADS: int = int(
        os.getenv("INFERENCE_NUM_THREADS", str(min(4, get_available_cpu_count())))
    )
    INFERENCE_WORKER_THREADS: int = int(os.getenv("INFERENCE_WORKER_THREADS", "2"))
    MAX_CONCURRENT_INFERENCE_JOBS: int = int(os.getenv("MAX_CONCURRENT_INFERENCE_JOBS", "2"))
    PREDICTION_CACHE_ENABLED: bool = os.getenv("PREDICTION_CACHE_ENABLED", "true").lower() in ("true", "1")
    PREDICTION_CACHE_CAPACITY: int = int(os.getenv("PREDICTION_CACHE_CAPACITY", "100000"))
    PREDICTION_CACHE_MAX_BYTES: int = int(os.getenv("PREDICTION_CACHE_MAX_BYTES", str(50 * 1024 * 1024)))
    SINGLE_FLIGHT_ENABLED: bool = os.getenv("SINGLE_FLIGHT_ENABLED", "true").lower() in ("true", "1")
    LENGTH_BUCKETING_ENABLED: bool = os.getenv("LENGTH_BUCKETING_ENABLED", "true").lower() in ("true", "1")

    # Durable Upload Storage
    UPLOAD_DIR: str = get_upload_dir()


settings = Settings()
