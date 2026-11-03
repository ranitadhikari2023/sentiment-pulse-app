"""
HTTP Middleware for Sentiment Pulse.
Enterprise-grade observability, correlation tracking, edge security integration,
and sliding-window adaptive rate limiting.
"""

import time
import math
import uuid
import re
import threading
from collections import deque
from typing import Dict, Tuple, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from core.config import settings
from core.logging import request_id_ctx, get_logger

logger = get_logger(__name__)

# Regular expressions for path classification
_STATUS_POLL_RE = re.compile(r"^/(api/)?feedback/import/[a-zA-Z0-9_-]+$")
_IMPORT_UPLOAD_RE = re.compile(r"^/(api/)?feedback/import$")
_EXEMPT_PATHS = {
    "/health",
    "/health/ready",
    "/health/live",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
    "/manifest.webmanifest",
}


class SlidingWindowRateLimiter:
    """
    Thread-safe, in-memory sliding window rate limiter.
    Provides endpoint-specific buckets to ensure high-frequency polling
    is never penalized while actively defending against brute-force abuse.
    """

    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self._lock = threading.Lock()
        self._history: Dict[str, deque] = {}
        self._last_cleanup = time.time()

    def _resolve_bucket(self, path: str, method: str) -> Tuple[str, int]:
        """Resolves rate limit bucket name and maximum allowed requests per window."""
        if path.startswith("/static") or path in _EXEMPT_PATHS:
            return "exempt", 0

        if method == "GET" and _STATUS_POLL_RE.match(path):
            return "status_poll", getattr(settings, "RATE_LIMIT_STATUS_POLL", 120)

        if method == "POST" and _IMPORT_UPLOAD_RE.match(path):
            return "import_upload", getattr(settings, "RATE_LIMIT_IMPORT_UPLOAD", 30)

        return "default", getattr(settings, "RATE_LIMIT_DEFAULT", 120)

    def is_rate_limited(self, client_ip: str, path: str, method: str) -> Tuple[bool, int, int, int]:
        """
        Determines whether the request should be rate-limited under sliding window.
        Returns:
            (is_limited, limit, remaining, retry_after_seconds)
        """
        bucket_name, limit = self._resolve_bucket(path, method)
        if bucket_name == "exempt" or limit <= 0:
            return False, 0, 9999, 0

        key = f"{client_ip}:{bucket_name}"
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Periodic cleanup of idle keys every 300 seconds
            if now - self._last_cleanup > 300:
                self._cleanup_stale_keys(now)
                self._last_cleanup = now

            timestamps = self._history.setdefault(key, deque())

            # Prune timestamps older than sliding window
            while timestamps and timestamps[0] <= window_start:
                timestamps.popleft()

            current_count = len(timestamps)
            if current_count >= limit:
                oldest_timestamp = timestamps[0]
                retry_after = max(1, math.ceil(oldest_timestamp + self.window_seconds - now))
                return True, limit, 0, retry_after

            # Record current request timestamp
            timestamps.append(now)
            remaining = max(0, limit - len(timestamps))
            return False, limit, remaining, 0

    def _cleanup_stale_keys(self, now: float) -> None:
        """Evicts keys with no requests within the sliding window to bound memory."""
        cutoff = now - self.window_seconds
        stale_keys = [
            k for k, dq in self._history.items()
            if not dq or dq[-1] < cutoff
        ]
        for k in stale_keys:
            del self._history[k]

    def reset(self) -> None:
        """Resets all tracking buckets (primarily used for unit testing)."""
        with self._lock:
            self._history.clear()


# Global sliding-window rate limiter instance
limiter = SlidingWindowRateLimiter(window_seconds=60)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Core middleware for Sentiment Pulse:
    1. Correlation tracking: Generates or propagates X-Request-ID.
    2. Edge observability: Propagates CF-Ray header if present from Cloudflare.
    3. Latency instrumentation: Attaches X-Process-Time.
    4. Early body size validation: Rejects oversized uploads at HTTP transport layer.
    5. Adaptive rate limiting: Protects endpoints while allowing resilient status polling.
    """

    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        cf_ray = request.headers.get("CF-Ray") or request.headers.get("cf-ray")
        token = request_id_ctx.set(req_id)
        start_time = time.perf_counter()

        # Extract client IP respecting trusted proxy headers
        client_ip = (
            request.headers.get("CF-Connecting-IP")
            or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "127.0.0.1")
        )

        try:
            # 1. Early check for oversized import requests via Content-Length before parsing body
            if request.method == "POST" and request.url.path in ("/feedback/import", "/api/feedback/import"):
                content_length = request.headers.get("content-length")
                if content_length and content_length.isdigit():
                    # Allow 1 MB overhead for multipart boundaries and form fields
                    if int(content_length) > settings.MAX_IMPORT_FILE_SIZE_BYTES + (1024 * 1024):
                        max_mb = settings.MAX_IMPORT_FILE_SIZE_BYTES // (1024 * 1024)
                        process_time = (time.perf_counter() - start_time) * 1000  # ms
                        headers = {
                            "X-Request-ID": req_id,
                            "X-Process-Time": f"{process_time:.2f}ms",
                        }
                        if cf_ray:
                            headers["CF-Ray"] = cf_ray
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": (
                                    f"Uploaded file exceeds maximum allowed size ({max_mb} MB). "
                                    f"File exceeds the maximum permitted size of 100 MB."
                                )
                            },
                            headers=headers,
                        )

            # 2. Sliding window rate limiting
            if getattr(settings, "RATE_LIMIT_ENABLED", True):
                is_limited, limit, remaining, retry_after = limiter.is_rate_limited(
                    client_ip, request.url.path, request.method
                )
                if is_limited:
                    process_time = (time.perf_counter() - start_time) * 1000
                    logger.warning(
                        f"Rate limit exceeded: IP={client_ip} Path={request.url.path} "
                        f"Method={request.method} Limit={limit} RetryAfter={retry_after}s ReqID={req_id}"
                    )
                    headers = {
                        "X-Request-ID": req_id,
                        "X-Process-Time": f"{process_time:.2f}ms",
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(retry_after),
                    }
                    if cf_ray:
                        headers["CF-Ray"] = cf_ray
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": f"Rate limit exceeded. Please retry in {retry_after} seconds.",
                            "retry_after": retry_after,
                            "request_id": req_id,
                        },
                        headers=headers,
                    )

            # 3. Process downstream request
            response: Response = await call_next(request)
            process_time = (time.perf_counter() - start_time) * 1000  # ms
            response.headers["X-Request-ID"] = req_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            if cf_ray:
                response.headers["CF-Ray"] = cf_ray

            return response

        except Exception as exc:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Unhandled exception processing {request.method} {request.url.path} "
                f"[{req_id}]: {exc}",
                exc_info=True,
            )
            raise
        finally:
            request_id_ctx.reset(token)
