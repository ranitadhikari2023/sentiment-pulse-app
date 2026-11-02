"""
Security and tenant identity management module for Sentiment Pulse.
Provides:
1. Cryptographic HMAC-SHA256 tenant token issuance and verification.
2. Trusted server-side tenant identity resolution (API Key, signed token, or trusted gateway).
3. Protection against client-side header spoofing and privilege escalation.
4. Fail-closed security validation for production environments.
"""

import os
import hmac
import hashlib
from typing import Optional, Tuple
from fastapi import Request, HTTPException, status

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


def generate_tenant_token(tenant_id: int, secret_key: Optional[str] = None) -> str:
    """
    Generates a cryptographically signed tenant token.
    Format: {tenant_id}:{hmac_sha256_hex_signature}
    """
    secret = settings.get_secret_key_bytes(secret_key)
    msg = f"tenant:{tenant_id}".encode("utf-8")
    sig = hmac.new(secret, msg, hashlib.sha256).hexdigest()
    return f"{tenant_id}:{sig}"


def verify_tenant_token(token: str, secret_key: Optional[str] = None) -> Optional[int]:
    """
    Validates a cryptographic tenant token using constant-time comparison.
    Returns tenant_id if valid, None if forged or expired.
    """
    if not token or ":" not in token:
        return None
    try:
        parts = token.split(":", 1)
        tenant_id = int(parts[0])
        provided_sig = parts[1]

        secret = settings.get_secret_key_bytes(secret_key)
        expected_sig = hmac.new(secret, f"tenant:{tenant_id}".encode("utf-8"), hashlib.sha256).hexdigest()

        if hmac.compare_digest(provided_sig, expected_sig):
            return tenant_id
        return None
    except Exception:
        return None


def resolve_trusted_tenant_identity(request: Request) -> Optional[int]:
    """
    Resolves authenticated tenant identity from trusted server-side sources:
    1. Bearer Token in Authorization header (cryptographic HMAC tenant token).
    2. API Key in X-API-Key header (mapped against settings.API_KEYS).
    3. Cryptographic signature in X-Tenant-Signature header verifying X-Tenant-ID.
    4. In dev/testing mode: X-Tenant-ID is permitted only when REQUIRE_TENANT_AUTH is False.
    5. Header spoofing prevention: if both credentials and X-Tenant-ID are supplied, they must match.
    """
    authenticated_tenant_id: Optional[int] = None
    auth_header = request.headers.get("Authorization")
    api_key_header = request.headers.get("X-API-Key")
    sig_header = request.headers.get("X-Tenant-Signature")
    raw_header_tenant = request.headers.get("X-Tenant-ID") or request.headers.get("X-User-ID")

    # 1. Check Bearer Token
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        verified = verify_tenant_token(token)
        if verified is not None:
            authenticated_tenant_id = verified
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or forged authentication token."
            )

    # 2. Check API Key
    elif api_key_header:
        # Check against configured API keys (format: {"key": tenant_id})
        configured_keys = getattr(settings, "API_KEYS", {})
        if api_key_header in configured_keys:
            authenticated_tenant_id = configured_keys[api_key_header]
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key."
            )

    # 3. Check Signed Header (X-Tenant-ID + X-Tenant-Signature)
    elif sig_header and raw_header_tenant:
        try:
            tid = int(raw_header_tenant)
            expected_token = generate_tenant_token(tid)
            expected_sig = expected_token.split(":", 1)[1]
            if hmac.compare_digest(sig_header, expected_sig):
                authenticated_tenant_id = tid
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid tenant identity signature."
                )
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Malformed tenant header."
            )

    # 4. Raw Header Handling (Dev/Test Fallback vs Production Fail-Closed)
    if authenticated_tenant_id is None and raw_header_tenant:
        if not raw_header_tenant.isdigit():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Tenant ID must be a numeric integer."
            )
        tid = int(raw_header_tenant)

        # In production or when REQUIRE_TENANT_AUTH is explicitly enabled, reject unauthenticated claims
        if getattr(settings, "REQUIRE_TENANT_AUTH", False) or settings.ENVIRONMENT == "production":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Untrusted client tenant header. Authenticated credential or signature required."
            )
        # Development / test mode fallback
        authenticated_tenant_id = tid

    # 5. Anti-Spoofing Cross-Check
    # If the client supplied an authenticated credential AND a raw X-Tenant-ID, they MUST match
    if authenticated_tenant_id is not None and raw_header_tenant:
        if raw_header_tenant.isdigit() and int(raw_header_tenant) != authenticated_tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Claimed tenant ID does not match authenticated identity."
            )

    return authenticated_tenant_id
