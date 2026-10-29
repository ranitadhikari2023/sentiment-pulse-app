"""
Test suite for system reliability, observability, and probe endpoints.
Tests liveness, readiness, correlation IDs, process time headers, and clean error handling.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, base_url="https://testserver") as test_client:
        yield test_client


def test_liveness_probe(client):
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json() == {"status": "alive"}


def test_readiness_probe(client):
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["database"] == "ok"
    assert data["model"] == "ok"
    assert data["ready"] is True


def test_request_id_and_process_time_middleware(client):
    # Standard request automatically gets a UUID X-Request-ID and X-Process-Time
    res = client.get("/health")
    assert res.status_code == 200
    assert "X-Request-ID" in res.headers
    assert "X-Process-Time" in res.headers
    assert res.headers["X-Process-Time"].endswith("ms")

    # Custom incoming X-Request-ID is preserved
    custom_id = "trace-req-alpha-998877"
    res_custom = client.get("/health", headers={"X-Request-ID": custom_id})
    assert res_custom.status_code == 200
    assert res_custom.headers["X-Request-ID"] == custom_id


def test_clean_error_response_no_stack_traces(client):
    # Malformed JSON body
    res = client.post(
        "/predict",
        content="not-valid-json",
        headers={"Content-Type": "application/json"}
    )
    assert res.status_code in (400, 422)
    # Ensure raw Python traceback is not leaked in response
    assert "Traceback" not in res.text


def test_early_413_resets_request_context_and_preserves_headers(client):
    """
    Verifies that when an oversized import request is rejected early by Content-Length:
    1. Returns HTTP 413 with proper detail body.
    2. Response headers preserve incoming X-Request-ID.
    3. Response headers include X-Process-Time.
    """
    from core.config import settings

    oversized_length = settings.MAX_IMPORT_FILE_SIZE_BYTES + (2 * 1024 * 1024)
    custom_trace_id = "trace-early-413-verification"

    res = client.post(
        "/feedback/import",
        headers={
            "Content-Length": str(oversized_length),
            "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryX",
            "X-Request-ID": custom_trace_id,
        },
        content=b"dummy"
    )

    assert res.status_code == 413
    assert res.headers["X-Request-ID"] == custom_trace_id
    assert "X-Process-Time" in res.headers
    assert res.headers["X-Process-Time"].endswith("ms")
    data = res.json()
    assert "Uploaded file exceeds maximum allowed size" in data["detail"]
    assert "100 MB" in data["detail"]


@pytest.mark.anyio
async def test_middleware_direct_dispatch_restores_context_on_early_413():
    """
    Regression test: Verifies that RequestContextMiddleware.dispatch explicitly
    resets request_id_ctx via try/finally upon early 413 rejection,
    preventing context variable leaks into the running context.
    """
    from starlette.requests import Request
    from core.middleware import RequestContextMiddleware
    from core.logging import request_id_ctx
    from core.config import settings

    middleware = RequestContextMiddleware(app=None)
    custom_id = "ctx-leak-guard-test-413"
    oversized = settings.MAX_IMPORT_FILE_SIZE_BYTES + (2 * 1024 * 1024)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/feedback/import",
        "headers": [
            (b"content-length", str(oversized).encode()),
            (b"x-request-id", custom_id.encode()),
        ],
    }
    request = Request(scope)

    # 1. Ensure initial context is clean
    assert request_id_ctx.get() is None

    # 2. Execute middleware dispatch
    response = await middleware.dispatch(request, call_next=None)

    # 3. Assertions on early 413 response
    assert response.status_code == 413
    assert response.headers["X-Request-ID"] == custom_id
    assert "X-Process-Time" in response.headers

    # 4. CRITICAL: The context variable must be restored to its pre-request state (None)
    # Under the unfixed code, this leaked custom_id because reset(token) was bypassed.
    assert request_id_ctx.get() is None, "request_id_ctx must be reset to None after early 413!"

    # 5. Verify that if a caller had an existing ambient context, it is preserved
    outer_token = request_id_ctx.set("outer-ambient-context")
    try:
        response2 = await middleware.dispatch(request, call_next=None)
        assert response2.status_code == 413
        assert request_id_ctx.get() == "outer-ambient-context", (
            "Caller's ambient context must be preserved after middleware dispatch"
        )
    finally:
        request_id_ctx.reset(outer_token)
