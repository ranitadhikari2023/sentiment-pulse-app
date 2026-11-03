"""
FastAPI Backend for Sentiment Pulse Customer Feedback Intelligence Platform.
Features:
- Lifespan initialization for database and ONNX inference engine
- Correlation ID middleware (X-Request-ID) and latency tracking
- Production CORS and security middleware
- Customer feedback persistence, pagination, multi-criteria filtering, and deletion
- Real-time single and batch prediction endpoints
- Bounded CSV chunked imports and formula-injection-safe exports
- Business intelligence analytics and customer sentiment trends
- Separation of liveness, readiness, and monitoring health checks
- Interactive OpenAPI / Swagger documentation
"""

import os
import uuid
import json
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from sqlalchemy import delete

from core.config import settings
from core.logging import setup_logging, get_logger
from core.middleware import RequestContextMiddleware
from db.session import init_db, SessionLocal
from db.models import ImportJob, FeedbackStaging
from services.sentiment_service import sentiment_service

# Routers
from api.routes.predictions import router as predictions_router
from api.routes.feedback import router as feedback_router, api_router as api_feedback_router
from api.routes.imports import router as imports_router, api_router as api_imports_router
from api.routes.analytics import router as analytics_router, api_router as api_analytics_router
from api.routes.health import router as health_router, load_model_metrics

# Re-exported Pydantic Schemas for backward compatibility
from schemas.prediction import (
    SentimentRequest,
    SentimentScores,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
)
from schemas.health import HealthResponse

# Initialize structured logging
setup_logging()
logger = get_logger(__name__)


# Lifespan context manager for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database schema...")
    init_db()

    # Self-healing startup recovery: preserve resumable checkpoints, purge orphaned unrecoverable jobs
    try:
        import hashlib
        with SessionLocal() as db:
            stale_jobs = db.query(ImportJob).filter(ImportJob.status.in_(["PENDING", "PROCESSING"])).all()
            for j in stale_jobs:
                is_recoverable = False
                if j.file_path and os.path.exists(j.file_path):
                    if j.file_hash:
                        try:
                            with open(j.file_path, "rb") as f_chk:
                                if hashlib.sha256(f_chk.read()).hexdigest() == j.file_hash:
                                    is_recoverable = True
                        except Exception:
                            is_recoverable = False
                    else:
                        is_recoverable = True

                if is_recoverable:
                    # Preserve staging rows and checkpoint progress for safe resumption
                    j.status = "PENDING"
                    j.stage = "QUEUED"
                    j.last_checkpoint_row = j.processed_rows
                    j.updated_at = datetime.now(timezone.utc)
                    logger.info(
                        f"Preserved recoverable import job '{j.id}' at checkpoint row {j.processed_rows} "
                        f"with intact source file across server restart."
                    )
                else:
                    j.status = "FAILED"
                    j.stage = "FAILED"
                    j.error_summary = json.dumps([
                        "Import job was interrupted and source CSV file is missing or corrupted."
                    ])
                    j.completed_at = datetime.now(timezone.utc)
                    j.updated_at = datetime.now(timezone.utc)
                    db.execute(delete(FeedbackStaging).where(FeedbackStaging.import_job_id == j.id))
                    logger.warning(
                        f"Marked unrecoverable import job '{j.id}' as FAILED and purged staging records."
                    )
            if stale_jobs:
                db.commit()
                logger.info(f"Reconciled {len(stale_jobs)} interrupted import jobs on startup.")
    except Exception as recovery_err:
        logger.critical(f"Startup job recovery check failed: {recovery_err}", exc_info=True)
        raise RuntimeError(f"Startup recovery failed due to database or schema error: {recovery_err}") from recovery_err

    logger.info("Verifying SentimentAnalyzer inference engine readiness...")
    if not sentiment_service.is_ready():
        sentiment_service._initialize()
    app.state.analyzer = sentiment_service.analyzer
    logger.info("Inference engine ready. Starting API server.")
    yield
    logger.info("Sentiment Pulse application shutting down.")


app = FastAPI(
    title="Sentiment Pulse Intelligence Platform API",
    description="Customer feedback sentiment analysis platform with persistent feedback records, batch CSV ingestion, and business intelligence.",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# 1. Attach Request Context Middleware (X-Request-ID & X-Process-Time)
app.add_middleware(RequestContextMiddleware)

# 2. Attach CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Exception Handler ensuring structured JSON error responses rather than plain text
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    logger.error(f"Global unhandled error [{req_id}] processing {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred. Please try again later.",
            "request_id": req_id
        }
    )


# 3. Mount Templates & Static Files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# 4. Include Route Handlers (Mount static subpaths before parameterized /{id} paths)
app.include_router(health_router)
app.include_router(predictions_router)
app.include_router(analytics_router)
app.include_router(api_analytics_router)
app.include_router(imports_router)
app.include_router(api_imports_router)
app.include_router(feedback_router)
app.include_router(api_feedback_router)


# 5. Frontend HTML Routes (No authentication required)
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """
    Renders the web frontend interface.
    Opens directly to the dashboard without authentication.
    """
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)

    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html>
        <head><title>{settings.APP_NAME}</title></head>
        <body style="font-family: sans-serif; text-align: center; padding: 50px;">
            <h2>{settings.APP_NAME} API is Running</h2>
            <p>Customer Feedback Intelligence Platform</p>
            <p>Interactive API Docs: <a href="/docs">/docs</a> | Healthcheck: <a href="/health">/health</a></p>
        </body>
        </html>
        """,
        status_code=200
    )




# Helper function kept for backwards compatibility
def get_model_metrics() -> Dict[str, Any]:
    return load_model_metrics()


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
