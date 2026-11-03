"""
CSV batch import routes for customer feedback intelligence.
Handles file validation, size limits, chunked execution, and import job status polling.
No authentication required.
"""

import os
import json
import uuid
import hashlib
import tempfile
from typing import Optional, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Request, Response
from sqlalchemy.orm import Session
from sqlalchemy.exc import ProgrammingError, OperationalError, InternalError, SQLAlchemyError

from db.session import get_db
from db.models import ImportJob
from services.import_service import ImportService
from schemas.import_job import ImportJobResponse
from core.config import settings
from core.logging import get_logger
from core.security import resolve_trusted_tenant_identity

logger = get_logger(__name__)

router = APIRouter(prefix="/feedback/import", tags=["Feedback Import"])
api_router = APIRouter(prefix="/api/feedback/import", tags=["Feedback Import"])


def _cleanup_temp_file(temp_file: Optional[Any], temp_path: Optional[str]) -> None:
    """
    Safely closes the temporary file handle if open, then deletes the file on disk.
    Ensures Windows file handles are fully released before unlink is attempted,
    preventing [WinError 32] sharing violations and orphaned files.
    Logs warnings and errors instead of silently swallowing cleanup failures.
    """
    if temp_file is not None:
        try:
            if not getattr(temp_file, "closed", False):
                temp_file.close()
        except Exception as close_err:
            logger.warning(f"Error closing temporary file handle {temp_path}: {close_err}")

    if temp_path and os.path.exists(temp_path):
        try:
            os.unlink(temp_path)
        except OSError as unlink_err:
            logger.error(f"Failed to unlink temporary file {temp_path}: {unlink_err}")


def log_import_event(
    request_id: str,
    endpoint: str,
    status_code: int,
    job_id: Optional[str] = None,
    file_size: Optional[int] = None,
    stage: Optional[str] = None,
    exc: Optional[Exception] = None,
) -> None:
    """
    Structured request-level logger using correlation ID.
    Never logs CSV contents, secrets, database credentials, or customer PII.
    """
    exc_type = type(exc).__name__ if exc else None
    log_msg = (
        f"Import Request [{request_id}] | Endpoint: {endpoint} | Status: {status_code} | "
        f"JobID: {job_id or 'none'} | FileSize: {file_size if file_size is not None else 'unknown'} | "
        f"Stage: {stage or 'unknown'}"
    )
    if exc:
        log_msg += f" | ExcType: {exc_type}: {exc}"
        logger.error(log_msg, exc_info=True)
    else:
        logger.info(log_msg)


async def _upload_handler(
    request: Request,
    file: UploadFile,
    db: Session,
    background_tasks: BackgroundTasks,
    response: Optional[Response] = None,
):
    if response is not None:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    endpoint = request.url.path
    job_id = None
    stage = "RECEIVED"
    total_bytes = 0
    temp_file = None
    temp_path = None
    ownership_transferred = False

    try:
        # Resolve trusted tenant identity
        stage = "TENANT_RESOLVE"
        user_id = resolve_trusted_tenant_identity(request)

        # Enforce concurrent running imports limit (globally and per tenant)
        stage = "CONCURRENCY_CHECK"
        max_concurrent = settings.MAX_CONCURRENT_IMPORT_JOBS
        max_per_tenant = settings.MAX_CONCURRENT_PER_TENANT

        if user_id is not None:
            tenant_active = db.query(ImportJob).filter(
                ImportJob.user_id == user_id,
                ImportJob.status.in_(["PENDING", "PROCESSING"])
            ).count()
            if tenant_active >= max_per_tenant:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Concurrent import limit reached for tenant ({tenant_active}/{max_per_tenant}). Please wait for active jobs to complete."
                )

        active_jobs = db.query(ImportJob).filter(ImportJob.status.in_(["PENDING", "PROCESSING"])).count()
        if active_jobs >= max_concurrent:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Concurrent import limit reached ({active_jobs}/{max_concurrent}). Please wait for active jobs to complete."
            )

        # Verify file extension
        stage = "EXTENSION_CHECK"
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are accepted for feedback imports."
            )

        # Fast boundary check if size attribute is populated by ASGI/multipart layer
        if file.size is not None and file.size > settings.MAX_IMPORT_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                detail=f"Uploaded file exceeds maximum allowed size ({settings.MAX_IMPORT_FILE_SIZE_BYTES // (1024 * 1024)} MB). File exceeds the maximum permitted size of 100 MB."
            )

        # Read content in bounded streaming chunks (64 KB) into temporary/durable upload storage while computing SHA-256
        stage = "STREAMING_UPLOAD"
        CHUNK_SIZE = 64 * 1024
        total_bytes = 0
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        job_id = str(uuid.uuid4())
        hasher = hashlib.sha256()

        temp_file = tempfile.NamedTemporaryFile(dir=settings.UPLOAD_DIR, delete=False, suffix=".csv")
        temp_path = temp_file.name

        try:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > settings.MAX_IMPORT_FILE_SIZE_BYTES:
                    _cleanup_temp_file(temp_file, temp_path)
                    temp_file = None
                    temp_path = None
                    raise HTTPException(
                        status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                        detail=f"Uploaded file exceeds maximum allowed size ({settings.MAX_IMPORT_FILE_SIZE_BYTES // (1024 * 1024)} MB). File exceeds the maximum permitted size of 100 MB."
                    )
                temp_file.write(chunk)
                hasher.update(chunk)
            temp_file.flush()
        except Exception:
            _cleanup_temp_file(temp_file, temp_path)
            temp_file = None
            temp_path = None
            raise

        if total_bytes == 0:
            _cleanup_temp_file(temp_file, temp_path)
            temp_file = None
            temp_path = None
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty."
            )

        # Close the file handle before passing to validator/worker (critical for Windows)
        _cleanup_temp_file(temp_file, None)
        temp_file = None
        file_hash = hasher.hexdigest()

        # Validate CSV header structure quickly before scheduling background processing
        stage = "VALIDATING_HEADER"
        _, header_error = ImportService.validate_csv_header(temp_path)
        if header_error:
            stage = "RECORDING_HEADER_ERROR"
            job = ImportService.create_import_job(
                db, file.filename, user_id=user_id, job_id=job_id, file_path=temp_path, file_hash=file_hash
            )
            job.status = "FAILED"
            job.stage = "FAILED"
            job.total_rows = 0
            job.successful_rows = 0
            job.failed_rows = 0
            job.error_summary = json.dumps([header_error])
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(job)
            _cleanup_temp_file(None, temp_path)
            temp_path = None
            log_import_event(
                request_id=req_id,
                endpoint=endpoint,
                status_code=status.HTTP_202_ACCEPTED,
                job_id=job.id,
                file_size=total_bytes,
                stage="REJECTED_HEADER",
            )
            return job

        # Create tracking job in PENDING state with durable file path and SHA-256 fingerprint
        stage = "CREATING_JOB"
        job = ImportService.create_import_job(
            db, file.filename, user_id=user_id, job_id=job_id, file_path=temp_path, file_hash=file_hash
        )

        # Transfer ownership of durable file to background worker
        stage = "SCHEDULING_WORKER"
        background_tasks.add_task(
            ImportService.process_csv_file_background,
            job_id=job.id,
            file_path=temp_path
        )
        ownership_transferred = True

        log_import_event(
            request_id=req_id,
            endpoint=endpoint,
            status_code=status.HTTP_202_ACCEPTED,
            job_id=job.id,
            file_size=total_bytes,
            stage=job.stage,
        )
        return job

    except HTTPException as he:
        log_import_event(
            request_id=req_id,
            endpoint=endpoint,
            status_code=he.status_code,
            job_id=job_id,
            file_size=total_bytes or getattr(file, "size", None),
            stage=stage,
            exc=he,
        )
        raise he
    except (ProgrammingError, InternalError) as db_schema_err:
        log_import_event(
            request_id=req_id,
            endpoint=endpoint,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            job_id=job_id,
            file_size=total_bytes,
            stage=stage,
            exc=db_schema_err,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "Database schema error: required column is missing. Please ensure database migrations are up to date.",
                "error_type": "database_schema_error",
                "request_id": req_id,
                "job_id": job_id,
            }
        )
    except OperationalError as db_conn_err:
        log_import_event(
            request_id=req_id,
            endpoint=endpoint,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            job_id=job_id,
            file_size=total_bytes,
            stage=stage,
            exc=db_conn_err,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "detail": "Database connection failure: unable to connect to database. Please try again shortly.",
                "error_type": "database_connection_error",
                "request_id": req_id,
                "job_id": job_id,
            }
        )
    except (OSError, IOError) as io_err:
        log_import_event(
            request_id=req_id,
            endpoint=endpoint,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            job_id=job_id,
            file_size=total_bytes,
            stage=stage,
            exc=io_err,
        )
        raise io_err
    except Exception as exc:
        log_import_event(
            request_id=req_id,
            endpoint=endpoint,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            job_id=job_id,
            file_size=total_bytes,
            stage=stage,
            exc=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "detail": "An internal server error occurred while processing the import.",
                "error_type": "server_processing_error",
                "request_id": req_id,
                "job_id": job_id,
            }
        )
    finally:
        try:
            await file.close()
        except Exception as file_close_err:
            logger.warning(f"Error closing uploaded file: {file_close_err}")

        if not ownership_transferred and temp_path is not None:
            _cleanup_temp_file(temp_file, temp_path)


def _status_handler(request: Request, job_id: str, db: Session, response: Optional[Response] = None):
    if response is not None:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    job = ImportService.get_import_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Import job '{job_id}' not found."
        )

    # Multi-tenant isolation: verify ownership if job is associated with a specific user/tenant
    user_id = resolve_trusted_tenant_identity(request)
    if job.user_id is not None:
        if user_id is None or user_id != job.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Import job belongs to another tenant."
            )

    return job


def _resume_handler(
    request: Request,
    job_id: str,
    background_tasks: BackgroundTasks,
    db: Session,
    response: Optional[Response] = None,
):
    if response is not None:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"

    job = ImportService.get_import_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Import job '{job_id}' not found."
        )

    # Multi-tenant isolation: verify ownership if job is associated with a specific user/tenant
    user_id = resolve_trusted_tenant_identity(request)
    if job.user_id is not None:
        if user_id is None or user_id != job.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: Import job belongs to another tenant."
            )

    try:
        return ImportService.resume_import_job(db, job_id, background_tasks=background_tasks)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )


for r in (router, api_router):
    @r.post("", response_model=ImportJobResponse, status_code=status.HTTP_202_ACCEPTED)
    async def upload_csv_endpoint(
        request: Request,
        response: Response,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="CSV file with feedback text"),
        db: Session = Depends(get_db)
    ):
        """Uploads and schedules a customer feedback CSV file for background ingestion."""
        return await _upload_handler(request, file, db, background_tasks, response=response)

    @r.get("/{job_id}", response_model=ImportJobResponse)
    def get_job_status_endpoint(
        request: Request,
        response: Response,
        job_id: str,
        db: Session = Depends(get_db)
    ):
        """Retrieves import job status and progress."""
        return _status_handler(request, job_id, db, response=response)

    @r.post("/{job_id}/resume", response_model=ImportJobResponse, status_code=status.HTTP_202_ACCEPTED)
    def resume_job_endpoint(
        request: Request,
        response: Response,
        job_id: str,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
    ):
        """Resumes an interrupted or paused customer feedback import job from its last checkpoint."""
        return _resume_handler(request, job_id, background_tasks, db, response=response)
