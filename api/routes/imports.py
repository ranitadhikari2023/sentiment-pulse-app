"""
CSV batch import routes for customer feedback intelligence.
Handles file validation, size limits, chunked execution, and import job status polling.
No authentication required.
"""

import os
import json
import tempfile
from typing import Optional, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session

from db.session import get_db
from services.import_service import ImportService
from schemas.import_job import ImportJobResponse
from core.config import settings
from core.logging import get_logger

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


async def _upload_handler(file: UploadFile, db: Session, background_tasks: BackgroundTasks):
    temp_file = None
    temp_path = None
    ownership_transferred = False

    try:
        # Verify file extension
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

        # Read content in bounded streaming chunks (64 KB) into temporary file on disk
        CHUNK_SIZE = 64 * 1024
        total_bytes = 0
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_path = temp_file.name

        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > settings.MAX_IMPORT_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                    detail=f"Uploaded file exceeds maximum allowed size ({settings.MAX_IMPORT_FILE_SIZE_BYTES // (1024 * 1024)} MB). File exceeds the maximum permitted size of 100 MB."
                )
            temp_file.write(chunk)

        temp_file.flush()
        temp_file.close()

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty."
            )

        # Validate CSV header structure quickly before scheduling background processing
        _, header_error = ImportService.validate_csv_header(temp_path)
        if header_error:
            job = ImportService.create_import_job(db, file.filename)
            job.status = "FAILED"
            job.total_rows = 0
            job.successful_rows = 0
            job.failed_rows = 0
            job.error_summary = json.dumps([header_error])
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(job)
            return job

        # Create tracking job in PENDING state
        job = ImportService.create_import_job(db, file.filename)

        # Transfer ownership of temporary file to background worker
        background_tasks.add_task(
            ImportService.process_csv_file_background,
            job_id=job.id,
            file_path=temp_path
        )
        ownership_transferred = True
        return job

    finally:
        try:
            await file.close()
        except Exception as file_close_err:
            logger.warning(f"Error closing uploaded file: {file_close_err}")

        if not ownership_transferred:
            _cleanup_temp_file(temp_file, temp_path)


def _status_handler(job_id: str, db: Session):
    job = ImportService.get_import_job(db, job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Import job '{job_id}' not found."
        )
    return job


for r in (router, api_router):
    @r.post("", response_model=ImportJobResponse, status_code=status.HTTP_202_ACCEPTED)
    async def upload_csv_endpoint(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(..., description="CSV file with feedback text"),
        db: Session = Depends(get_db)
    ):
        """Uploads and schedules a customer feedback CSV file for background ingestion."""
        return await _upload_handler(file, db, background_tasks)

    @r.get("/{job_id}", response_model=ImportJobResponse)
    def get_job_status_endpoint(
        job_id: str,
        db: Session = Depends(get_db)
    ):
        """Retrieves import job status and progress."""
        return _status_handler(job_id, db)
