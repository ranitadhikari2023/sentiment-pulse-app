"""
CSV import processing pipeline and secure export service.
Includes bounded chunking, encoding fallbacks, ImportJob state tracking, and CSV formula injection sanitization.
Authentication and user-isolation checks have been permanently removed.
"""

import io
import os
import csv
import uuid
import json
import tempfile
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple, Optional
from sqlalchemy.orm import Session

from db.models import Feedback, ImportJob
from core.config import settings
from core.logging import get_logger
from services.sentiment_service import sentiment_service

logger = get_logger(__name__)

# Bounded CSV field limit: 1 MB (1,048,576 bytes) prevents memory explosion from runaway
# unclosed quotes or massive fields, while easily accommodating long customer feedback text.
MAX_CSV_FIELD_SIZE = 1024 * 1024  # 1 MB
try:
    csv.field_size_limit(MAX_CSV_FIELD_SIZE)
except (OverflowError, AttributeError):
    pass

# Maximum allowed characters for individual feedback text field in CSV rows.
# Consistent with application schemas (schemas/feedback.py caps at 5,000, bulk.js at 2,000),
# with generous 10,000 character allowance for detailed customer reviews/transcripts.
MAX_FEEDBACK_TEXT_LENGTH = 10000


def sanitize_csv_cell(value: Any) -> str:
    """
    Prevents CSV Formula Injection (Excel Macro Execution) by prepending a single quote
    if the cell value begins with dangerous characters: =, +, -, @, tab, or carriage return.
    """
    if value is None:
        return ""
    text_val = str(value)
    if text_val and text_val[0] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{text_val}"
    return text_val


class ImportService:
    @staticmethod
    def create_import_job(db: Session, filename: str) -> ImportJob:
        job = ImportJob(
            id=str(uuid.uuid4()),
            user_id=None,
            filename=filename,
            total_rows=0,
            processed_rows=0,
            successful_rows=0,
            failed_rows=0,
            status="PENDING",
            created_at=datetime.now(timezone.utc)
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get_import_job(db: Session, job_id: str) -> Optional[ImportJob]:
        """Retrieval of import job by ID."""
        return db.query(ImportJob).filter(ImportJob.id == job_id).first()

    @classmethod
    def validate_csv_header(cls, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fast pre-validation of CSV encoding and header row.
        Returns (encoding, error_message).
        If valid, returns (detected_encoding, None).
        If invalid, returns (None or encoding, descriptive_error_message).
        """
        encoding = cls._detect_file_encoding(file_path)
        if encoding is None:
            return None, "Failed to decode CSV file: unsupported encoding."

        try:
            with open(file_path, "r", encoding=encoding, newline="") as csvfile:
                reader = csv.reader(csvfile)
                try:
                    header_row = next(reader)
                except StopIteration:
                    return encoding, "CSV file is completely empty."

                header_map = {}
                for idx, col in enumerate(header_row):
                    col_cleaned = col.strip().lstrip("\ufeff\ufffe").lower()
                    header_map[col_cleaned] = idx

                has_text_col = any(
                    candidate in header_map
                    for candidate in ("text", "feedback", "review", "content", "comment")
                )
                if not has_text_col:
                    return encoding, "CSV must contain a 'text', 'review', or 'feedback' column."

                return encoding, None
        except Exception as exc:
            return encoding, f"Malformed CSV header structure: {str(exc)}"

    @classmethod
    def process_csv_file_background(cls, job_id: str, file_path: str) -> None:
        """
        Asynchronous background task handler for CSV ingestion.
        Opens its own dedicated database session using SessionLocal, reloads
        the ImportJob, processes the CSV file, and ensures reliable session closure
        and temporary file cleanup.
        """
        from db.session import SessionLocal

        db = SessionLocal()
        try:
            job = db.query(ImportJob).filter(ImportJob.id == job_id).first()
            if not job:
                logger.error(f"Background task: ImportJob '{job_id}' not found.")
                return

            cls.process_csv_file(db, job, file_path)
        except Exception as exc:
            logger.error(f"Unhandled exception in background import job '{job_id}': {exc}", exc_info=True)
            try:
                db.rollback()
                failed_job = db.query(ImportJob).filter(ImportJob.id == job_id).first()
                if failed_job and failed_job.status not in ("COMPLETED", "FAILED"):
                    failed_job.status = "FAILED"
                    failed_job.successful_rows = 0
                    failed_job.failed_rows = failed_job.total_rows or 0
                    failed_job.error_summary = json.dumps([
                        f"Background task execution failed: {str(exc)}"
                    ])
                    failed_job.completed_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

            # Guarantees deletion of the temporary file once processing completes or fails
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except OSError as unlink_err:
                    logger.error(f"Background task failed to unlink temporary file {file_path}: {unlink_err}")

    @classmethod
    def process_csv_bytes(cls, db: Session, job: ImportJob, content: bytes) -> ImportJob:
        """
        Parses and processes CSV bytes, writing to a temporary file to keep execution bounded,
        delegating to process_csv_file for backward compatibility.
        """
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()
            return cls.process_csv_file(db, job, tmp.name)
        finally:
            if os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except OSError as unlink_err:
                    logger.error(f"Failed to unlink temporary file {tmp.name}: {unlink_err}")

    @classmethod
    def process_csv_file(cls, db: Session, job: ImportJob, file_path: str) -> ImportJob:
        """
        Parses and processes CSV from file in bounded chunks, streaming rows to avoid
        unbounded memory usage, creating Feedback records within an atomic transaction.
        All records are flushed in chunks to keep memory bounded, and committed atomically
        at the end. On any transaction failure, all inserted records are rolled back.
        """
        try:
            job.status = "PROCESSING"
            db.commit()

            # Attempt decoding with fallback strategies in 64 KB verification blocks
            encoding = cls._detect_file_encoding(file_path)
            if encoding is None:
                job.status = "FAILED"
                job.error_summary = json.dumps(["Failed to decode CSV file: unsupported encoding."])
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return job

            with open(file_path, "r", encoding=encoding, newline="") as csvfile:
                reader = csv.reader(csvfile)
                try:
                    header_row = next(reader)
                except StopIteration:
                    job.status = "FAILED"
                    job.error_summary = json.dumps(["CSV file is completely empty."])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                # Normalize header indices and strip zero-width BOM markers if present
                header_map: Dict[str, int] = {}
                for idx, col in enumerate(header_row):
                    col_cleaned = col.strip().lstrip("\ufeff\ufffe").lower()
                    header_map[col_cleaned] = idx

                # Locate text column
                text_col_idx = None
                for candidate in ("text", "feedback", "review", "content", "comment"):
                    if candidate in header_map:
                        text_col_idx = header_map[candidate]
                        break

                if text_col_idx is None:
                    job.status = "FAILED"
                    job.error_summary = json.dumps(["CSV must contain a 'text', 'review', or 'feedback' column."])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                category_idx = header_map.get("category")
                source_idx = header_map.get("source")
                rating_idx = header_map.get("rating")

                # Count rows in a streaming pass to bound memory while calculating exact count
                total_data_rows = sum(1 for _ in reader)
                if total_data_rows > settings.MAX_IMPORT_ROWS:
                    job.status = "FAILED"
                    job.error_summary = json.dumps([
                        f"CSV exceeds maximum allowed rows ({total_data_rows} > {settings.MAX_IMPORT_ROWS})."
                    ])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                if total_data_rows == 0:
                    job.status = "FAILED"
                    job.error_summary = json.dumps(["CSV file contains no data rows."])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                job.total_rows = total_data_rows
                db.commit()

                # Rewind file to start of rows for bounded chunk processing
                csvfile.seek(0)
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row

                chunk_size = settings.IMPORT_CHUNK_SIZE
                errors: List[str] = []
                current_chunk: List[List[str]] = []
                processed_so_far = 0

                for row in reader:
                    current_chunk.append(row)
                    if len(current_chunk) >= chunk_size:
                        cls._process_row_chunk(
                            db=db,
                            job=job,
                            chunk=current_chunk,
                            start_row_idx=processed_so_far + 1,
                            text_col_idx=text_col_idx,
                            category_idx=category_idx,
                            source_idx=source_idx,
                            rating_idx=rating_idx,
                            errors=errors
                        )
                        processed_so_far += len(current_chunk)
                        current_chunk = []

                if current_chunk:
                    cls._process_row_chunk(
                        db=db,
                        job=job,
                        chunk=current_chunk,
                        start_row_idx=processed_so_far + 1,
                        text_col_idx=text_col_idx,
                        category_idx=category_idx,
                        source_idx=source_idx,
                        rating_idx=rating_idx,
                        errors=errors
                    )
                    processed_so_far += len(current_chunk)

                job.status = "COMPLETED" if job.failed_rows < job.total_rows else "FAILED"
                if errors:
                    job.error_summary = json.dumps(errors[:50])  # Cap at 50 errors in summary
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(job)
                return job

        except csv.Error as csv_err:
            logger.error(f"CSV parsing error in job {job.id}: {csv_err}")
            db.rollback()
            try:
                failed_job = db.query(ImportJob).filter(ImportJob.id == job.id).first()
                if failed_job:
                    failed_job.status = "FAILED"
                    failed_job.successful_rows = 0
                    failed_job.failed_rows = failed_job.total_rows or job.total_rows
                    failed_job.error_summary = json.dumps([
                        f"CSV parsing error: field exceeds maximum allowed size (1 MB) or malformed quotes ({str(csv_err)})."
                    ])
                    failed_job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    db.refresh(failed_job)
                    return failed_job
            except Exception:
                pass
            job.status = "FAILED"
            job.successful_rows = 0
            job.failed_rows = job.total_rows
            job.error_summary = json.dumps([f"CSV parsing error: {str(csv_err)}"])
            return job

        except Exception as exc:
            logger.error(f"Error processing import job {job.id}: {exc}", exc_info=True)
            db.rollback()
            try:
                failed_job = db.query(ImportJob).filter(ImportJob.id == job.id).first()
                if failed_job:
                    failed_job.status = "FAILED"
                    # Atomic rollback policy: 0 records persisted when transaction fails
                    failed_job.successful_rows = 0
                    failed_job.failed_rows = failed_job.total_rows or job.total_rows
                    failed_job.error_summary = json.dumps([
                        f"Database transaction failure, import rolled back completely: {str(exc)}"
                    ])
                    failed_job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    db.refresh(failed_job)
                    return failed_job
            except Exception:
                pass
            job.status = "FAILED"
            job.successful_rows = 0
            job.failed_rows = job.total_rows
            job.error_summary = json.dumps([f"Import processing failed: {str(exc)}"])
            return job

    @staticmethod
    def _detect_file_encoding(file_path: str) -> Optional[str]:
        """
        Detects valid encoding by reading small chunks without loading the entire file into memory.
        Supports utf-8-sig (handles UTF-8 BOM), utf-8, latin-1, and cp1252.
        """
        for enc in ("utf-8-sig", "utf-8"):
            try:
                with open(file_path, "r", encoding=enc, errors="strict") as f:
                    while f.read(65536):
                        pass
                return enc
            except UnicodeDecodeError:
                continue

        for enc in ("latin-1", "cp1252"):
            try:
                with open(file_path, "r", encoding=enc, errors="strict") as f:
                    while f.read(65536):
                        pass
                return enc
            except Exception:
                continue
        return None

    @staticmethod
    def _process_row_chunk(
        db: Session,
        job: ImportJob,
        chunk: List[List[str]],
        start_row_idx: int,
        text_col_idx: int,
        category_idx: Optional[int],
        source_idx: Optional[int],
        rating_idx: Optional[int],
        errors: List[str],
    ) -> None:
        new_records: List[Feedback] = []

        for offset, row in enumerate(chunk):
            row_idx = start_row_idx + offset
            if text_col_idx >= len(row):
                errors.append(f"Row {row_idx}: Missing text column.")
                job.failed_rows += 1
                continue

            raw_text = row[text_col_idx].strip()
            if not raw_text:
                errors.append(f"Row {row_idx}: Empty text.")
                job.failed_rows += 1
                continue

            if len(raw_text) > MAX_FEEDBACK_TEXT_LENGTH:
                errors.append(
                    f"Row {row_idx}: Text exceeds maximum allowed length ({MAX_FEEDBACK_TEXT_LENGTH:,} characters)."
                )
                job.failed_rows += 1
                continue

            cat_val = (
                row[category_idx].strip()
                if category_idx is not None and category_idx < len(row)
                else None
            )
            src_val = (
                row[source_idx].strip()
                if source_idx is not None and source_idx < len(row)
                else "csv_import"
            )
            rating_val = None
            if rating_idx is not None and rating_idx < len(row) and row[rating_idx].strip():
                try:
                    parsed_r = int(float(row[rating_idx].strip()))
                    if 1 <= parsed_r <= 5:
                        rating_val = parsed_r
                except ValueError:
                    pass

            try:
                pred = sentiment_service.predict(raw_text)
                scores = pred.get("scores", {})
                fb = Feedback(
                    user_id=None,
                    text=raw_text,
                    clean_text=pred.get("text", raw_text),
                    sentiment=pred.get("sentiment", "unknown"),
                    confidence=float(pred.get("confidence", 0.0)),
                    positive_score=float(scores.get("positive", 0.0)),
                    neutral_score=float(scores.get("neutral", 0.0)),
                    negative_score=float(scores.get("negative", 0.0)),
                    category=cat_val[:50] if cat_val else None,
                    source=src_val[:50] if src_val else "csv_import",
                    rating=rating_val,
                    model_version="3.0.0",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
                new_records.append(fb)
                job.successful_rows += 1
            except Exception as e:
                errors.append(f"Row {row_idx}: Inference failed - {str(e)}")
                job.failed_rows += 1

        if new_records:
            db.add_all(new_records)
            db.flush()
            # Bounded memory: expunge persisted Feedback records from session identity map
            for fb in new_records:
                db.expunge(fb)

        job.processed_rows += len(chunk)

    @staticmethod
    def generate_csv_export(db: Session) -> str:
        """
        Generates CSV string of all feedback records with formula injection protection.
        """
        records = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "id", "text", "sentiment", "confidence", "positive_score", "neutral_score",
            "negative_score", "category", "source", "rating", "model_version", "created_at"
        ])

        for r in records:
            writer.writerow([
                sanitize_csv_cell(r.id),
                sanitize_csv_cell(r.text),
                sanitize_csv_cell(r.sentiment),
                sanitize_csv_cell(f"{r.confidence:.4f}"),
                sanitize_csv_cell(f"{r.positive_score:.4f}"),
                sanitize_csv_cell(f"{r.neutral_score:.4f}"),
                sanitize_csv_cell(f"{r.negative_score:.4f}"),
                sanitize_csv_cell(r.category),
                sanitize_csv_cell(r.source),
                sanitize_csv_cell(r.rating),
                sanitize_csv_cell(r.model_version),
                sanitize_csv_cell(r.created_at.isoformat() if r.created_at else "")
            ])

        return output.getvalue()
