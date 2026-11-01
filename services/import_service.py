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

# Canonical field identifiers
CANONICAL_FIELD_TEXT = "text"
CANONICAL_FIELD_CATEGORY = "category"
CANONICAL_FIELD_SOURCE = "source"
CANONICAL_FIELD_RATING = "rating"

# Explicitly documented aliases for required feedback text
TEXT_COLUMN_ALIASES: Dict[str, str] = {
    # Primary aliases (documented on accepted CSV spec)
    "text": CANONICAL_FIELD_TEXT,
    "review": CANONICAL_FIELD_TEXT,
    "feedback": CANONICAL_FIELD_TEXT,
    # Extended aliases
    "review_text": CANONICAL_FIELD_TEXT,
    "reviewtext": CANONICAL_FIELD_TEXT,
    "feedback_text": CANONICAL_FIELD_TEXT,
    "feedbacktext": CANONICAL_FIELD_TEXT,
    "customer_review": CANONICAL_FIELD_TEXT,
    "customer_reviews": CANONICAL_FIELD_TEXT,
    "customer_feedback": CANONICAL_FIELD_TEXT,
    "user_review": CANONICAL_FIELD_TEXT,
    "user_reviews": CANONICAL_FIELD_TEXT,
    "user_feedback": CANONICAL_FIELD_TEXT,
    "content": CANONICAL_FIELD_TEXT,
    "comment": CANONICAL_FIELD_TEXT,
    "comments": CANONICAL_FIELD_TEXT,
    "message": CANONICAL_FIELD_TEXT,
    "body": CANONICAL_FIELD_TEXT,
    # Dataset-specific aliases (e.g. Kaggle sentiment datasets)
    "sentimenttext": CANONICAL_FIELD_TEXT,
    "sentiment_text": CANONICAL_FIELD_TEXT,
}

CATEGORY_COLUMN_ALIASES: Dict[str, str] = {
    "category": CANONICAL_FIELD_CATEGORY,
    "categories": CANONICAL_FIELD_CATEGORY,
    "tag": CANONICAL_FIELD_CATEGORY,
    "tags": CANONICAL_FIELD_CATEGORY,
    "topic": CANONICAL_FIELD_CATEGORY,
    "department": CANONICAL_FIELD_CATEGORY,
}

SOURCE_COLUMN_ALIASES: Dict[str, str] = {
    "source": CANONICAL_FIELD_SOURCE,
    "sources": CANONICAL_FIELD_SOURCE,
    "channel": CANONICAL_FIELD_SOURCE,
    "platform": CANONICAL_FIELD_SOURCE,
    "origin": CANONICAL_FIELD_SOURCE,
}

RATING_COLUMN_ALIASES: Dict[str, str] = {
    "rating": CANONICAL_FIELD_RATING,
    "ratings": CANONICAL_FIELD_RATING,
    "score": CANONICAL_FIELD_RATING,
    "stars": CANONICAL_FIELD_RATING,
    "star_rating": CANONICAL_FIELD_RATING,
}

ALL_COLUMN_ALIASES: Dict[str, str] = {
    **TEXT_COLUMN_ALIASES,
    **CATEGORY_COLUMN_ALIASES,
    **SOURCE_COLUMN_ALIASES,
    **RATING_COLUMN_ALIASES,
}

CANDIDATE_DELIMITERS: List[str] = [",", ";", "\t", "|"]


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

    @staticmethod
    def normalize_header(header: Any) -> str:
        """
        Normalizes a CSV column header:
        - Strips leading and trailing whitespace
        - Strips quotes (single and double quotes)
        - Strips UTF-8/UTF-16 BOM and zero-width/invisible characters
        - Strips Latin-1 decoded UTF-8 BOM artifact ('ï»¿')
        - Strips whitespace again after stripping BOM
        - Converts to lowercase
        """
        if header is None:
            return ""
        text = str(header).strip()
        if not text:
            return ""

        # Remove quotes if present
        text = text.strip("'\"").strip()

        # Strip BOM markers and zero-width characters (covers UTF-8 \ufeff, \ufffe, zero-width spaces)
        # Also strip 'ï»¿' which is the raw byte sequence \xef\xbb\xbf decoded as Latin-1/cp1252
        for bom_prefix in ("\ufeff", "\ufffe", "\u200b", "\u200c", "\u200d", "\u2060", "ï»¿"):
            if text.startswith(bom_prefix):
                text = text[len(bom_prefix):].strip()

        text = text.strip("'\"").strip().lower()
        return text

    @classmethod
    def resolve_canonical_field(cls, header: Any) -> Optional[str]:
        """
        Resolves a raw header string to its canonical field name ('text', 'category', 'source', 'rating').
        Matches exact normalized string, normalized with spaces/hyphens replaced by underscores,
        or base name before parentheses.
        """
        norm = cls.normalize_header(header)
        if not norm:
            return None
        if norm in ALL_COLUMN_ALIASES:
            return ALL_COLUMN_ALIASES[norm]

        # Try replacing hyphens and spaces with underscores
        slug = norm.replace("-", "_").replace(" ", "_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        if slug in ALL_COLUMN_ALIASES:
            return ALL_COLUMN_ALIASES[slug]

        # Check prefix before parentheses e.g. "rating (1-5)" -> "rating"
        base = norm.split("(")[0].strip()
        if base and base in ALL_COLUMN_ALIASES:
            return ALL_COLUMN_ALIASES[base]

        return None

    @classmethod
    def map_header_row(
        cls, header_row: List[str]
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Dict[str, int]]:
        """
        Maps header row to column indices:
        Returns (text_col_idx, category_idx, source_idx, rating_idx, header_map).
        Prioritizes primary aliases ('text', 'review', 'feedback') over extended aliases.
        First occurrence of a column name takes precedence; duplicates are safely ignored.
        Empty header cells are ignored.
        """
        col_map: Dict[str, int] = {}
        best_text_idx: Optional[int] = None
        best_text_priority: int = 999

        category_idx: Optional[int] = None
        source_idx: Optional[int] = None
        rating_idx: Optional[int] = None

        for idx, col in enumerate(header_row):
            norm = cls.normalize_header(col)
            if not norm:
                continue

            canonical = cls.resolve_canonical_field(col)
            if not canonical:
                continue

            if norm not in col_map:
                col_map[norm] = idx

            if canonical == CANONICAL_FIELD_TEXT:
                slug = norm.replace("-", "_").replace(" ", "_")
                if norm == "text":
                    prio = 1
                elif norm == "review":
                    prio = 2
                elif norm == "feedback":
                    prio = 3
                elif slug in ("review_text", "reviewtext"):
                    prio = 4
                elif slug in ("feedback_text", "feedbacktext"):
                    prio = 5
                elif "review" in slug or "feedback" in slug:
                    prio = 6
                else:
                    prio = 7

                if prio < best_text_priority:
                    best_text_priority = prio
                    best_text_idx = idx

            elif canonical == CANONICAL_FIELD_CATEGORY and category_idx is None:
                category_idx = idx
            elif canonical == CANONICAL_FIELD_SOURCE and source_idx is None:
                source_idx = idx
            elif canonical == CANONICAL_FIELD_RATING and rating_idx is None:
                rating_idx = idx

        return best_text_idx, category_idx, source_idx, rating_idx, col_map

    @classmethod
    def detect_csv_delimiter(cls, file_path: str, encoding: str) -> str:
        """
        Detects CSV delimiter (',', ';', '\\t', '|') safely and deterministically.
        Reads sample lines from the file and tests candidate delimiters against header row
        to see if any produces a valid required text column.
        Falls back to csv.Sniffer or defaults to comma ','.
        """
        sample_text = ""
        first_non_empty_line = ""
        try:
            with open(file_path, "r", encoding=encoding, newline="") as f:
                for line in f:
                    if line.strip():
                        first_non_empty_line = line
                        sample_text += line
                        if len(sample_text) >= 8192:
                            break
                        # Read next few lines for dialect sniffer context
                        for extra_line in f:
                            if extra_line.strip():
                                sample_text += extra_line
                                if len(sample_text) >= 8192:
                                    break
                        break
        except Exception as e:
            logger.warning(f"Error reading sample for delimiter detection: {e}")
            return ","

        if not first_non_empty_line:
            return ","

        # Priority 1: Check if any candidate delimiter cleanly parses the header row
        # into a column that maps to our required text field.
        for delim in CANDIDATE_DELIMITERS:
            try:
                reader = csv.reader(io.StringIO(first_non_empty_line), delimiter=delim)
                row = next(reader, None)
                if row and len(row) > 1:
                    for cell in row:
                        if cls.resolve_canonical_field(cell) == CANONICAL_FIELD_TEXT:
                            return delim
            except Exception:
                continue

        # If only 1 column, but matches text column, return comma
        try:
            reader = csv.reader(io.StringIO(first_non_empty_line), delimiter=",")
            row = next(reader, None)
            if row and any(cls.resolve_canonical_field(c) == CANONICAL_FIELD_TEXT for c in row):
                return ","
        except Exception:
            pass

        # Priority 2: Use csv.Sniffer with candidate delimiters
        if sample_text:
            try:
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample_text, delimiters=",\t;|")
                if dialect.delimiter in CANDIDATE_DELIMITERS:
                    return dialect.delimiter
            except Exception:
                pass

        # Priority 3: Count delimiter occurrences in first line
        delim_counts = {d: first_non_empty_line.count(d) for d in CANDIDATE_DELIMITERS}
        best_delim = max(delim_counts, key=delim_counts.get)
        if delim_counts[best_delim] > 0:
            return best_delim

        return ","

    @classmethod
    def validate_csv_header(cls, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fast pre-validation of CSV encoding, delimiter, and header row.
        Returns (encoding, error_message).
        If valid, returns (detected_encoding, None).
        If invalid, returns (None or encoding, descriptive_error_message).
        """
        encoding = cls._detect_file_encoding(file_path)
        if encoding is None:
            return None, "CSV encoding is not supported. Please use UTF-8, UTF-8 BOM, or Latin-1."

        try:
            delimiter = cls.detect_csv_delimiter(file_path, encoding)
            with open(file_path, "r", encoding=encoding, newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=delimiter)
                header_row = None
                for row in reader:
                    if row and any(cell.strip() for cell in row):
                        header_row = row
                        break

                if header_row is None:
                    return encoding, "CSV file is completely empty."

                text_idx, _, _, _, _ = cls.map_header_row(header_row)

                if text_idx is None:
                    return encoding, "CSV header must contain a 'text', 'review', or 'feedback' column."

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

            # Attempt decoding with fallback strategies
            encoding = cls._detect_file_encoding(file_path)
            if encoding is None:
                job.total_rows = 0
                job.processed_rows = 0
                job.successful_rows = 0
                job.failed_rows = 0
                job.status = "FAILED"
                job.error_summary = json.dumps(["CSV encoding is not supported. Please use UTF-8, UTF-8 BOM, or Latin-1."])
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                return job

            delimiter = cls.detect_csv_delimiter(file_path, encoding)

            with open(file_path, "r", encoding=encoding, newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=delimiter)
                header_row = None
                for row in reader:
                    if row and any(cell.strip() for cell in row):
                        header_row = row
                        break

                if header_row is None:
                    job.total_rows = 0
                    job.processed_rows = 0
                    job.successful_rows = 0
                    job.failed_rows = 0
                    job.status = "FAILED"
                    job.error_summary = json.dumps(["CSV file is completely empty."])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                text_col_idx, category_idx, source_idx, rating_idx, _ = cls.map_header_row(header_row)

                if text_col_idx is None:
                    job.total_rows = 0
                    job.processed_rows = 0
                    job.successful_rows = 0
                    job.failed_rows = 0
                    job.status = "FAILED"
                    job.error_summary = json.dumps(["CSV header must contain a 'text', 'review', or 'feedback' column."])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                # Count non-empty data rows in a streaming pass to bound memory while calculating exact count
                total_data_rows = 0
                for row in reader:
                    if row and any(cell.strip() for cell in row):
                        total_data_rows += 1

                if total_data_rows > settings.MAX_IMPORT_ROWS:
                    job.total_rows = total_data_rows
                    job.processed_rows = 0
                    job.successful_rows = 0
                    job.failed_rows = 0
                    job.status = "FAILED"
                    job.error_summary = json.dumps([
                        f"CSV exceeds maximum allowed rows ({total_data_rows} > {settings.MAX_IMPORT_ROWS}). Maximum batch limit is {settings.MAX_IMPORT_ROWS_FORMATTED} records."
                    ])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                if total_data_rows == 0:
                    job.total_rows = 0
                    job.processed_rows = 0
                    job.successful_rows = 0
                    job.failed_rows = 0
                    job.status = "FAILED"
                    job.error_summary = json.dumps(["CSV file contains no data rows."])
                    job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    return job

                job.total_rows = total_data_rows
                db.commit()

                # Rewind file to start of rows for bounded chunk processing
                csvfile.seek(0)
                reader = csv.reader(csvfile, delimiter=delimiter)

                # Skip blank lines and header row
                for row in reader:
                    if row and any(cell.strip() for cell in row):
                        break

                chunk_size = getattr(settings, "DB_INSERT_CHUNK_SIZE", settings.IMPORT_CHUNK_SIZE)
                errors: List[str] = []
                current_chunk: List[List[str]] = []
                processed_so_far = 0

                for row in reader:
                    # Skip empty rows safely
                    if not row or not any(cell.strip() for cell in row):
                        continue

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
                        f"CSV parsing error: field exceeds maximum allowed size (1 MB) or malformed quotes ({str(csv_err)}). CSV parsing failed. Please check the file format and quoting."
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
            job.error_summary = json.dumps([f"CSV parsing error: field exceeds maximum allowed size (1 MB) or malformed quotes ({str(csv_err)}). CSV parsing failed. Please check the file format and quoting."])
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
        Supports utf-8-sig (handles UTF-8 BOM and standard UTF-8), latin-1, and cp1252.
        Rejects unsupported encodings such as UTF-16 and binary.
        """
        # First check raw leading bytes for BOM markers
        try:
            with open(file_path, "rb") as f:
                lead = f.read(4)
                if lead.startswith(b"\xef\xbb\xbf"):
                    return "utf-8-sig"
                if lead.startswith((b"\xff\xfe", b"\xfe\xff")):
                    # UTF-16 is unsupported per spec
                    return None
        except Exception:
            pass

        # Try utf-8-sig and utf-8
        for enc in ("utf-8-sig", "utf-8"):
            try:
                with open(file_path, "r", encoding=enc, errors="strict") as f:
                    while f.read(65536):
                        pass
                return enc
            except UnicodeDecodeError:
                continue

        # Fallback to Latin-1 / cp1252
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

            # Skip empty rows inside chunk if any passed through
            if not row or not any(cell.strip() for cell in row):
                continue

            if text_col_idx >= len(row):
                if len(errors) < 100:
                    errors.append(f"Row {row_idx}: Missing text column.")
                job.failed_rows += 1
                continue

            raw_text = row[text_col_idx].strip()
            if not raw_text:
                if len(errors) < 100:
                    errors.append(f"Row {row_idx}: Empty text.")
                job.failed_rows += 1
                continue

            if len(raw_text) > MAX_FEEDBACK_TEXT_LENGTH:
                if len(errors) < 100:
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
                if len(errors) < 100:
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
