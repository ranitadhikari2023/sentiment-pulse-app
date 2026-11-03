"""
Schemas for CSV batch import job status and results.
Enterprise-grade reliability contract for polling and ingestion monitoring.
"""

from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, ConfigDict, model_validator


class ImportJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    job_id: Optional[str] = None
    user_id: Optional[int] = None
    filename: str
    total_rows: int = 0
    processed_rows: int = 0
    successful_rows: int = 0
    inserted_rows: Optional[int] = None
    failed_rows: int = 0
    skipped_rows: int = 0
    progress: float = 0.0
    last_checkpoint_row: Optional[int] = 0
    status: str
    stage: Optional[str] = "QUEUED"
    message: Optional[str] = None
    error_summary: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @model_validator(mode="after")
    def populate_computed_fields(self) -> "ImportJobResponse":
        # Ensure job_id is synchronized with id
        if not self.job_id:
            self.job_id = self.id

        # Ensure inserted_rows matches successful_rows
        if self.inserted_rows is None:
            self.inserted_rows = self.successful_rows

        # Calculate accurate progress percentage bounded between 0.0 and 100.0
        if self.total_rows > 0:
            calc_pct = (self.processed_rows / self.total_rows) * 100.0
            self.progress = round(min(100.0, max(0.0, calc_pct)), 2)
        elif self.status == "COMPLETED":
            self.progress = 100.0
        else:
            self.progress = 0.0

        # Build consistent, human-readable status message if not already set
        if not self.message:
            if self.status == "COMPLETED":
                if self.failed_rows > 0:
                    self.message = (
                        f"Import completed with warnings: {self.successful_rows:,} inserted, "
                        f"{self.failed_rows:,} rejected."
                    )
                else:
                    self.message = f"Import completed successfully ({self.successful_rows:,} records inserted)."
            elif self.status == "FAILED":
                self.message = f"Import job failed ({self.stage or 'FAILED'})."
            elif self.status == "PENDING":
                self.message = "Import job queued in background. Waiting for worker."
            elif self.stage == "FINALIZING":
                self.message = "Finalizing batch and committing records to primary storage."
            elif self.stage == "PARSING":
                self.message = "Parsing CSV file structure and detecting delimiters."
            elif self.stage == "VALIDATING":
                self.message = "Validating columns and calculating record boundaries."
            elif self.stage == "INSERTING" or self.status == "PROCESSING":
                if self.total_rows > 0:
                    self.message = f"Processing customer feedback ({self.processed_rows:,} / {self.total_rows:,} records)."
                else:
                    self.message = "Processing customer feedback in background."
            else:
                self.message = f"Import job is {self.status.lower()}."

        return self
