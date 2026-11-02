"""add feedback_staging table and stage column to import_jobs

Revision ID: e6a1b2c3d4e5
Revises: d5f3a1829e01
Create Date: 2026-11-02 20:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6a1b2c3d4e5'
down_revision: Union[str, Sequence[str], None] = 'd5f3a1829e01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create feedback_staging table and add stage column to import_jobs."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    # 1. Add stage column to import_jobs if missing
    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "stage" not in columns:
            op.add_column(
                "import_jobs",
                sa.Column("stage", sa.String(50), nullable=False, server_default="QUEUED")
            )
            try:
                op.create_index("ix_import_jobs_stage", "import_jobs", ["stage"])
            except Exception:
                pass

    # 2. Create feedback_staging table if missing
    if "feedback_staging" not in tables:
        op.create_table(
            "feedback_staging",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("import_job_id", sa.String(36), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("clean_text", sa.Text(), nullable=True),
            sa.Column("sentiment", sa.String(20), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("positive_score", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("neutral_score", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("negative_score", sa.Float(), nullable=False, server_default="0.0"),
            sa.Column("category", sa.String(50), nullable=True),
            sa.Column("source", sa.String(50), nullable=True),
            sa.Column("rating", sa.Integer(), nullable=True),
            sa.Column("model_version", sa.String(30), nullable=False, server_default="3.0.0"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_feedback_staging_import_job_id", "feedback_staging", ["import_job_id"])
        op.create_index("ix_feedback_staging_user_id", "feedback_staging", ["user_id"])


def downgrade() -> None:
    """Drop feedback_staging table and remove stage column from import_jobs."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "feedback_staging" in tables:
        op.drop_table("feedback_staging")

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "stage" in columns:
            op.drop_column("import_jobs", "stage")

