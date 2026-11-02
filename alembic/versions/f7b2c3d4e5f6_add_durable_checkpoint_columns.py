"""add durable checkpoint and upload columns to import_jobs and feedback_staging

Revision ID: f7b2c3d4e5f6
Revises: e6a1b2c3d4e5
Create Date: 2026-11-03 00:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'e6a1b2c3d4e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "file_path" not in columns:
            op.add_column("import_jobs", sa.Column("file_path", sa.String(512), nullable=True))
        if "file_hash" not in columns:
            op.add_column("import_jobs", sa.Column("file_hash", sa.String(64), nullable=True))
        if "last_checkpoint_row" not in columns:
            op.add_column("import_jobs", sa.Column("last_checkpoint_row", sa.Integer(), nullable=False, server_default="0"))

    if "feedback_staging" in tables:
        columns = [c["name"] for c in inspector.get_columns("feedback_staging")]
        if "row_number" not in columns:
            op.add_column("feedback_staging", sa.Column("row_number", sa.Integer(), nullable=True))
            try:
                op.create_index("ix_feedback_staging_row_number", "feedback_staging", ["row_number"])
            except Exception:
                pass


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "feedback_staging" in tables:
        columns = [c["name"] for c in inspector.get_columns("feedback_staging")]
        if "row_number" in columns:
            op.drop_column("feedback_staging", "row_number")

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "last_checkpoint_row" in columns:
            op.drop_column("import_jobs", "last_checkpoint_row")
        if "file_hash" in columns:
            op.drop_column("import_jobs", "file_hash")
        if "file_path" in columns:
            op.drop_column("import_jobs", "file_path")

