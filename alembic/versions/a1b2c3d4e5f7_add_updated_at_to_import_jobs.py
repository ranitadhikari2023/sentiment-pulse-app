"""add updated_at column to import_jobs for finalization lease tracking

Revision ID: a1b2c3d4e5f7
Revises: f7b2c3d4e5f6
Create Date: 2026-11-03 02:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, Sequence[str], None] = 'f7b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == "sqlite"
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "updated_at" not in columns:
            # Step 1: Add updated_at as nullable WITHOUT server_default so existing rows remain NULL initially
            op.add_column(
                "import_jobs",
                sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True)
            )
            # Step 2: Backfill existing records from their historical created_at timestamp
            conn.execute(sa.text(
                "UPDATE import_jobs SET updated_at = COALESCE(created_at, CURRENT_TIMESTAMP) WHERE updated_at IS NULL"
            ))
            # Step 3: Apply permanent server default for all future inserts
            if is_sqlite:
                with op.batch_alter_table("import_jobs", schema=None) as batch_op:
                    batch_op.alter_column("updated_at", server_default=sa.func.now())
            else:
                conn.execute(sa.text("ALTER TABLE import_jobs ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP"))
        else:
            # If column already exists (e.g. from runtime sync), backfill any NULL values
            conn.execute(sa.text(
                "UPDATE import_jobs SET updated_at = COALESCE(created_at, CURRENT_TIMESTAMP) WHERE updated_at IS NULL"
            ))
            if not is_sqlite:
                try:
                    conn.execute(sa.text("ALTER TABLE import_jobs ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP"))
                except Exception:
                    pass

        # Defensively reconcile any missing columns from earlier revisions
        if "stage" not in columns:
            op.add_column("import_jobs", sa.Column("stage", sa.String(50), server_default="QUEUED", nullable=False))
        if "file_path" not in columns:
            op.add_column("import_jobs", sa.Column("file_path", sa.String(512), nullable=True))
        if "file_hash" not in columns:
            op.add_column("import_jobs", sa.Column("file_hash", sa.String(64), nullable=True))
        if "last_checkpoint_row" not in columns:
            op.add_column("import_jobs", sa.Column("last_checkpoint_row", sa.Integer(), server_default="0", nullable=False))

    if "feedback_staging" in tables:
        staging_cols = [c["name"] for c in inspector.get_columns("feedback_staging")]
        if "row_number" not in staging_cols:
            op.add_column("feedback_staging", sa.Column("row_number", sa.Integer(), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "updated_at" in columns:
            with op.batch_alter_table("import_jobs", schema=None) as batch_op:
                batch_op.drop_column("updated_at")
