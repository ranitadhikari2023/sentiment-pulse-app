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
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "updated_at" not in columns:
            op.add_column("import_jobs", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    if "import_jobs" in tables:
        columns = [c["name"] for c in inspector.get_columns("import_jobs")]
        if "updated_at" in columns:
            op.drop_column("import_jobs", "updated_at")

