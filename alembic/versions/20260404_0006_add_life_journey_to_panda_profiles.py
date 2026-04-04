"""add life_journey column to panda_profiles

Revision ID: 20260404_0006
Revises: 20260404_0005
Create Date: 2026-04-04 16:55:00
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260404_0006"
down_revision = "20260404_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "panda_profiles",
        sa.Column("life_journey", sa.Text(), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("panda_profiles", "life_journey", schema="core")
