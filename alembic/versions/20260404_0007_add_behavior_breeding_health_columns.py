"""add twin/personality/breeding/health columns to panda_profiles

Revision ID: 20260404_0007
Revises: 20260404_0006
Create Date: 2026-04-04 17:10:00
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260404_0007"
down_revision = "20260404_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "panda_profiles",
        sa.Column("is_twin", sa.Boolean(), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column("personality_tags", sa.Text(), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column("breeding_notes", sa.Text(), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column("health_notes", sa.Text(), nullable=True),
        schema="core",
    )


def downgrade() -> None:
    op.drop_column("panda_profiles", "health_notes", schema="core")
    op.drop_column("panda_profiles", "breeding_notes", schema="core")
    op.drop_column("panda_profiles", "personality_tags", schema="core")
    op.drop_column("panda_profiles", "is_twin", schema="core")
