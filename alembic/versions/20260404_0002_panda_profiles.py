"""add panda_profiles table

Revision ID: 20260404_0002
Revises: 20260404_0001
Create Date: 2026-04-04 15:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260404_0002"
down_revision = "20260404_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "panda_profiles",
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("chinese_name", sa.String(length=255), nullable=True),
        sa.Column("sex", sa.String(length=32), nullable=True),
        sa.Column("age_years", sa.Integer(), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("birth_year", sa.Integer(), nullable=True),
        sa.Column("current_location", sa.Text(), nullable=True),
        sa.Column("zoo_or_facility", sa.String(length=255), nullable=True),
        sa.Column("city_region", sa.String(length=255), nullable=True),
        sa.Column("country", sa.String(length=128), nullable=True),
        sa.Column("mother", sa.String(length=255), nullable=True),
        sa.Column("father", sa.String(length=255), nullable=True),
        sa.Column("babies_had_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("on_loan", sa.Boolean(), nullable=True),
        sa.Column("ownership_category", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("source_id", name="pk_panda_profiles"),
        schema="core",
    )
    op.create_index(
        "ix_core_panda_profiles_name",
        "panda_profiles",
        ["name"],
        unique=False,
        schema="core",
    )


def downgrade() -> None:
    op.drop_index("ix_core_panda_profiles_name", table_name="panda_profiles", schema="core")
    op.drop_table("panda_profiles", schema="core")
