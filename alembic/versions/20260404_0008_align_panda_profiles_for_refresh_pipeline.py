"""align panda_profiles schema for refresh pipeline upsert

Revision ID: 20260404_0008
Revises: 20260404_0007
Create Date: 2026-04-04 18:15:00
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260404_0008"
down_revision = "20260404_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "panda_profiles",
        sa.Column("age_years", sa.Integer(), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column("birth_year", sa.Integer(), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column("current_location", sa.Text(), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        schema="core",
    )
    op.add_column(
        "panda_profiles",
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=True,
            server_default=sa.func.now(),
        ),
        schema="core",
    )

    op.execute(
        """
        UPDATE core.panda_profiles
        SET current_location =
            CASE
                WHEN zoo_or_facility IS NULL THEN NULL
                ELSE CONCAT_WS(', ', zoo_or_facility, city_region, country)
            END,
            age_years = CASE
                WHEN birth_date IS NOT NULL
                    THEN DATE_PART('year', AGE(CURRENT_DATE, birth_date))::int
                ELSE NULL
            END,
            birth_year = CASE
                WHEN birth_date IS NOT NULL THEN EXTRACT(YEAR FROM birth_date)::int
                ELSE NULL
            END,
            ingested_at = COALESCE(ingested_at, NOW())
        """
    )


def downgrade() -> None:
    op.drop_column("panda_profiles", "ingested_at", schema="core")
    op.drop_column("panda_profiles", "raw_payload", schema="core")
    op.drop_column("panda_profiles", "current_location", schema="core")
    op.drop_column("panda_profiles", "birth_year", schema="core")
    op.drop_column("panda_profiles", "age_years", schema="core")
