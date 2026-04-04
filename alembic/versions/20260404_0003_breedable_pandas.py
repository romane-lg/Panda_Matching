"""add breedable_pandas table

Revision ID: 20260404_0003
Revises: 20260404_0002
Create Date: 2026-04-04 16:05:00
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260404_0003"
down_revision = "20260404_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "breedable_pandas",
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("sex", sa.String(length=32), nullable=True),
        sa.Column("age_years", sa.Integer(), nullable=False),
        sa.Column("babies_had_count", sa.Integer(), nullable=False),
        sa.Column("eligible_rule", sa.String(length=255), nullable=False),
        sa.Column(
            "refreshed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("source_id", name="pk_breedable_pandas"),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["core.panda_profiles.source_id"],
            name="fk_breedable_pandas_source_id_panda_profiles",
            ondelete="CASCADE",
        ),
        schema="core",
    )

    op.execute(
        """
        INSERT INTO core.breedable_pandas (
            source_id,
            name,
            sex,
            age_years,
            babies_had_count,
            eligible_rule
        )
        SELECT
            source_id,
            name,
            sex,
            DATE_PART('year', AGE(CURRENT_DATE, birth_date))::int AS age_years,
            babies_had_count,
            'age_years >= 4 AND babies_had_count <= 8' AS eligible_rule
        FROM core.panda_profiles
        WHERE birth_date IS NOT NULL
          AND DATE_PART('year', AGE(CURRENT_DATE, birth_date)) >= 4
          AND babies_had_count <= 8
        """
    )


def downgrade() -> None:
    op.drop_table("breedable_pandas", schema="core")
