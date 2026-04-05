"""add curated panda profile overrides for agent explainability

Revision ID: 20260405_0010
Revises: 20260405_0009
Create Date: 2026-04-05 17:15:00
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260405_0010"
down_revision = "20260405_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "panda_profile_overrides",
        sa.Column("canonical_name", sa.String(length=255), nullable=False),
        sa.Column("aliases", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("personality_tags", sa.Text(), nullable=True),
        sa.Column("health_notes", sa.Text(), nullable=True),
        sa.Column("agent_summary", sa.Text(), nullable=True),
        sa.Column("source_label", sa.String(length=255), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("canonical_name", name="pk_panda_profile_overrides"),
        schema="core",
    )
    op.create_index(
        "ix_core_panda_profile_overrides_source_id",
        "panda_profile_overrides",
        ["source_id"],
        unique=False,
        schema="core",
    )

    op.execute(
        """
        CREATE OR REPLACE VIEW core.panda_profiles_agent AS
        SELECT
            p.source_id,
            p.name,
            p.chinese_name,
            p.sex,
            p.age_years,
            p.birth_date,
            p.birth_year,
            p.current_location,
            p.zoo_or_facility,
            p.city_region,
            p.country,
            p.mother,
            p.father,
            COALESCE(NULLIF(o.agent_summary, ''), p.description) AS description_agent,
            p.description AS description_raw,
            p.life_journey,
            p.is_twin,
            COALESCE(NULLIF(o.personality_tags, ''), p.personality_tags) AS personality_tags_agent,
            p.personality_tags AS personality_tags_raw,
            p.breeding_notes,
            COALESCE(NULLIF(o.health_notes, ''), p.health_notes) AS health_notes_agent,
            p.health_notes AS health_notes_raw,
            p.babies_had_count,
            p.lineage,
            p.on_loan,
            p.ownership_category,
            p.status,
            o.source_label AS curated_source_label,
            o.updated_at AS curated_updated_at
        FROM core.panda_profiles p
        LEFT JOIN core.panda_profile_overrides o
          ON (
              p.source_id = o.source_id
              OR lower(trim(p.name)) = lower(trim(o.canonical_name))
          )
        """
    )


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS core.panda_profiles_agent")
    op.drop_index(
        "ix_core_panda_profile_overrides_source_id",
        table_name="panda_profile_overrides",
        schema="core",
    )
    op.drop_table("panda_profile_overrides", schema="core")
