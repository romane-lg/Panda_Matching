"""add panda profile photo metadata

Revision ID: 20260405_0013
Revises: 20260405_0012
Create Date: 2026-04-05 20:30:00
"""
from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "20260405_0013"
down_revision = "20260405_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("panda_profiles", sa.Column("photo_url", sa.Text(), nullable=True), schema="core")
    op.add_column(
        "panda_profiles",
        sa.Column("photo_source_url", sa.Text(), nullable=True),
        schema="core",
    )
    op.create_index(
        "ix_core_panda_profiles_photo_url",
        "panda_profiles",
        ["photo_url"],
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
            o.updated_at AS curated_updated_at,
            p.photo_url,
            p.photo_source_url
        FROM core.panda_profiles p
        LEFT JOIN core.panda_profile_overrides o
          ON (
              p.source_id = o.source_id
              OR lower(trim(p.name)) = lower(trim(o.canonical_name))
          )
        """
    )


def downgrade() -> None:
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
    op.drop_index(
        "ix_core_panda_profiles_photo_url",
        table_name="panda_profiles",
        schema="core",
    )
    op.drop_column("panda_profiles", "photo_source_url", schema="core")
    op.drop_column("panda_profiles", "photo_url", schema="core")
