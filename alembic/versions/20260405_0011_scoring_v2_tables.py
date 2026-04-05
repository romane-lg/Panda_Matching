"""add v2 scoring feature tables

Revision ID: 20260405_0011
Revises: 20260405_0010
Create Date: 2026-04-05 18:10:00
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260405_0011"
down_revision = "20260405_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "panda_text_features",
        sa.Column("source_id", sa.Text(), nullable=False),
        sa.Column("personality_tags_norm", sa.Text(), nullable=True),
        sa.Column("health_risk_score", sa.Numeric(6, 3), nullable=False, server_default="0"),
        sa.Column("fertility_status_score", sa.Numeric(6, 3), nullable=False, server_default="0.5"),
        sa.Column("aggression_level_score", sa.Numeric(6, 3), nullable=False, server_default="0.2"),
        sa.Column(
            "past_breeding_success_score",
            sa.Numeric(6, 3),
            nullable=False,
            server_default="0.5",
        ),
        sa.Column(
            "availability_window_score",
            sa.Numeric(6, 3),
            nullable=False,
            server_default="0.7",
        ),
        sa.Column("extraction_confidence", sa.Numeric(6, 3), nullable=False, server_default="0.5"),
        sa.Column("evidence_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("source_id", name="pk_panda_text_features"),
        schema="core",
    )

    op.create_table(
        "match_scores_v2",
        sa.Column("focal_panda_id", sa.Text(), nullable=False),
        sa.Column("candidate_panda_id", sa.Text(), nullable=False),
        sa.Column("base_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("adjustment_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("final_score_v2", sa.Numeric(8, 3), nullable=False),
        sa.Column("bio_component", sa.Numeric(8, 3), nullable=False),
        sa.Column("behavior_component", sa.Numeric(8, 3), nullable=False),
        sa.Column("logistics_component", sa.Numeric(8, 3), nullable=False),
        sa.Column("pair_history_component", sa.Numeric(8, 3), nullable=False),
        sa.Column("age_gap_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("personality_compat_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("health_penalty_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("aggression_penalty_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("breeding_success_score", sa.Numeric(8, 3), nullable=False),
        sa.Column("score_breakdown_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("top_positive_factors", sa.Text(), nullable=True),
        sa.Column("top_negative_factors", sa.Text(), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint(
            "focal_panda_id",
            "candidate_panda_id",
            name="pk_match_scores_v2",
        ),
        schema="core",
    )
    op.create_index(
        "ix_core_match_scores_v2_focal",
        "match_scores_v2",
        ["focal_panda_id"],
        unique=False,
        schema="core",
    )


def downgrade() -> None:
    op.drop_index("ix_core_match_scores_v2_focal", table_name="match_scores_v2", schema="core")
    op.drop_table("match_scores_v2", schema="core")
    op.drop_table("panda_text_features", schema="core")
