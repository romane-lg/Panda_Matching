"""change source_id columns from integer to text

Revision ID: 20260405_0009
Revises: 20260404_0008
Create Date: 2026-04-05 11:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "20260405_0009"
down_revision = "20260404_0008"
branch_labels = None
depends_on = None


def _capture_dependent_view_defs(bind: sa.Connection) -> list[tuple[int, str, str]]:
    # Stable dependency order observed in this project:
    # panda_profiles_clean + matching_features + breedeable_pandas
    # -> candidate_pairs -> recommended_matches
    # -> directional_recommended_matches -> ranked_directional_recommended_matches
    ordered_views = [
        "core.panda_profiles_clean",
        "core.matching_features",
        "core.breedeable_pandas",
        "core.candidate_pairs",
        "core.recommended_matches",
        "core.directional_recommended_matches",
        "core.ranked_directional_recommended_matches",
    ]

    captured: list[tuple[int, str, str]] = []
    for depth, fq_name in enumerate(ordered_views):
        view_def = bind.execute(
            sa.text("SELECT pg_get_viewdef(to_regclass(:fq_name), true)"),
            {"fq_name": fq_name},
        ).scalar_one_or_none()
        if view_def:
            captured.append((depth, fq_name, str(view_def)))

    return captured


def _drop_views_descending(captured_views: list[tuple[int, str, str]]) -> None:
    for _, fq_name, _ in sorted(captured_views, key=lambda x: x[0], reverse=True):
        op.execute(f"DROP VIEW IF EXISTS {fq_name}")


def _recreate_views_ascending(captured_views: list[tuple[int, str, str]]) -> None:
    for _, fq_name, view_def in sorted(captured_views, key=lambda x: x[0]):
        op.execute(f"CREATE VIEW {fq_name} AS {view_def}")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    core_tables = set(inspector.get_table_names(schema="core"))
    has_breedable = "breedable_pandas" in core_tables
    captured_views = _capture_dependent_view_defs(bind)
    _drop_views_descending(captured_views)

    if has_breedable:
        fks = inspector.get_foreign_keys("breedable_pandas", schema="core")
        fk_names = {fk.get("name") for fk in fks}
        if "fk_breedable_pandas_source_id_panda_profiles" in fk_names:
            op.drop_constraint(
                "fk_breedable_pandas_source_id_panda_profiles",
                "breedable_pandas",
                schema="core",
                type_="foreignkey",
            )

    op.alter_column(
        "panda_profiles",
        "source_id",
        schema="core",
        existing_type=sa.Integer(),
        type_=sa.Text(),
        existing_nullable=False,
        postgresql_using="source_id::text",
    )

    if has_breedable:
        op.alter_column(
            "breedable_pandas",
            "source_id",
            schema="core",
            existing_type=sa.Integer(),
            type_=sa.Text(),
            existing_nullable=False,
            postgresql_using="source_id::text",
        )

        op.create_foreign_key(
            "fk_breedable_pandas_source_id_panda_profiles",
            "breedable_pandas",
            "panda_profiles",
            ["source_id"],
            ["source_id"],
            source_schema="core",
            referent_schema="core",
            ondelete="CASCADE",
        )

    _recreate_views_ascending(captured_views)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    core_tables = set(inspector.get_table_names(schema="core"))
    has_breedable = "breedable_pandas" in core_tables
    captured_views = _capture_dependent_view_defs(bind)
    _drop_views_descending(captured_views)

    if has_breedable:
        fks = inspector.get_foreign_keys("breedable_pandas", schema="core")
        fk_names = {fk.get("name") for fk in fks}
        if "fk_breedable_pandas_source_id_panda_profiles" in fk_names:
            op.drop_constraint(
                "fk_breedable_pandas_source_id_panda_profiles",
                "breedable_pandas",
                schema="core",
                type_="foreignkey",
            )

    if has_breedable:
        op.alter_column(
            "breedable_pandas",
            "source_id",
            schema="core",
            existing_type=sa.Text(),
            type_=sa.Integer(),
            existing_nullable=False,
            postgresql_using="source_id::integer",
        )

    op.alter_column(
        "panda_profiles",
        "source_id",
        schema="core",
        existing_type=sa.Text(),
        type_=sa.Integer(),
        existing_nullable=False,
        postgresql_using="source_id::integer",
    )

    if has_breedable:
        op.create_foreign_key(
            "fk_breedable_pandas_source_id_panda_profiles",
            "breedable_pandas",
            "panda_profiles",
            ["source_id"],
            ["source_id"],
            source_schema="core",
            referent_schema="core",
            ondelete="CASCADE",
        )

    _recreate_views_ascending(captured_views)
