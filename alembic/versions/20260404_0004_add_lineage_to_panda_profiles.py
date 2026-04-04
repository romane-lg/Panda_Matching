"""add lineage column to panda_profiles and backfill family groups

Revision ID: 20260404_0004
Revises: 20260404_0003
Create Date: 2026-04-04 16:25:00
"""
from __future__ import annotations

import re
from collections import defaultdict, deque

import sqlalchemy as sa

from alembic import op

revision = "20260404_0004"
down_revision = "20260404_0003"
branch_labels = None
depends_on = None


UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "null"}


def _base_name(name: str | None) -> str:
    cleaned = re.sub(r"\s*\(.*?\)\s*$", "", (name or "")).strip()
    return cleaned


def _valid_name(name: str | None) -> str | None:
    normalized = _base_name(name)
    if normalized.lower() in UNKNOWN_VALUES:
        return None
    return normalized


def upgrade() -> None:
    op.add_column(
        "panda_profiles",
        sa.Column("lineage", sa.Integer(), nullable=True),
        schema="core",
    )

    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """
            SELECT source_id, name, mother, father
            FROM core.panda_profiles
            ORDER BY source_id
            """
        )
    ).mappings().all()

    panda_node_to_id: dict[str, int] = {}
    graph: dict[str, set[str]] = defaultdict(set)

    def connect(a: str, b: str) -> None:
        graph[a].add(b)
        graph[b].add(a)

    for row in rows:
        source_id = int(row["source_id"])
        panda_name = _valid_name(row["name"]) or f"PANDA_{source_id}"
        panda_node = f"panda:{source_id}"
        panda_node_to_id[panda_node] = source_id

        # Connect each panda to a normalized name node.
        connect(panda_node, f"person:{panda_name.lower()}")

        mother = _valid_name(row.get("mother"))
        father = _valid_name(row.get("father"))
        if mother:
            connect(panda_node, f"person:{mother.lower()}")
        if father:
            connect(panda_node, f"person:{father.lower()}")

    visited: set[str] = set()
    components: list[list[int]] = []

    for node in panda_node_to_id:
        if node in visited:
            continue
        queue: deque[str] = deque([node])
        visited.add(node)
        panda_ids: list[int] = []

        while queue:
            cur = queue.popleft()
            if cur.startswith("panda:"):
                panda_ids.append(panda_node_to_id[cur])
            for nxt in graph.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)

        if panda_ids:
            components.append(sorted(set(panda_ids)))

    components.sort(key=lambda ids: (ids[0], len(ids)))

    lineage_updates: list[dict[str, int]] = []
    lineage_num = 1
    for group in components:
        for source_id in group:
            lineage_updates.append({"source_id": source_id, "lineage": lineage_num})
        lineage_num += 1

    if lineage_updates:
        bind.execute(
            sa.text(
                """
                UPDATE core.panda_profiles
                SET lineage = :lineage
                WHERE source_id = :source_id
                """
            ),
            lineage_updates,
        )


def downgrade() -> None:
    op.drop_column("panda_profiles", "lineage", schema="core")
