from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from typing import Any

import psycopg

UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "null"}


def _base_name(name: str | None) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", (name or "")).strip()


def _valid_name(name: str | None) -> str | None:
    normalized = _base_name(name)
    if normalized.lower() in UNKNOWN_VALUES:
        return None
    return normalized


def compute_lineage_updates(rows: list[dict[str, Any]]) -> list[tuple[int, str]]:
    panda_node_to_id: dict[str, str] = {}
    graph: dict[str, set[str]] = defaultdict(set)

    def connect(a: str, b: str) -> None:
        graph[a].add(b)
        graph[b].add(a)

    for row in rows:
        source_id_raw = row.get("source_id")
        if source_id_raw is None:
            continue
        source_id = str(source_id_raw)
        panda_name = _valid_name(str(row.get("name"))) or f"PANDA_{source_id}"

        panda_node = f"panda:{source_id}"
        panda_node_to_id[panda_node] = source_id

        connect(panda_node, f"person:{panda_name.lower()}")

        mother_raw = row.get("mother")
        father_raw = row.get("father")
        mother = _valid_name(mother_raw if isinstance(mother_raw, str) else None)
        father = _valid_name(father_raw if isinstance(father_raw, str) else None)
        if mother:
            connect(panda_node, f"person:{mother.lower()}")
        if father:
            connect(panda_node, f"person:{father.lower()}")

    visited: set[str] = set()
    components: list[list[str]] = []

    for node in panda_node_to_id:
        if node in visited:
            continue

        queue: deque[str] = deque([node])
        visited.add(node)
        panda_ids: list[str] = []

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

    updates: list[tuple[int, int]] = []
    lineage_num = 1
    for component in components:
        for source_id in component:
            updates.append((lineage_num, source_id))
        lineage_num += 1

    return updates


def main() -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")

    db_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(
                """
                SELECT source_id, name, mother, father
                FROM core.panda_profiles
                ORDER BY source_id
                """
            )
            rows = cur.fetchall()

            updates = compute_lineage_updates(rows)

            cur.execute("UPDATE core.panda_profiles SET lineage = NULL")
            cur.executemany(
                """
                UPDATE core.panda_profiles
                SET lineage = %s
                WHERE source_id = %s
                """,
                updates,
            )

        conn.commit()

    print(f"Recomputed lineage for {len(updates)} pandas")


if __name__ == "__main__":
    main()
