#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "ERROR: DATABASE_URL is not set"
  exit 1
fi

if [[ "${DATABASE_URL}" == postgresql+psycopg://* ]]; then
  PSQL_URL="postgresql://${DATABASE_URL#postgresql+psycopg://}"
else
  PSQL_URL="${DATABASE_URL}"
fi

echo "[1/5] Applying migrations"
alembic upgrade head

echo "[2/5] Syncing panda profiles from source"
python scripts/import_blackandwhitebear.py

echo "[3/5] Syncing descriptions and enrichment"
python scripts/sync_panda_descriptions.py

echo "[4/5] Recomputing lineage groups"
python scripts/recompute_lineage.py

echo "[5/5] Validation counts"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS panda_profiles FROM core.panda_profiles;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS breedeable_pandas FROM core.breedeable_pandas;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS ranked_matches FROM core.ranked_directional_recommended_matches;"

echo "Refresh pipeline completed"
