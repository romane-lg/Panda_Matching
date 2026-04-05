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

echo "[1/8] Applying migrations"
alembic upgrade head

echo "[2/8] Syncing panda profiles from source"
python scripts/import_blackandwhitebear.py

echo "[3/8] Syncing descriptions and enrichment"
python scripts/sync_panda_descriptions.py

echo "[4/8] Loading curated personality and health overrides"
python scripts/load_curated_profiles.py

echo "[5/8] Recomputing lineage groups"
python scripts/recompute_lineage.py

echo "[6/8] Extracting text-derived scoring features"
python scripts/extract_text_features.py

echo "[7/8] Computing explainable v2 match scores"
python scripts/compute_match_scores_v2.py

echo "[8/8] Validation counts"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS panda_profiles FROM core.panda_profiles;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS breedeable_pandas FROM core.breedeable_pandas;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS ranked_matches FROM core.ranked_directional_recommended_matches;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS curated_overrides FROM core.panda_profile_overrides;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS text_features FROM core.panda_text_features;"
psql "${PSQL_URL}" -c "SELECT COUNT(*) AS match_scores_v2 FROM core.match_scores_v2;"

echo "Refresh pipeline completed"
