# Panda Matching

Panda Matching is a PostgreSQL, FastAPI, and React application for exploring panda breeding compatibility. It combines a SQL matching pipeline, curated panda profile enrichment, explainable pair scoring, a REST API, and a browser chatbot.

## What Is Included

- PostgreSQL schema managed by Alembic.
- Import and refresh scripts for panda profile data from Black and White Bear.
- Layered SQL views for profile cleanup, eligibility, candidate generation, and ranked recommendations.
- Curated profile overrides for agent-facing personality and health notes.
- Text-derived feature extraction and explainable v2 match scoring.
- FastAPI endpoints for matches, explanations, blockers, pandas, and chat.
- Persistent chat memory in PostgreSQL.
- Optional Databricks foundation model mode with deterministic fallback behavior.
- React + Vite frontend with chat, match cards, panda photos, and catalog browsing.
- Offline regression tests and optional MLflow-based LLM evaluation.

## Repository Layout

```text
alembic/                 Database migrations
data/                    Evaluation data and project data files
docs/                    Additional project notes
frontend/                React + TypeScript + Vite UI
scripts/                 Import, refresh, scoring, and evaluation scripts
src/panda_matching/      Python package
tests/                   Pytest test suite
docker-compose.yml       Local PostgreSQL service
pyproject.toml           Python package and tool configuration
```

Important Python modules:

- `src/panda_matching/api/routes.py`: FastAPI app and REST routes.
- `src/panda_matching/api/chat_ui.py`: simple server-rendered chat page at `/chat`.
- `src/panda_matching/agent/router.py`: deterministic routing plus optional LLM routing.
- `src/panda_matching/agent/llm.py`: Databricks model serving calls.
- `src/panda_matching/agent/tools.py`: SQL-backed tool functions and chat memory.
- `src/panda_matching/ingest/`: import helpers for source data.
- `src/panda_matching/observability/`: MLflow tracing setup.

## Requirements

- Python 3.10+
- `uv`
- Docker, for local PostgreSQL
- Node.js and npm, for the React frontend
- `psql`, if you want to run `scripts/refresh_pipeline.sh`

The default local database URL is:

```text
postgresql+psycopg://panda:panda@localhost:5432/panda_matching
```

## Setup

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Install Python dependencies:

```bash
uv sync --extra dev
```

Set the database URL if you are not using the default:

```bash
export DATABASE_URL=postgresql+psycopg://panda:panda@localhost:5432/panda_matching
```

Apply migrations:

```bash
uv run alembic upgrade head
```

Run the full data refresh pipeline:

```bash
set -a; source .env; set +a
uv run ./scripts/refresh_pipeline.sh
```

The refresh script applies migrations, imports panda profiles, syncs descriptions, loads curated overrides, recomputes lineage, extracts text features, computes v2 scores, and prints validation counts.

## Run The API

```bash
uv run panda-matching-api
```

Useful URLs:

- `http://localhost:8000/health`
- `http://localhost:8000/docs`
- `http://localhost:8000/chat`

Current API endpoints:

- `GET /health`
- `GET /chat`
- `GET /pandas`
- `GET /matches/top?panda_name=<name>&k=<n>`
- `GET /matches/explain?focal_id=<id>&candidate_id=<id>`
- `GET /matches/blockers?focal_id=<id>`
- `POST /agent/chat`

## Run The React Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/agent`, `/matches`, `/pandas`, and `/health` to `http://localhost:8000`, so run the FastAPI service first.

Build the frontend:

```bash
cd frontend
npm run build
```

## Database Objects

The project uses the `core` schema for application tables and views. Main objects include:

| Object | Type | Purpose |
| --- | --- | --- |
| `panda_profiles` | Table | Raw and enriched panda profile records, including photo metadata. |
| `panda_profile_overrides` | Table | Curated agent-facing profile summaries, health notes, and personality tags. |
| `panda_profiles_agent` | View | Merged raw + curated profile view used by the API and agent tools. |
| `breedeable_pandas` | View | Current eligibility view used by the matching pipeline. The misspelling is legacy and still intentional in the database. |
| `matching_features` | View | Profile-level matching and data quality features. |
| `candidate_pairs` | View | Pair generation and compatibility blockers. |
| `recommended_matches` | View | Original pair scoring layer. |
| `directional_recommended_matches` | View | Pair results converted into focal/candidate rows. |
| `ranked_directional_recommended_matches` | View | Original ranked recommendation output. |
| `panda_text_features` | Table | Text-derived behavioral and health scoring features. |
| `match_scores_v2` | Table | Explainable pair-level v2 scoring components. |
| `ranked_directional_recommended_matches_v2` | View | Preferred ranked recommendation output with score breakdowns. |
| `chat_sessions` | Table | One row per chat session. |
| `chat_messages` | Table | Durable chat transcript. |
| `chat_state` | Table | Session memory such as the last referenced panda. |

## Matching Pipeline

```text
panda_profiles
  -> panda_profiles_clean
  -> panda_profiles_agent
  -> breedeable_pandas
  -> matching_features
  -> candidate_pairs
  -> recommended_matches
  -> directional_recommended_matches
  -> ranked_directional_recommended_matches
  -> panda_text_features + match_scores_v2
  -> ranked_directional_recommended_matches_v2
```

Use the v2 ranked view for new product work when it exists. The API falls back to the original ranked view if the v2 view is unavailable.

Example query:

```sql
SELECT *
FROM core.ranked_directional_recommended_matches_v2
WHERE lower(focal_panda_name) = lower('Ai Bao')
ORDER BY recommendation_rank_v2
LIMIT 5;
```

## Chat And LLM Mode

`POST /agent/chat` always uses deterministic SQL-backed tools for data access. Optional Databricks LLM mode can plan or compose richer responses, but the router falls back to deterministic behavior if LLM mode is disabled, unavailable, or rate-limited.

Enable LLM mode with:

```bash
export DATABRICKS_LLM_ENABLED=true
export DATABRICKS_HOST=https://<your-workspace-host>
export DATABRICKS_TOKEN=<your-token>
export DATABRICKS_LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
```

Optional retry controls:

```bash
export DATABRICKS_LLM_MAX_RETRIES=2
export DATABRICKS_LLM_RETRY_BACKOFF_SECONDS=1.5
export DATABRICKS_LLM_COOLDOWN_SECONDS=30
```

MLflow tracing can be configured with:

```bash
export MLFLOW_TRACKING_URI=<tracking-uri>
export MLFLOW_EXPERIMENT_ID=<experiment-id>
```

## CLI

The package exposes a small import CLI:

```bash
uv run panda-matching import --source <source-name> --file <path-to-json-or-jsonl>
```

For the Black and White Bear source and downstream scoring, use the scripts in `scripts/` or the full refresh pipeline.

## Testing And Evaluation

Run the unit tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Run the free offline regression suite:

```bash
make eval-free
```

Run the optional paid LLM benchmark:

```bash
DATASET=panda_chatbot_llm_benchmark_v1 RUN_NAME=llm-benchmark-smoke make eval-paid
```

The paid benchmark requires `OPENAI_API_KEY`, `MLFLOW_TRACKING_URI`, `MLFLOW_EXPERIMENT_ID`, and LLM mode environment variables. See `docs/evaluation.md` for more detail.

## Notes

- The local `mlflow.db`, `mlartifacts/`, `tmp/`, and Python/Node build artifacts are development outputs, not source code.
- This project is a prototype decision-support system. Recommendations should be treated as explainable software outputs, not biological or veterinary authority.
