# Panda Matching Database

A PostgreSQL + Python system for panda breeding compatibility and conversational match exploration.

The project now includes:

* Core SQL matching pipeline (eligibility → candidate pairs → recommendations)
* Curated panda profile overrides (personality + health expert notes)
* Text-derived feature extraction for explainable scoring
* v2 pair scoring with score breakdowns
* FastAPI endpoints + browser chat UI for agent-style interaction
* Persistent chat memory (`chat_sessions`, `chat_messages`, `chat_state`)
* Optional Databricks LLM conversational mode (planner/response + SQL tool fallback)

---

# Project Overview

This database simulates a panda breeding matching system. The goal is to recommend compatible breeding pairs based on biological constraints, lineage risk, personality compatibility, reproductive history, and data quality.

The system is built as a layered SQL pipeline where each view builds on the previous one.

Pipeline structure:

```
Raw panda data
    ↓
Clean panda profiles
    ↓
Curated profile overrides
    ↓
Agent-ready merged profile view
    ↓
Breeding eligibility
    ↓
Matching features
    ↓
Candidate pairs
    ↓
Scored recommendations
    ↓
Directional recommendations
    ↓
Text feature extraction
    ↓
Explainable v2 pair scoring
    ↓
Ranked matches per panda (v2)
```

---

# Application Structure

The Python application is now split by responsibility instead of keeping API, UI, LLM, and routing logic in one file.

Main modules:

* `src/panda_matching/api/routes.py` – FastAPI app, route definitions, request/response models, server entrypoint
* `src/panda_matching/api/chat_ui.py` – browser chat UI HTML/JS
* `src/panda_matching/agent/llm.py` – Databricks foundation model invocation, planning prompt, answer composition
* `src/panda_matching/agent/tools.py` – SQL/data access helpers, chat memory persistence, structured tool payloads
* `src/panda_matching/agent/router.py` – LLM-first routing plus deterministic fallback routing
* `src/panda_matching/cli.py` – CLI import entrypoint

Operational notes:

* `uv run panda-matching-api` still works through `panda_matching.api:run`
* `POST /agent/chat` is the user-facing conversational endpoint
* raw structured outputs are still available through dedicated REST endpoints such as `GET /matches/top`

---

# Database Schema Overview

Schema used: `core`

Main objects:

| Object                                    | Type  | Purpose                                                    |
| ----------------------------------------- | ----- | ---------------------------------------------------------- |
| panda_profiles                            | Table | Raw + enriched panda profile records                       |
| panda_profile_overrides                   | Table | Curated health/personality overrides for named pandas      |
| chat_sessions                             | Table | One row per conversation session                           |
| chat_messages                             | Table | Durable chat transcript with role/intent/data payload      |
| chat_state                                | Table | Persistent short-term session memory (`last_panda_name`, etc.) |
| panda_profiles_agent                      | View  | Merged agent view (`raw` + `curated` fallback columns)     |
| panda_profiles_clean                      | View  | Cleaned and standardized panda data                        |
| breedeable_pandas                         | View  | Pandas eligible for breeding                               |
| matching_features                         | View  | Matching and data quality features per panda               |
| candidate_pairs                           | View  | All possible panda pairs                                   |
| recommended_matches                       | View  | Compatibility scoring for pairs                            |
| directional_recommended_matches           | View  | Pair → focal/candidate directional format                  |
| ranked_directional_recommended_matches    | View  | Original ranked recommendations                            |
| panda_text_features                       | Table | Text-derived numeric/behavioral scoring features per panda |
| match_scores_v2                           | Table | Explainable pair-level v2 scoring outputs                  |
| ranked_directional_recommended_matches_v2 | View  | Final v2 ranked recommendations with breakdown             |

---

# Table: panda_profiles (Base Table)

This is the main raw dataset containing panda information.

### Key Columns

* source_id – Unique panda ID
* name – Panda name
* chinese_name – Chinese name
* sex – Male/Female
* birth_date – Date of birth
* zoo_or_facility – Current zoo
* city_region – City/Region
* country – Country
* mother – Mother name
* father – Father name
* babies_had_count – Number of babies
* on_loan – Whether panda is on loan
* ownership_category – Ownership type
* status – alive/deceased
* lineage – Genetic lineage group
* personality_tags – Comma-separated personality traits
* breeding_notes – Breeding notes
* health_notes – Health notes

This table contains both structured and semi-structured descriptive data.

---

# View: panda_profiles_clean

This view:

* trims text
* standardizes null values
* converts "Unknown" to NULL
* standardizes casing
* prepares data for matching logic

Purpose:
Provides a clean dataset for all downstream views.

---

# View: breadeable_pandas

Determines which pandas are eligible for breeding.

### Eligibility Rules

* Panda must be alive
* Age ≥ 5 years
* Babies had ≤ 8

Outputs:

* source_id
* name
* sex
* age_years
* babies_had_count
* eligible_rule
* refreshed_at

This is the first filtering layer.

---

# View: matching_features

This view converts messy panda data into structured matching signals.

### Features Included

* age_years
* is_alive
* is_breeding_age
* babies_had_count
* has_known_mother
* has_known_father
* has_known_lineage
* has_personality_data
* has_health_data
* has_breeding_notes
* has_parentage_data
* notes_flag
* data_quality_score
* confidence_level
* match_status

### Data Quality Score

Score is based on:

* known parents
* known lineage
* personality data
* health data
* breeding notes

This score represents how reliable the panda data is for matching.

---

# View: candidate_pairs

This view generates all possible panda pairs from the eligible pandas.

### Pair Features

* panda_1_id
* panda_2_id
* same_sex_flag
* age_gap_years
* same_lineage_flag
* shares_known_parent_flag
* loan_conflict_flag
* pair_eligibility
* pair_risk_reason
* pair_score

This is the pair generation layer where compatibility constraints are evaluated.

---

# View: recommended_matches

This view scores pairs based on compatibility rules.

### Scoring Factors

Score increases if:

* Male panda is older than female
* Personality tags overlap
* Panda has zero babies and is breeding age
* Pandas have high data quality scores

Score decreases if:

* Panda has more than 5 babies
* Loan conflicts exist
* Large age gaps
* Low data quality

This view produces a recommendation score for each pair.

---

# View: directional_recommended_matches

Pairs are converted into a directional format:

Instead of:

```
Panda A + Panda B
```

We create:

```
Panda A → Candidate Panda B
Panda B → Candidate Panda A
```

This allows the system to easily answer:

* “Who are the best matches for Bao Li?”
* “Top matches for Long Long”
* “Rank female matches for Xiao Qi Ji”

---

# View: ranked_directional_recommended_matches_v2

Explainable final recommendation layer.

Adds v2 scoring artifacts:

```
final_score_v2
recommendation_rank_v2
score_breakdown_json
top_positive_factors
top_negative_factors
```

Ranking candidates per panda based on final v2 score.

This is the preferred output for the agent/API.

---

# Example Queries

### Top matches for a panda

```sql
SELECT *
FROM core.ranked_directional_recommended_matches
WHERE focal_panda_name = 'Long Long'
ORDER BY recommendation_rank;
```

### Top matches using explainable v2 score

```sql
SELECT *
FROM core.ranked_directional_recommended_matches_v2
WHERE focal_panda_name = 'Ai Bao'
ORDER BY recommendation_rank_v2
LIMIT 5;
```

### Top 5 matches for every panda

```sql
SELECT *
FROM core.ranked_directional_recommended_matches
WHERE recommendation_rank <= 5
ORDER BY focal_panda_name, recommendation_rank;
```

### Best matches overall

```sql
SELECT *
FROM core.ranked_directional_recommended_matches
ORDER BY recommendation_score DESC
LIMIT 20;
```

### Pandas with no recommendations

```sql
SELECT bp.*
FROM core.breadeable_pandas bp
LEFT JOIN (
    SELECT DISTINCT focal_panda_id AS source_id
    FROM core.ranked_directional_recommended_matches
) rm
ON bp.source_id = rm.source_id
WHERE rm.source_id IS NULL;
```

---

# System Architecture Summary

The database follows a layered recommendation system architecture:

```
Raw Data Layer
    panda_profiles

Data Cleaning Layer
    panda_profiles_clean

Eligibility Layer
    breadeable_pandas

Feature Engineering Layer
    matching_features

Pair Generation Layer
    candidate_pairs

Scoring Layer
    recommended_matches

Recommendation Layer
    directional_recommended_matches

Ranking Layer
    ranked_directional_recommended_matches
```

This structure allows the system to be:

* Explainable
* Extendable
* Queryable by an AI agent
* Suitable for dashboards
* Suitable for matching optimization

---

# Refresh Pipeline Job

The project includes an end-to-end refresh job script that updates source data and recomputes core derived fields.

Script:

```
scripts/refresh_pipeline.sh
```

Steps executed:

1. Apply latest migrations (`alembic upgrade head`)
2. Sync latest pandas from source (`scripts/import_blackandwhitebear.py`)
3. Sync descriptions, life journey, twin/personality/breeding/health enrichment (`scripts/sync_panda_descriptions.py`)
4. Load curated profile overrides (`scripts/load_curated_profiles.py`)
5. Recompute lineage groups (`scripts/recompute_lineage.py`)
6. Extract text-derived feature vectors (`scripts/extract_text_features.py`)
7. Compute explainable v2 pair scores (`scripts/compute_match_scores_v2.py`)
8. Validate counts for:
   - `core.panda_profiles`
   - `core.breedeable_pandas`
   - `core.ranked_directional_recommended_matches`
   - `core.panda_profile_overrides`
   - `core.panda_text_features`
   - `core.match_scores_v2`

Run manually:

```bash
source .venv/bin/activate
set -a; source .env; set +a
./scripts/refresh_pipeline.sh
```

Scheduling options:

* local `cron`
* GitHub Actions (scheduled workflow)
* Databricks Job (if you orchestrate from Databricks)

---

# Possible Future Improvements

* Expand curated coverage beyond current famous-panda subset
* Add previous-interaction signal from historical pairing outcomes
* Add API auth and role-based access controls
* Add regression evaluation suite for v1 vs v2 score stability
* Add dashboard for v2 score component drift over time

---

# API and Chat

Run API:

```bash
uv run panda-matching-api
```

Useful URLs:

* `http://localhost:8000/health`
* `http://localhost:8000/docs`
* `http://localhost:8000/chat` (browser chat UI)

Main endpoints:

* `GET /matches/top?panda_name=<name>&k=<n>`
* `GET /matches/explain?focal_id=<id>&candidate_id=<id>`
* `GET /matches/blockers?focal_id=<id>`
* `POST /agent/chat`

## Conversational Mode (Databricks LLM)

`POST /agent/chat` supports two modes:

1. **LLM mode (preferred):** Databricks LLM plans tool calls and generates conversational responses.
2. **Fallback mode:** existing deterministic regex/router logic executes if LLM is disabled or unavailable.

This keeps factual answers grounded in your SQL pipeline while improving conversational quality.

In practice, the chatbot now uses a **split-response architecture**:

* **Deterministic rendering first** for simple and high-risk factual prompts:
  * profile questions (`who is`, age, location, health, personality, fun facts)
  * counts (`how many male pandas`, `how many eligible pandas`, total pandas, etc.)
  * unsupported requests (`write me a poem`, weather, off-topic prompts)
  * structured ranking/comparison prompts where the SQL output is already sufficient
* **LLM-assisted responses** only where free-form reasoning is actually useful:
  * richer top-match summaries
  * pairwise match reasoning
  * ranking explanations
  * ambiguous or multi-step conversational prompts

This design avoids hallucinations on straightforward factual answers while still letting the LLM improve the parts of the product that actually benefit from natural-language reasoning.

The browser chat UI is now user-facing:

* ranked matches are returned as readable summaries
* profile and blocker answers are summarized in plain language
* raw tool payloads are not shown in normal chat responses

If you need raw structured output for debugging or integration, use the dedicated REST endpoints such as `GET /matches/top`, `GET /matches/explain`, and `GET /matches/blockers`.

### Enable LLM mode

Add these env vars to `.env` (or export in shell):

```bash
DATABRICKS_LLM_ENABLED=true
DATABRICKS_HOST=https://dbc-831610b8-d4ee.cloud.databricks.com
DATABRICKS_TOKEN=<your_token>
DATABRICKS_LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
```

Then restart the API process.

### Recommended endpoint

This project currently uses a Databricks-hosted foundation model endpoint rather than a custom endpoint you deploy yourself.

Recommended value:

* `DATABRICKS_LLM_ENDPOINT=databricks-meta-llama-3-3-70b-instruct`

Notes:

* `databricks-gpt-5-4-mini` may exist in the workspace but can still be unavailable to your user due to Databricks-side rate limiting.
* If the configured endpoint cannot be invoked, the app logs the LLM failure and falls back to deterministic chat behavior.

### Notes

* LLM mode is optional. If `DATABRICKS_LLM_ENABLED=false`, chat remains fully rule-based.
* Tool execution remains deterministic (`top_matches`, `blockers`, `explain_match`, etc.).
* Chat memory is persisted in DB, so sessions survive restarts and multi-worker deployments.
* If you are testing LLM mode, successful responses usually use intents prefixed with `llm_` such as `llm_top_matches`.
* If Databricks returns `429 Too Many Requests`, the app logs the failure and falls back to deterministic routing and rendering instead of returning an empty answer.

---

# Evaluation Starter

The repo now includes both a **free full regression suite** and a **small paid LLM quality benchmark**.

Files:

* `data/evaluation/panda_chatbot_eval_v1.json` – full hybrid evaluation set for broad regression coverage
* `data/evaluation/panda_chatbot_llm_benchmark_v1.json` – curated LLM quality benchmark set
* `scripts/run_offline_regression_suite.py` – free programmatic regression suite
* `scripts/create_mlflow_eval_dataset.py` – creates or updates an MLflow evaluation dataset using `mlflow.genai.datasets.create_dataset()`
* `scripts/register_mlflow_scorers.py` – registers MLflow scorers
* `scripts/run_baseline_eval.py` – runs MLflow `mlflow.genai.evaluate()` against a named dataset

Recommended workflow:

1. Install dev dependencies including MLflow:

```bash
uv sync --extra dev
```

2. Preview the starter dataset without calling MLflow:

```bash
uv run python scripts/create_mlflow_eval_dataset.py --dry-run
```

3. Set your MLflow environment:

```bash
export MLFLOW_TRACKING_URI=<your_tracking_uri>
export MLFLOW_EXPERIMENT_ID=<your_experiment_id>
```

4. Create or update the evaluation dataset in MLflow:

```bash
uv run python scripts/create_mlflow_eval_dataset.py
```

### Free full regression suite

Use the offline suite for broad, cheap, repeatable regression checks:

```bash
uv run python scripts/run_offline_regression_suite.py --local-only
```

This suite avoids paid model calls and checks things like:

* no exceptions
* no raw JSON
* no debug artifacts
* non-empty answers
* expected counts where labeled
* expected names in ranking answers
* pairwise winner matching
* approximate age wording

### Paid LLM quality benchmark

Use the curated MLflow benchmark only for prompts where the LLM should add value:

```bash
set -a; source .env; set +a
export OPENAI_API_KEY=<your_key>
export MLFLOW_TRACKING_URI=http://127.0.0.1:5051
export MLFLOW_EXPERIMENT_ID=1
export DATABRICKS_LLM_ENABLED=true

uv run python scripts/run_baseline_eval.py \
  --dataset-name panda_chatbot_llm_benchmark_v1 \
  --max-examples 3 \
  --run-name llm-benchmark-smoke \
  --exclude-scorer panda_tool_call_correctness
```

The benchmark should be run in small batches if Databricks serving is rate-limited.

The full regression dataset is intentionally hybrid:

* some cases contain hard factual expectations such as count answers
* others focus on behaviors such as grounding, concise ranked summaries, no raw JSON, and safe fallback behavior

Use these datasets as the initial benchmark, then add:

* production failure cases from real traces
* follow-up memory cases
* unsupported or ambiguous requests
* pairwise breeding-judgment questions you care about most

### Current Evaluation Status

Current repo status:

* the free offline regression suite passes end to end
* the curated paid LLM benchmark has been run successfully in batches
* Databricks `429 Too Many Requests` made smaller benchmark batches necessary
* deterministic routing now handles the core prompt classes that were previously unstable:
  * profile questions
  * count questions
  * ranked top-match questions
  * ranking explanations such as `why is X first/second`
  * pairwise comparison and pair-opinion prompts

Practical takeaway:

* use the free local regression suite as the broad guardrail
* use the curated paid benchmark in small batches for quality checks on reasoning-heavy prompts

### One-command runs

There is now a small `Makefile` wrapper for the two main evaluation paths.

Free regression suite:

```bash
make eval-free
```

Paid MLflow benchmark:

```bash
DATASET=panda_chatbot_llm_benchmark_remaining_batch2 \
RUN_NAME=llm-benchmark-batch2 \
make eval-paid
```

Notes:

* `make eval-free` does not use paid judge calls
* `make eval-paid` expects your environment to already include:
  * `OPENAI_API_KEY`
  * `MLFLOW_TRACKING_URI`
  * `MLFLOW_EXPERIMENT_ID`
  * `DATABRICKS_LLM_ENABLED=true`
* if `DATASET` is not provided, `make eval-paid` defaults to `panda_chatbot_llm_benchmark_v1`

---

# Final Note

This project is not just a database.
It is a **rule-based recommendation system implemented in SQL** for panda breeding compatibility.

The database transforms raw panda information into ranked partner recommendations using eligibility rules, compatibility constraints, behavioral data, and data quality scoring.

---

If this is for GitHub, name the file:

```
README.md
```
