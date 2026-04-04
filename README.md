# Panda Matching Database

A PostgreSQL database designed to support a panda breeding compatibility and matching recommendation system. The system cleans panda data, determines breeding eligibility, generates candidate pairs, scores compatibility, and produces ranked match recommendations.

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
Ranked matches per panda
```

---

# Database Schema Overview

Schema used: `core`

Main objects:

| Object                                 | Type  | Purpose                                                  |
| -------------------------------------- | ----- | -------------------------------------------------------- |
| panda_profiles                         | Table | Raw panda data                                           |
| panda_profiles_clean                   | View  | Cleaned and standardized panda data                      |
| breadeable_pandas                      | View  | Pandas eligible for breeding                             |
| matching_features                      | View  | Matching and data quality features per panda             |
| candidate_pairs                        | View  | All possible panda pairs                                 |
| recommended_matches                    | View  | Compatibility scoring for pairs                          |
| directional_recommended_matches        | View  | Converts pairs into focal panda → candidate panda format |
| ranked_directional_recommended_matches | View  | Ranked match recommendations per panda                   |

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

# View: ranked_directional_recommended_matches

Final recommendation layer.

Adds:

```
recommendation_rank
```

Ranking candidates per panda based on recommendation score.

This is the main output table for the matching system.

---

# Example Queries

### Top matches for a panda

```sql
SELECT *
FROM core.ranked_directional_recommended_matches
WHERE focal_panda_name = 'Long Long'
ORDER BY recommendation_rank;
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
4. Recompute lineage groups (`scripts/recompute_lineage.py`)
5. Validate counts for:
   - `core.panda_profiles`
   - `core.breedeable_pandas`
   - `core.ranked_directional_recommended_matches`

Run manually:

```bash
source .venv/bin/activate
export DATABASE_URL=postgresql+psycopg://panda:panda@localhost:5432/panda_matching
./scripts/refresh_pipeline.sh
```

Scheduling options:

* local `cron`
* GitHub Actions (scheduled workflow)
* Databricks Job (if you orchestrate from Databricks)

---

# Possible Future Improvements

* Genetic relatedness scoring
* Location compatibility scoring
* Health compatibility scoring
* Breeding success probability model
* Machine learning pair scoring
* Dashboard visualization
* API for match queries
* Agent interface for natural language queries

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
