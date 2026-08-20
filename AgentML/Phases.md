# Phases.md — AgentML

Build order. Each phase should be independently testable before moving to the next — don't start Phase N+1 with Phase N half-working.

## Phase 0 — Foundation (Day 1-3)
- Set up repo structure per Architecture.md
- Define `AgentMLState` schema in `src/orchestrator/state.py`
- Set up LangGraph skeleton with stub agents (each just passes state through unchanged)
- Groq client abstraction (`src/llm/groq_client.py`) with a basic "hello world" call working
- `.env.example`, `requirements.txt`
- **Done when:** a CSV can be uploaded and flow through all 7 stub agents end-to-end without errors, producing an empty/placeholder report.

## Phase 1 — Dataset Profiling Agent (Day 3-4)
- Load CSV, compute shape, dtypes, missing %, duplicate count, per-column stats
- Populate `dataset_profile` in state
- Unit tests: clean dataset, dataset with missing values, dataset with all-duplicate rows
- **Done when:** profiling report is accurate and logged for at least 3 varied sample CSVs.

## Phase 2 — Data Preprocessing Agent (Day 4-6)
- Missing value imputation, dedup, categorical encoding, numeric scaling
- `preprocessing_log` populated with every transformation applied and why
- Unit tests: dataset with unencodable free-text column, all-numeric dataset, mixed dataset
- **Done when:** clean dataset is produced and log is human-readable/explainable.

## Phase 3 — Problem Detection Agent (Day 6-7)
- Target column heuristic detection (last column / name match / user-specified)
- Task type classification logic (see Architecture.md detection rules)
- Confidence scoring + reasoning string
- Human checkpoint wiring (LangGraph `interrupt()`) for low-confidence cases
- **Done when:** correctly detects task type on at least 5 varied sample datasets (clear classification, clear regression, clear clustering, ambiguous case, no-target case), and the checkpoint actually pauses/resumes correctly.

## Phase 4 — Model Zoo + Experiment Orchestrator (Day 7-11)
- Build fixed model zoo per task type (classification, regression, clustering — see Architecture.md)
- Implement ThreadPoolExecutor-based concurrent experiment runs
- Log every experiment run (model, params, metrics, runtime, success/failure) to state
- Optional human checkpoint for scope (experiment count / time cap)
- **Done when:** runs the full model zoo against a sample dataset for each task type and logs complete, correct results — including at least one deliberately-broken run to confirm failure isolation works.

## Phase 5 — Model Evaluation Agent (Day 11-12)
- Task-appropriate metric selection (see Architecture.md metric table)
- Ranking logic, best-model selection
- Human checkpoint for override
- **Done when:** produces a correct ranked leaderboard for each task type, and override checkpoint correctly updates `best_model_id`.

## Phase 6 — Report Generation Agent (Day 12-14)
- Jinja2 template for structured report sections (dataset summary, preprocessing log, detected task + reasoning, experiment leaderboard, evaluation summary, final model)
- Groq LLM narration layer on top of the structured data (not replacing it)
- Markdown output; optional PDF conversion
- **Done when:** a full end-to-end run (Phase 0-6 chained) produces a complete, readable, accurate report for a real CSV.

## Phase 7 — Integration, Edge Cases, Polish (Day 14-18)
- Full end-to-end testing across classification/regression/clustering datasets
- Edge cases: empty CSV, single-column CSV, all-missing column, huge cardinality categorical, no numeric columns at all
- Streamlit UI wiring (upload → progress → checkpoint prompts → final report display/download)
- **Done when:** system handles at least 3 real-world messy CSVs (not toy datasets) without crashing, and checkpoints are usable through the UI, not just the CLI.

## Phase 8 — Buffer / Hardening (Day 18-20)
- Reserved for whatever broke during Phase 7 (something always does)
- Performance check on a larger CSV (e.g., 50k+ rows) to make sure nothing chokes
- Final documentation pass, README, demo script

---
**Total: ~20 days**, assuming a small team working part-time. Do not start Phase N+1 agent code until Phase N's tests pass — this project's biggest risk is agent boundary rework mid-build (the exact issue caught and fixed in the CRCM project's architecture doc).
