# AgentML — Session Memory Log

> This file is updated at the end of every session.
> Future sessions read this first to avoid re-reading the entire codebase.

---

## Current Phase

**Phase 4 — Model Zoo + Experiment Orchestrator** ✅ COMPLETE

---

## Last Updated

2026-08-20T19:35:00+05:30

---

## What's Done

### Phase 0 — Foundation (✅ Complete — but see "Known Issues")
- Set up repo structure.
- Defined `AgentMLState` schema.
- Added LangGraph skeleton with stub agents.
- Groq client abstraction.
- `.env.example`, `requirements.txt`.
- Verification of end-to-end stub flow.

### Phase 1 — Dataset Profiling Agent (✅ Complete)
- Load CSV with pandas, compute shape, dtypes, missing %, duplicate count, per-column stats.
- Populate `dataset_profile` in state.
- Unit tests covering: clean dataset, missing values, all-duplicate rows, corrupt CSV (hard-fail).
- 4/4 tests pass.

### Phase 2 — Data Preprocessing Agent (✅ Complete)
- Built preprocessing logic inside `src/agents/preprocessing_agent.py` using scikit-learn (`ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`, `StandardScaler`).
- Deduplication, dropping unencodable free-text (high cardinality) features, imputation (median for numeric, mode for categorical), categorical one-hot encoding, and numerical standard scaling.
- Populaled `preprocessing_log` in state detailing all transformations, parameters, and reasons in a human-readable format.
- Unit tests in `tests/test_preprocessing_agent.py` covering unencodable columns, all-numeric data, and mixed datasets.
- 3/3 tests pass.
- Clean dataset generated and saved to `data/clean_dataset_[session_id].csv`.

### Phase 3 — Problem Detection Agent (✅ Complete — verified)
- Implemented `src/agents/problem_detection_agent.py` (already existed before this session; verified here):
  - Target column heuristic detection (last column / name match / user-specified).
  - Rule-based task-type classification (classification / regression / clustering).
  - Confidence scoring + Groq-narrated reasoning.
  - Human-in-the-loop `interrupt()` for low-confidence cases (< 0.7).
- Unit tests in `tests/test_problem_detection_agent.py`: clear classification, clear regression, clustering (no target), ambiguous numeric with interrupt + resume.
- 4/4 tests pass.

### Phase 4 — Model Zoo + Experiment Orchestrator (✅ Complete — all "done when" criteria met)
- Built fixed model zoos in `src/model_zoo/` (one file per task type, scikit-learn + xgboost only, fixed grids, no Optuna/Hyperopt):
  - `classification_models.py` — DummyClassifier (baseline), LogisticRegression, KNeighborsClassifier, RandomForestClassifier, GradientBoostingClassifier, XGBClassifier (6 models).
  - `regression_models.py` — DummyRegressor (baseline), LinearRegression, Ridge, KNeighborsRegressor, RandomForestRegressor, GradientBoostingRegressor, XGBRegressor (7 models).
  - `clustering_models.py` — KMeans (k=3, k=5), GaussianMixture (k=3), AgglomerativeClustering (k=3, ward), DBSCAN (fixed eps/min_samples).
  - `__init__.py` defines a uniform `ModelSpec` TypedDict so the orchestrator can iterate without knowing which zoo it's calling.
- Implemented `src/agents/experiment_orchestrator_agent.py`:
  - Dispatches to the correct zoo based on `state["task_type"]`.
  - Runs all experiments concurrently via `concurrent.futures.ThreadPoolExecutor` (NOT asyncio — Rules.md §1).
  - Per Rules.md §4: a single experiment throwing an exception is **logged and skipped** (recoverable error appended to state); the rest continue.
  - **If ALL experiments fail** → hard fail with non-recoverable `all_experiments_failed` error including a diagnostic summary naming every failed model and its error type.
  - Logs every run to `state["experiment_results"]` with model name, family, params, success, error_type/error_message, metrics, runtime_seconds, n_train_samples, n_test_samples.
  - Task-appropriate metric selection (classification: accuracy/f1/precision/recall; regression: rmse/mae/r2; clustering: silhouette/n_clusters/n_noise).
  - Architecture.md §6 checkpoint 2 (scope) is offered via `interrupt()` — always offered, not forced, accepts override of `max_experiments`, `max_workers`, `time_cap_seconds`. Falls back to defaults if invoked outside a LangGraph runtime.
- 8/8 orchestrator tests pass (`tests/test_experiment_orchestrator.py`), including:
  - Full-zoo happy path for classification, regression, clustering.
  - **Failure isolation test** — injected `_BrokenClassifier` (always raises on `fit()`) into the classification zoo. Confirmed the broken entry is logged with `success=False`, `error_type="RuntimeError"`, `error_message="deliberately broken estimator..."`, `runtime_seconds=0.0811`, and that all OTHER models still ran and succeeded. Orchestrator's overall status remained `"completed"` (not `"failed"`).
  - **All-fail hard-fail test** — entire zoo replaced with broken estimators → status `"failed"`, `all_experiments_failed` error contains diagnostic summary listing every model name and error.
  - **Scope checkpoint test** — invoked inside compiled graph, `interrupt()` fires, resume with `max_experiments=2` causes exactly 2 experiments to run.
  - Direct `_run_single_experiment()` unit tests for both happy and failure paths.

**Test results (excluding pre-existing broken Phase 0 tests):** `19 passed` in ~10s
- 4 Phase 1 profiling agent tests
- 3 Phase 2 preprocessing agent tests
- 4 Phase 3 problem detection agent tests
- 8 Phase 4 experiment orchestrator tests

---

## What's In Progress

Nothing — waiting for user confirmation to start Phase 5 (Model Evaluation Agent).

---

## Known Issues / Blockers

### Pre-existing Phase 0 test failures (NOT introduced by Phase 4)
- 5 tests in `tests/test_phase0.py` fail with: `"Checkpointer requires one or more of the following 'configurable' keys: thread_id, checkpoint_ns, checkpoint_id"`.
- Root cause: those tests call `graph_module.build_graph().invoke(initial)` directly without passing `config={"configurable": {"thread_id": ...}}`. The MemorySaver checkpointer requires a thread_id.
- The non-test `run_pipeline()` helper in `graph.py` already passes the correct config — only the Phase 0 tests' direct `g.invoke(initial)` calls are missing it.
- Fix is mechanical (~5 line edit to the failing tests or to `_wrap()` in `graph.py` to auto-thread-id); awaiting confirmation to address.

### Phase 4 minor observation
- DBSCAN with eps=0.5 / min_samples=5 on small toy datasets frequently produces "no clusters" (all points noise), which is itself an informative data-quality signal. Logged but not flagged as a failure. If this becomes annoying in reports, swap to a data-driven eps heuristic in Phase 6 (post-Phase-5 confirmation).

---

## Decisions Made Since the Design Docs

| # | Decision | Reason |
|---|----------|--------|
| 1 | Align folder structure to `Architecture.md` §5 | Renamed agent files from Phase 0 stubs to match Architecture.md exactly (e.g. `profiling_agent.py`, `preprocessing_agent.py`, etc.). |
| 2 | Align state schema to `Architecture.md` §3 | Replaced Phase 0 custom schema fields with the exact 14 keys specified in Architecture.md. |
| 3 | Use standard Python types for statistics | All stats in `dataset_profile` are sanitized to standard float/int/str/None to ensure JSON serializability for the Streamlit UI and LLM calls. |
| 4 | Throw native exceptions for hard fails | In `profiling_agent.py`, failures to read or parse the CSV raise ValueError/ParserError directly to abort the pipeline immediately, satisfying Rules.md §4. |
| 5 | Drop unencodable columns with cardinality >20 & unique ratio >0.4 | This rule-based check filters out IDs and unique text descriptors, logging them in `preprocessing_log` and generating a warning in `errors`. |
| 6 | Column prefix cleanup | Cleaned up output features from `ColumnTransformer` (stripped `numeric__` and `categorical__` prefixes) to maintain readable CSV output column names. |
| 7 | **Model zoo split into `src/model_zoo/` package** (Phase 4) | Three files (`classification_models.py`, `regression_models.py`, `clustering_models.py`) + `__init__.py` defining the `ModelSpec` TypedDict. Per Rules.md §9 — keeps the orchestrator file under the ~300-line cap. |
| 8 | **One experiment per ModelSpec, no cartesian grid expansion** (Phase 4) | Each zoo entry has a single fixed param dict applied as-is. Keeps scope tight per Phases.md Phase 4 "fixed grids" and Rules.md §7. |
| 9 | **Dummy baseline always included** (Phase 4) | DummyClassifier/DummyRegressor (predict majority class / mean) included in classification/regression zoos so the leaderboard always has something to beat. |
| 10 | **Selective `RuntimeError` swallow around `interrupt()`** (Phase 4) | `interrupt()` raises a special GraphInterrupt inside a compiled LangGraph (correct pause behavior) but raises a plain `RuntimeError("Called get_config outside of a runnable context")` when called from a unit test. Only the latter is caught — the real GraphInterrupt propagates so the checkpoint actually pauses. |
| 11 | **Per-experiment metrics chosen by task type** (Phase 4) | Classification → accuracy/f1/precision/recall (weighted if multiclass). Regression → rmse/mae/r2. Clustering → silhouette (if ≥2 clusters), n_clusters, n_noise. Centralizing the mapping here means Phase 5 can re-use the same metric names without re-computation. |
| 12 | **All-fail detection happens AFTER all runs complete** (Phase 4) | Don't short-circuit the ThreadPoolExecutor on individual failures — let every model attempt to run, then assess at the end. Gives the user a complete diagnostic rather than an early termination. |

---

## Key File Locations

```
ML_workflow/
├── AgentML/                  ← Project design specifications
│   ├── PRD.md
│   ├── Architecture.md
│   ├── Rules.md
│   ├── Phases.md
│   ├── Design.md
│   └── Memory.md
├── src/
│   ├── orchestrator/
│   │   ├── state.py          ← AgentMLState TypedDict (exact Architecture.md schema)
│   │   └── graph.py          ← LangGraph StateGraph pipeline (already wired for Phase 4)
│   ├── agents/
│   │   ├── profiling_agent.py           ← Dataset Profiling Agent (implemented)
│   │   ├── preprocessing_agent.py       ← Data Preprocessing Agent (implemented)
│   │   ├── problem_detection_agent.py   ← Problem Detection Agent (implemented)
│   │   ├── experiment_orchestrator_agent.py  ← Experiment Orchestrator (implemented)
│   │   ├── evaluation_agent.py          ← Model Evaluation Agent stub
│   │   └── report_agent.py              ← Report Agent stub
│   ├── model_zoo/                       ← Phase 4 NEW PACKAGE
│   │   ├── __init__.py                  ← ModelSpec TypedDict
│   │   ├── classification_models.py     ← 6 fixed classifiers
│   │   ├── regression_models.py         ← 7 fixed regressors
│   │   └── clustering_models.py         ← 5 fixed clusterers
│   └── llm/
│       └── groq_client.py    ← Thin Groq wrapper
├── tests/
│   ├── test_phase0.py                   ← Foundation tests (5 currently failing — pre-existing)
│   ├── test_profiling_agent.py          ← 4 passing
│   ├── test_preprocessing_agent.py      ← 3 passing
│   ├── test_problem_detection_agent.py  ← 4 passing
│   └── test_experiment_orchestrator.py  ← 8 passing (Phase 4 NEW)
├── data/
│   ├── sample.csv            ← Clean binary classification dataset
│   ├── sample_missing.csv    ← Dataset containing missing values
│   └── sample_duplicates.csv ← Dataset containing duplicate rows
├── .env.example
├── requirements.txt
└── Memory.md                 ← THIS FILE
```

---

## Next Steps (Phase 5 — Model Evaluation Agent)

When the user gives the go-ahead:

1. Implement `evaluation_agent()` in `src/agents/evaluation_agent.py`:
   - Read `state["experiment_results"]` produced by Phase 4.
   - Rank models per task type:
     - Classification: rank by `accuracy` (primary), then `f1` as tiebreaker.
     - Regression: rank by `r2` descending; tiebreak on `rmse` ascending.
     - Clustering: rank by `silhouette` descending (NaN treated as worst).
   - Skip models with `success=False` (they're already isolated, never re-raise).
   - Build `state["ranking"]` as a list of `{model_id, model_name, primary_metric, metrics}`.
   - Set `state["best_model_id"]` to the top entry's `model_id`.
   - Append a reasoning string to `state["detection_reasoning"]` or a new field (per Rules.md §5 — explainability).
   - Architecture.md §6 checkpoint 3 (always-offered) — confirm or override the best-model pick via `interrupt()`. Use the same selective-RuntimeError swallow pattern as Phase 4.

2. Add unit tests in `tests/test_evaluation_agent.py` covering:
   - Classification ranking with a clear winner.
   - Regression ranking with a clear winner.
   - Clustering ranking (handling NaN silhouette gracefully).
   - Human override of best_model_id via interrupt + resume.
   - Mixed success/failure result list (failed models excluded from ranking).

3. Verify Phase 5 "done when" criteria: produces a correct ranked leaderboard for each task type, and override checkpoint correctly updates `best_model_id`.

4. Address pre-existing Phase 0 test failures (5 min mechanical fix — pass `configurable={"thread_id": ...}` to the test's `g.invoke(initial)` calls).
