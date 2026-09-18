# AgentML — Session Memory Log

> This file is updated at the end of every session.
> Future sessions read this first to avoid re-reading the entire codebase.

---

## Current Phase

**Phase 10 — Feature Engineering Selection (4th human checkpoint)** ✅ COMPLETE
**Project: DONE + extension.** Feature-engineering checkpoint added.
Full test suite green (64 passed).

---

## Last Updated

2026-08-27T21:00:00+05:30

---

## What's Done

### Phase 0 — Foundation (✅ Complete)
- Repo structure, `AgentMLState` schema (matches `Architecture.md` §3), LangGraph
  skeleton, `src/llm/groq_client.py` (only module allowed to call Groq),
  `.env.example`, `requirements.txt`.
- **The historical Phase 0 test failures** (checkpointer requiring `thread_id` on
  direct `g.invoke(initial)` calls) were resolved in a prior Phase 7 session by
  passing `config={"configurable": {"thread_id": ...}}` and adding an
  auto-resume helper. All Phase 0 tests now pass.

### Phase 1 — Dataset Profiling Agent (✅ Complete)
- Loads CSV with pandas; computes shape, dtypes, missing %, duplicate count,
  per-column stats into `dataset_profile`. Invalid/unreadable CSV hard-fails.
- 4/4 tests pass.

### Phase 2 — Data Preprocessing Agent (✅ Complete)
- sklearn `ColumnTransformer` pipeline: dedup, imputation (median numeric / mode
  categorical), `OneHotEncoder`, `StandardScaler`, drops unencodable high-cardinality
  columns (cardinality > 20 & unique ratio > 0.4). Logs every op with reasoning.
- **Phase 7 hardening added:** drops 100%-missing columns (recoverable warning),
  and hard-fails (`zero_features_error`) if preprocessing leaves 0 features.
- 3/3 tests pass (+ edge-case tests).

### Phase 3 — Problem Detection Agent (✅ Complete)
- Target resolution precedence (Phase 9 revised): **user instruction (chat bar) →
  explicit user-specified column → auto-detect heuristic (name match / last column) →
  clustering (only when explicitly requested)**. Rule-based task classification
  (classification / regression / clustering), confidence scoring, Groq-narrated
  reasoning, `interrupt()` when confidence < 0.7.
- NOTE: when a detection checkpoint is resumed, it MUST be resumed with an
  explicit `{"target_column": ..., "task_type": ...}` payload. Resuming with an
  empty `{}` makes LangGraph re-trigger the interrupt indefinitely (fixed in Phase 8).
- 9/9 tests pass.

### Phase 4 — Model Zoo + Experiment Orchestrator (✅ Complete)
- Fixed zoos in `src/model_zoo/` (one file per task type, sklearn + xgboost only,
  fixed grids): classification 6, regression 7, clustering 5. Uniform `ModelSpec`
  TypedDict in `__init__.py`.
- `experiment_orchestrator_agent` dispatches by `task_type`, runs concurrently via
  `ThreadPoolExecutor`, logs every run to `experiment_results`, isolates single-model
  failures (log-and-continue), and hard-fails with `all_experiments_failed` if ALL fail.
- Scope checkpoint (max_experiments / max_workers / time_cap) offered via `interrupt()`.
- 8/8 tests pass.

### Phase 5 — Model Evaluation Agent (✅ Complete)
- Task-appropriate metrics (accuracy/f1 vs r2/rmse vs silhouette/n_clusters/n_noise),
  ranked leaderboard (excludes failures), best-model selection, human override via
  `interrupt()`.
- 14/14 tests pass.

### Phase 6 — Report Generation Agent (✅ Complete)
- Jinja2 template `templates/report_template.md.j2` (all required sections incl.
  failed-experiments leaderboard), Groq narration layered on top (never alters facts,
  Rules.md §3), graceful fallback to template-only report when Groq is unavailable.
- Verified end-to-end and via `--simulate-groq-failure`.

### Phase 7 — Integration, Edge Cases, Polish (✅ Complete) — `app.py`
- **Streamlit UI** (`app.py`): file upload, **clickable workflow status bar** (each
  stage shows ○ pending / ● running / ✓ done / ⚠ / ✗, with implementation details
  hidden until a node is clicked), human-in-the-loop checkpoint panels (detection /
  scope / best-model override), a **Workflow Complete summary** at the end, and a
  report viewer + Markdown/PDF download. Verified it boots.
- **Edge cases handled** and tested in `tests/test_edge_cases.py`: empty CSV (hard
  fail), single-column/target-only (hard fail, `zero_features_error`), all-missing
  columns (dropped with warning), no-numeric datasets (categorical-only pipeline),
  huge-cardinality categorical (dropped).
- **Real-world messy CSV validated** (`data/messy_loan_data.csv`: ID column, 100%-null
  column, missing values, categorical feature) runs end-to-end to a completed report.

### Phase 8 — Buffer / Hardening (✅ Complete this session)
1. **Fixed a Phase 7 integration bug — detection-checkpoint infinite loop.**
   Resuming the Phase 3 detection `interrupt()` with an empty `{}` payload re-triggers
   the interrupt indefinitely (LangGraph treats empty resume as "no value"). This bit
   the auto-resume helpers in `test_phase0.py`, `scripts/run_phase6_e2e.py`, and
   `run_pipeline()`. Fix: always resume detection with the detected target/task.
   Added a regression test (`test_edge_case_detection_checkpoint_resume_does_not_loop`).
2. **Hardened the CLI convenience runner** — `run_pipeline()` in `graph.py` now
   auto-accepts every checkpoint to complete end-to-end, so the documented
   `python -m src.orchestrator.graph` command finishes (previously it stopped at the
   first interrupt).
3. **Performance check on 50k+ rows — PASS.** `scripts/perf_check_50k.py` generates a
   55,000-row × 13-column classification CSV and runs the full pipeline:
   - Total wall time **~36 s** (no timeout, nothing choked).
   - Setup → profiling → preprocessing → detection: **3.3 s**.
   - Concurrent training of 6 models: **31.3 s** (dominant cost).
   - Evaluation + report: **<0.1 s**. 0 hard errors.
4. **Documentation pass — COMPLETE.** Full `README.md` (setup, running the Streamlit
   app, CLI, demo script, tests, performance, known limitations). `.env.example`
   confirmed current (`GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_TEMPERATURE`). `.gitignore`
   updated to exclude generated `outputs/` and `data/clean_dataset_*` / `data/upload_*`.
5. **PDF download FIXED** — the app's "Download PDF" button never appeared because
   the PDF backend (`weasyprint`+`markdown`) was an uninstalled optional dependency
   and `_maybe_pdf()` silently returned `None`. Rewrote PDF generation using
   **reportlab** (pure Python, already installed, Windows-friendly) in a new module
   `src/utils/pdf_export.py`. Now every report auto-produces a `.pdf` next to the
   `.md`, and the Streamlit download button works. See decision #18.
6. **Final Memory.md summary — this file.**

### Phase 9 — Natural-Language Prediction Target (✅ Complete this session)
1. **Chat-bar instruction drives the prediction target.** After uploading a CSV the
   user can describe what to predict in plain language (e.g. *"this is the Titanic
   CSV, predict who survived or not"*). `problem_detection_agent.infer_target_from_instruction()`
   asks Groq for a strict `{"target_column","task_type"}` JSON, validated/fuzzy-matched
   against the real CSV columns. The resolved target drives the rest of the run.
2. **Precedence + bug fix.** Resolution order is now instruction → explicit column →
   auto-detect → clustering. This fixed a latent bug: a blank/`None` target silently
   forced `clustering` instead of auto-detecting (contradicted the old UI label).
3. **Explicit clustering is now expressed via the instruction** ("no target / just
   cluster it"), not via `target_column=None` (which now means auto-detect). The old
   `None`-means-clustering unit test was updated to the new contract.
4. **Graceful fallback.** Without a Groq key/model, or if parsing/validation fails, the
   agent falls back to the heuristic (recoverable) — never blocks. Auto-detect name
   list expanded (`survived, churn, default, response, outcome, result, prediction`).
5. **Streamlit UI** replaces the target text box with a chat/text area. The Problem
   Detection card shows the user's instruction + resolved target. PassengerId/Name-style
   free-text ID columns are already dropped by the Phase 2 high-cardinality rule (no change).
6. **Tests.** Detection suite grew to 9 tests (instruction→classification / clustering /
   unknown-column fallback / blank auto-detect regression / parse-error fallback), plus
   narration mocked for determinism. Full suite: **56 passed**.
7. **Graph reorder (bug fix).** Problem Detection now runs **before** Data Preprocessing
   (`graph.py`: profiling → detection → preprocessing). Previously preprocessing ran first;
   when the target wasn't known yet (chat instruction / `target_column=None`), the target
   column was StandardScaled to `[-1, 1]`, which broke XGBoost ("Invalid classes inferred…
   got [-1. 1.]"). Detection reads the raw file, so it doesn't need the clean data. Added a
   regression test asserting the binary target stays `{0, 1}` after preprocessing.

---

## Current Status

- **All 64 tests pass** (`python -m pytest tests/ -v`).
- **E2E verified** by `scripts/run_phase6_e2e.py` (including `--simulate-groq-failure`
  fallback) and the CLI runner.
- **Streamlit app boots and serves** (`streamlit run app.py`).
- **50k-row performance test passes** (~36 s end-to-end).
- **Live Groq narration VERIFIED working.** A user-supplied `GROQ_API_KEY` in the
  local `.env` is active; the default model was migrated (see decision #17) and an
  e2e run produced a **Full (structured + LLM narration)** report with 0 hard errors.
- **Natural-language target VERIFIED working** — a chat instruction of *"predict who
  survived"* on a Titanic-like CSV resolves to `Survived` / `classification`
  end-to-end (Phase 9).
- **Classification ranking now F1-first with imbalance handling.** Primary metric
  changed from `accuracy` → `f1` (macro), tiebreaker `precision`; `pr_auc` reported
  (binary). Zoo grew to 7 models with `SVC_rbf`; `class_weight="balanced"` on
  LR/RF/SVC, XGBoost `scale_pos_weight` computed per-dataset. `class_balance` vs
  `{0: 0.95, 1: 0.05}` surfaced pre-training; <10% minority triggers an
  F1/PR-AUC-prioritized explanation. **80 tests pass** including a synthetic 95/5
  e2e proving the majority baseline (ACC 0.94, F1 0.00) ranks LAST.
- **DeepEval benchmarks extended from 2 agents → all 7 pipeline areas.** 8 test
  functions / 9 benchmark rows (Problem Detection, Profiling, FE suggestions,
  Preprocessing, Orchestrator integrity, Evaluation ranking, Report faithfulness,
  hallucination, Q&A relevancy). Uses `metric.measure()` + rate-limit chain-aware
  backoff (free Groq TPM = 8000/min) so the whole file passes in one invocation:
  `pytest tests/test_agent_deepeval.py` → **8 passed**.

---

## Test Results

**`64 passed`**

| File | Count | Notes |
|------|-------|-------|
| `test_phase0.py` | 11 | schema, stub/E2E pipeline, edge cases |
| `test_profiling_agent.py` | 8 | clean/missing/duplicates/corrupt + feature-suggestion detection (datetime/corr-pair/binning) + no-op |
| `test_preprocessing_agent.py` | 5 | unencodable/all-numeric/mixed + selected+custom feature application + malicious-formula safety |
| `test_problem_detection_agent.py` | 9 | classif/regress/cluster/ambiguous+interrupt + instruction-driven/fallback/auto-detect |
| `test_experiment_orchestrator.py` | 8 | zoos, failure isolation, all-fail, scope, single-run |
| `test_evaluation_agent.py` | 14 | ranking strategies, overrides |
| `test_edge_cases.py` | 9 | empty/single-col/all-missing/no-numeric/huge-card + detection-resume + binary-target-not-scaled + feature-eng-checkpoint-always-offered |

---

## Known Issues / Known Limitations (all explicitly documented)

- **Groq narration requires a live key + a currently-supported model.** The old
  default `llama3-70b-8192` was decommissioned by Groq (400 `model_decommissioned`);
  the default is now `openai/gpt-oss-20b`. Without a valid key/model, narration
  falls back to template-only reports (fallback verified).
- **Fixed model zoos only** — no HPO / deep learning (by design, Rules.md §3).
- **Free-text / huge-cardinality columns are dropped**, not vectorized.
- **PDF export uses `reportlab`** (pure Python, in `requirements.txt`). It covers the
  report's standard constructs (headings, bold/italic/inline code, pipe tables);
  unusual custom Markdown may not render identically to a browser.
- **Experiment phase is CPU-bound** — large datasets / many workers can be slow;
  confirm scope vs. time at the orchestrator checkpoint.
- No other open blockers.

---

## Decisions Made Since the Design Docs

(inherited from prior phases) — see the "Decisions Made" section in the repo history
(`Architecture.md`-aligned schemas, standard-Python stats for JSON-serializability,
native-exception hard fails, rule-based column dropping, `model_zoo` package split,
one fixed config per ModelSpec, Dummy baselines, selective RuntimeError swallow around
`interrupt()`, task-based metrics, all-fail detection after all runs, unified report
fallback).

**New decisions this session (Phase 8):**
| # | Decision | Reason |
|---|----------|--------|
| 15 | Auto-resume detection checkpoints with explicit target/task payload, never `{}` | Empty resume re-triggers LangGraph interrupt indefinitely (observed bug). |
| 16 | `run_pipeline()` auto-accepts all checkpoints | Makes the CLI/documented runner finish end-to-end and matches the UI's accept-defaults flow. |
| 17 | Default Groq model migrated `llama3-70b-8192` → `openai/gpt-oss-20b` | Groq decommissioned the llama3-70b-8192 offering (verified via API model list); gpt-oss-20b is currently supported, fast, and needs no extra params. Applied in `groq_client.py`, `.env.example`, and the local `.env`. |
| 18 | PDF export rewritten to use `reportlab` (`src/utils/pdf_export.py`) instead of optional `weasyprint`+`markdown` | weasyprint was an uninstalled optional dep, so no PDF was ever generated and the app's Download-PDF button never appeared. reportlab is pure Python, Windows-friendly, and already installed; PDF is now generated by default. |
| 19 | Chat bar (natural-language instruction) is the primary way to specify the prediction target; `target_column=None` now means **auto-detect**, and genuine clustering is requested via the instruction ("no target / just cluster") | Users think in natural language (e.g. Titanic → "predict who survived"); LLM resolves it to a real column/task with graceful fallback. Also fixes the latent bug where a blank target silently forced clustering (see Phase 9). |
| 20 | Problem Detection runs **before** Data Preprocessing in the graph | Preprocessing was scaling the (then-unknown) target column to `[-1, 1]`, breaking XGBoost on binary labels. Detection reads the raw file, so it doesn't need the clean data; running it first lets preprocessing exclude/preserve the target. |
| 21 | Added a **4th, always-offered, user-driven "Feature Engineering Selection" checkpoint** between Problem Detection and Data Preprocessing (beyond the three in Architecture.md §6) | The user wants explicit, deterministic control over *which* engineered features are included, rather than full automation or LLM-driven auto-engineering. The Profiling Agent only **detects & suggests** plain-language candidates (datetime decompose, correlated-pair ratio/product, high-cardinality binning); the human opts in via a Streamlit checklist and/or adds custom formulas. Rules.md §6 forbids removing the original 3 checkpoints — this is an **addition** (allowed since documented). |
| 22 | Default behavior at the Feature Engineering checkpoint is **apply none** | Auto-accept / CLI / "Confirm" with nothing checked creates no suggested features — the user explicitly opts in to each one (transparency + determinism, consistent with Rules.md §3 & §5). |
| 23 | Custom feature formulas are evaluated with a **restricted safe parser** (`src/utils/safe_formula.py`), never raw `eval()` | Safety requirement: only `+ - * /`, parentheses, numbers, and existing column names are allowed. Function calls, imports, attribute access (dots), brackets, string literals, etc. are rejected with a `FormulaValidationError` and surfaced to the user (recoverable error entry) — proven by a malicious-input test (`__import__('os').system('ls')` is rejected, not executed). |

---

## Key File Locations

```
AgentML/                 Project design specs (PRD, Architecture, Rules, Phases, Design)
src/
  orchestrator/state.py  AgentMLState TypedDict
  orchestrator/graph.py  LangGraph pipeline + run_pipeline() (auto-resumes checkpoints)
  agents/                profiling, preprocessing, problem_detection,
                         experiment_orchestrator, evaluation, report
  model_zoo/             classification_models / regression_models / clustering_models
  llm/groq_client.py     ONLY module that may call Groq
  metrics/task_metrics.py
templates/report_template.md.j2
app.py                   Streamlit UI (Phase 7)
scripts/run_phase6_e2e.py
scripts/perf_check_50k.py
tests/                   all unit + integration + edge-case tests
data/                    sample + real-world-ish CSVs
outputs/reports/         generated reports (gitignored)
```

---

## How to Verify the Done Criteria

1. **Known issues resolved/documented** — the only recorded "blocker" (Phase 0
   checkpointer tests) is fixed; the detection-resume loop introduced this session is
   fixed and regression-tested; remaining items are documented known limitations above.
2. **50k+ row test** — `python scripts/perf_check_50k.py` completes in ~36 s.
3. **README from scratch** — `README.md` walks through venv → install → `.env` →
   `streamlit run app.py` (with checkpoint walk-through) → CLI → demo script → tests.
```