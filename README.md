# AgentML — Multi-Agent ML Experiment Orchestrator

AgentML is a **7-agent LangGraph pipeline** that takes a raw CSV and produces a
complete, human-readable machine-learning experiment report — autonomously. It
profiles the data, preprocesses it, detects the ML task type, runs a fixed model
zoo concurrently, ranks the results into a leaderboard, and writes a structured
Markdown report with optional LLM narration.

No deep-learning frameworks, no hyperparameter-optimization libraries, no
*automatic* feature engineering — you choose which engineered features to create
at a dedicated human checkpoint, and the pipeline applies them deterministically.
A transparent, explainable, rule-based automation built on
**scikit-learn + xgboost** inside **LangGraph**.

## Pipeline Overview

```
CSV → Profiling → Problem Detection → Feature Engineering (human checkpoint)
    → Preprocessing → Model Zoo (concurrent) → Model Evaluation
    → Report Generation → outputs/reports/report_<session>.md
```

| # | Agent | What it does |
|---|-------|--------------|
| 0 | Orchestrator | Initialises cross-cutting state (session id, status). |
| 1 | Dataset Profiling | Shape, dtypes, missing %, duplicate count, per-column stats; also **detects & suggests** feature-engineering opportunities (datetime decompose, correlated-pair ratios, binning) without applying them. |
| 2 | Problem Detection | Resolves target column + task type (classification / regression / clustering); pauses for confirmation when ambiguous. Runs **before** preprocessing so the (possibly LLM-resolved) target is excluded from feature scaling. |
| 3 | Feature Engineering (checkpoint) | Always-offered human checkpoint. User checks which suggested features to create and/or types a custom formula. Custom formulas are evaluated with a **restricted safe parser** (never `eval`). Default: apply none. |
| 4 | Data Preprocessing | Dedup, imputation (median/mode), one-hot encoding, standard scaling, drops unusable columns, and **deterministically applies** the user-selected + custom features (logged in `feature_engineering_log`). The resolved target column is preserved as-is (never scaled). |
| 5 | Experiment Orchestrator | Runs the fixed model zoo for the task type concurrently (ThreadPoolExecutor); isolates failures. |
| 6 | Model Evaluation | Builds a ranked leaderboard, picks the best model, offers a human override. |
| 7 | Report Generation | Renders a Jinja2 template + optional Groq narration; writes Markdown and a PDF (pure-Python reportlab). |

Human-in-the-loop checkpoints sit at Problem Detection (ambiguous task), a
Feature Engineering Selection step (always offered, right after detection), the
Model Orchestrator (scope), and Model Evaluation (best-model override). They are
offered via LangGraph `interrupt()` and are usable both from the Streamlit UI and
programmatically.

## Project Layout

```
AgentML/                 Project design specs (PRD, Architecture, Rules, Phases, etc.)
src/
  orchestrator/          state.py (AgentMLState) + graph.py (LangGraph pipeline)
  agents/                profiling, preprocessing, problem_detection,
                         experiment_orchestrator, evaluation, report
  model_zoo/             classification_models, regression_models, clustering_models
  llm/                   groq_client.py (the ONLY module that may call Groq)
  metrics/               task_metrics.py (metric selection + leaderboard ranking)
templates/               report_template.md.j2
scripts/                 run_phase6_e2e.py, perf_check_50k.py
tests/                   unit + integration tests
data/                    sample CSVs (incl. messy real-world-ish examples)
outputs/reports/         generated reports (gitignored)
```

## Requirements

- **Python 3.10+** (developed/verified on 3.13)
- Dependencies in `requirements.txt` (includes `langgraph`, `pandas`,
  `scikit-learn`, `xgboost`, `streamlit`, `jinja2`, `groq`, `python-dotenv`).
- An optional **Groq API key** for LLM narration. Without one, the pipeline still
  runs and produces a complete *template-only* report.
- The default LLM is `openai/gpt-oss-20b` (override with `GROQ_MODEL` in `.env`).
  Older Groq `llama3-*` model IDs are decommissioned and will 400.

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the Groq API key (optional but recommended)
cp .env.example .env        # Windows:  copy .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_from_https://console.groq.com
```

> If you skip the `.env` step, everything still works — the report simply falls
> back to template-only mode (no LLM narration).

## Running the App

### A. Streamlit UI (recommended for the demo)

```bash
streamlit run app.py
```

This opens a web dashboard where you can:

1. **Upload a CSV**, then **describe what to predict in the chat field** — e.g. *"this
   is the Titanic CSV, predict who survived or not"*. The pipeline resolves that into
   a target column + task type (grounded in your actual columns). Leave it blank to
   auto-detect, or say "just cluster it" for unsupervised.
2. Click **Initialize & Start Pipeline**.
3. Watch the **clickable workflow status bar** advance. Each stage shows its status
   (○ pending · ● running · ✓ done · ⚠ needs attention · ✗ failed). Implementation
   details stay hidden while agents run — click a stage's **Open** button to inspect
   its individual output.
4. Respond to **human-in-the-loop checkpoints** (amber panels) when the pipeline
   needs a decision — confirm/override the task detection, choose which
   **feature-engineering** suggestions to create (and optionally type a custom
   formula), set the experiment scope, and confirm/override the best model.
5. Once every stage finishes, a **Workflow Complete summary** shows the whole run
   at a glance (rows, columns, task type, best model) with the full **Report Viewer**
   (downloadable as Markdown or PDF, the PDF generated automatically via `reportlab`).

> **Note:** Usless unique-ID / free-text columns (e.g. `PassengerId`, `Name`) are
> dropped automatically during preprocessing. The natural-language target is optional —
> without it (and without a Groq key) the pipeline still auto-detects and runs.

### B. Command line

Run the full pipeline on a CSV and print a summary:

```bash
python -m src.orchestrator.graph data/loan_applications.csv approved
# first arg = CSV path (default data/sample.csv)
# second arg = optional target column (default: auto-detect)
```

### C. End-to-end demo script

Generates a realistic synthetic dataset and runs the whole pipeline, auto-accepting
each checkpoint. Add `--simulate-groq-failure` to prove the LLM-fallback path:

```bash
python scripts/run_phase6_e2e.py
python scripts/run_phase6_e2e.py --simulate-groq-failure
```

Output reports are written to `outputs/reports/report_<session_id>.md`.

## Sample Datasets

- `data/sample.csv` — clean binary classification.
- `data/sample_missing.csv` — classification with missing values.
- `data/sample_duplicates.csv` — classification with duplicate rows.
- `data/loan_applications.csv` — clean classification loan data.
- `data/messy_loan_data.csv` — messy*: an ID column, a 100%-null column, missing
  values, and a categorical feature (exercises the preprocessing edge cases).

## Running the Tests

```bash
python -m pytest tests/ -v
```

The suite covers profiling, preprocessing, problem detection, feature-engineering
suggestion detection + safe custom-formula handling, the experiment orchestrator
(including failure isolation and all-fail hard-fail), model evaluation (ranking +
human override), and edge cases (empty CSV, single-column, all-missing columns,
huge-cardinality categoricals, no-numeric datasets, detection-checkpoint resume,
and the always-offered feature-engineering checkpoint).

## Performance

`scripts/perf_check_50k.py` generates a 55,000-row classification CSV and times
the full pipeline through every agent. On a typical laptop the run completes in
well under a minute (profiling/preprocessing ~seconds; concurrent model training
is the dominant cost — ~30 s for 6 models). Nothing chokes or times out.

## Known Limitations

- **Groq narration is best-effort.** If the API key is missing/invalid or a call
  fails, the pipeline logs a recoverable error and produces a complete
  *template-only* report. It never blocks the flow.
- **Fixed model zoos only.** Each task type has a fixed, shallow set of models
  with fixed hyper-parameters — no HPO, no deep learning (by design, see
  `AgentML/Rules.md`).
- **Preprocessing drops free-text / huge-cardinality columns** (rule-based:
  cardinality > 20 with unique ratio > 0.4) and columns that are 100% missing.
  It does not do text vectorisation. Feature engineering is **user-driven**: the
  Profiling agent *suggests* candidates (datetime decompose, correlated-pair
  ratios, binning) and the user opts in via the Feature Engineering checkpoint;
  custom formulas are limited to `+ - * /`, parentheses, numbers, and existing
  column names (restricted safe parser, no `eval`).
- **PDF export** is generated with `reportlab` (pure Python, in `requirements.txt`).
  It covers the report's standard constructs (headings, bold/italic/inline code,
  pipe tables); unusual custom Markdown may not render identically to a browser.
- **Scaling.** The experiment phase is CPU-bound; very large datasets or many
  workers may be slow. Confirm the scope vs. time at the orchestrator checkpoint.

## Documentation

Full design decisions live in `AgentML/` (`PRD.md`, `Architecture.md`,
`Rules.md`, `Phases.md`, `Design.md`). The build log is summarized in
`Memory.md`.
