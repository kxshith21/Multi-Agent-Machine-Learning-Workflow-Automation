# Architecture.md — AgentML

## 1. High-Level Flow

```
User uploads CSV
      │
      ▼
┌─────────────────┐
│ Orchestrator     │  ← central controller, owns AgentMLState
└────────┬─────────┘
         ▼
┌─────────────────────┐
│ Dataset Profiling    │
└────────┬─────────────┘
         ▼
┌─────────────────────┐
│ Data Preprocessing   │
└────────┬─────────────┘
         ▼
┌─────────────────────┐
│ Problem Detection    │──► [HUMAN CHECKPOINT: confirm task type + target column
└────────┬─────────────┘     if confidence < threshold]
         ▼
   (branch: classification/regression path vs clustering path —
    same Experiment Orchestrator, different model zoo + metrics)
         ▼
┌─────────────────────┐
│ Experiment           │──► [HUMAN CHECKPOINT: optional — cap runtime/experiment count]
│ Orchestrator         │
└────────┬─────────────┘
         ▼
┌─────────────────────┐
│ Model Evaluation      │──► [HUMAN CHECKPOINT: confirm/override "best" model]
└────────┬─────────────┘
         ▼
┌─────────────────────┐
│ Report Generation     │
└────────┬─────────────┘
         ▼
   Final Markdown/PDF Report
```

Supervised/Unsupervised "pipelines" are **not agents** — they are conditional branches the Orchestrator takes based on Problem Detection's output. Branch-specific logic (train/test split vs. dimensionality prep) lives inside the Preprocessing Agent (task-aware step, run after task type is known) and inside the Experiment Orchestrator (different model zoo per task).

> **Task-aware prep:** for `task_type == "clustering"` only, the Preprocessing Agent
> compresses the imputed/scaled/encoded feature matrix to `min(50, n_features)`
> components (`TruncatedSVD` by default; `PCA` or `off` configurable via
> `PREPROCESS_DIM_REDUCTION`, component count via `PREPROCESS_DIM_REDUCTION_COMPONENTS`)
> before the model zoo runs. Supervised tasks keep raw, interpretable features.
> A failed reduction is recoverable and falls back to the unreduced matrix.

## 2. Agents (7 total)

| # | Agent | Responsibility |
|---|---|---|
| 1 | Orchestrator | Owns state, invokes agents, handles branching, manages checkpoints |
| 2 | Dataset Profiling | Structure/dtype/missing/duplicate analysis |
| 3 | Data Preprocessing | Cleaning, encoding, scaling, task-aware prep |
| 4 | Problem Detection | Task type + target column inference, with reasoning |
| 5 | Experiment Orchestrator | Runs model zoo concurrently, logs all results. Imbalanced classification: `class_weight="balanced"` (LR/RF/SVC) + per-dataset XGBoost `scale_pos_weight` (no SMOTE — sklearn/xgboost only) |
| 6 | Model Evaluation | Task-appropriate metrics, ranking, best-model selection. Classification ranks by F1 (macro), tiebreaker precision; `pr_auc` reported for binary; minority <10% adds an imbalance explanation to the reasoning |
| 7 | Report Generation | Final explainable report (LLM-narrated from structured state) |

## 3. Shared State Schema

```python
class AgentMLState(TypedDict):
    raw_file_path: str
    dataset_profile: dict
    clean_dataset_path: str
    preprocessing_log: list[dict]
    task_type: Literal["classification", "regression", "clustering"]
    target_column: str | None
    detection_confidence: float
    detection_reasoning: str
    class_balance: dict              # classification only: normalized value_counts (e.g. {"0": 0.9417, "1": 0.0583}) — surfaced before training
    experiment_results: list[dict]
    best_model_id: str
    ranking: list[dict]
    report_path: str
    errors: list[dict]
    status: Literal["running", "needs_human_input", "completed", "failed"]
```

## 4. Tech Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| LLM (reasoning + report narration) | Groq API (provider abstracted at code level) |
| Data handling | pandas |
| Profiling | pandas + ydata-profiling (or hand-rolled summary functions) |
| Preprocessing | scikit-learn (ColumnTransformer, SimpleImputer, OneHotEncoder, StandardScaler, TruncatedSVD/PCA for clustering) |
| Modeling | scikit-learn + xgboost |
| Concurrency | concurrent.futures.ThreadPoolExecutor |
| Experiment logging | JSON to state (v1); MLflow optional later |
| Report output | Markdown via Jinja2 → optional PDF |
| CLI/Demo interface | Streamlit (matches prior team convention) |

## 5. Folder Structure

```
agentml/
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Design.md
├── Memory.md
├── src/
│   ├── orchestrator/
│   │   ├── graph.py              # LangGraph state graph definition
│   │   └── state.py              # AgentMLState schema
│   ├── agents/
│   │   ├── profiling_agent.py
│   │   ├── preprocessing_agent.py
│   │   ├── problem_detection_agent.py
│   │   ├── experiment_orchestrator_agent.py
│   │   ├── evaluation_agent.py
│   │   └── report_agent.py
│   ├── model_zoo/
│   │   ├── classification_models.py
│   │   ├── regression_models.py
│   │   └── clustering_models.py
│   ├── metrics/
│   │   └── task_metrics.py       # metric mapping per task type
│   ├── llm/
│   │   └── groq_client.py        # provider abstraction
│   └── utils/
│       ├── file_io.py
│       └── logging.py
├── templates/
│   └── report_template.md.j2
├── data/
│   └── uploads/                  # gitignored
├── outputs/
│   └── reports/                  # gitignored
├── tests/
│   ├── test_profiling_agent.py
│   ├── test_preprocessing_agent.py
│   ├── test_problem_detection_agent.py
│   ├── test_experiment_orchestrator.py
│   └── test_evaluation_agent.py
├── app.py                        # Streamlit entrypoint
├── requirements.txt
└── .env.example                  # GROQ_API_KEY placeholder
```

## 6. Human-in-the-Loop Mechanism
Implemented via LangGraph's `interrupt()`, same pattern as CRCM. Three checkpoints:
1. Post Problem Detection (conditional — only if `detection_confidence` below threshold, e.g. 0.7)
2. Pre Experiment Orchestrator (optional, always offered — user can accept defaults or adjust scope)
3. Post Model Evaluation (always offered — confirm or override best model pick)

## 6a. Metrics & Ranking
Rule-based and reproducible (`src/metrics/task_metrics.py`).

| Task type | Primary (ranking) | Tiebreaker | Reported (never ranked) |
|---|---|---|---|
| Classification | `f1` (macro, higher better) | `precision` (macro) | `recall`, `accuracy`, `pr_auc` (binary only) |
| Regression | `r2` (higher better) | `rmse` (lower better) | `mae` |
| Clustering | `silhouette` (higher better, NaN = worst) | `n_clusters` | `n_noise` |

Classification deliberately ranks on **F1 rather than accuracy** so a
majority-class baseline cannot win an imbalanced dataset by predicting the modal
class. `class_balance` is logged by Problem Detection on the resolved target and
surfaced in the report; when any class is <10% of samples, the evaluation
reasoning states that F1/PR-AUC were prioritized over accuracy.

## 7. Error Handling Flow
See Rules.md §4 for the full policy. Summary: profiling failures are hard stops; preprocessing/experiment failures are logged and skipped where possible; ambiguous detection routes to a checkpoint rather than a silent guess.
