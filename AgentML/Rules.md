# Rules.md — AgentML

Boundaries for any AI (or human) writing code in this project. If a change conflicts with these rules, stop and flag it rather than silently deviating.

## 1. Libraries — Use / Avoid

**Use:**
- `pandas` for all dataframe operations
- `scikit-learn` for preprocessing transformers and the classical model zoo
- `xgboost` for gradient boosting
- `langgraph` for orchestration and state management
- `concurrent.futures.ThreadPoolExecutor` for experiment concurrency
- `Jinja2` for report templating
- `streamlit` for the demo UI
- Groq API via a thin abstraction layer (`src/llm/groq_client.py`) — never call the Groq SDK directly from an agent file

**Avoid (v1):**
- No deep learning frameworks (PyTorch, TensorFlow, Keras) — out of scope, adds GPU dependency we don't have
- No `asyncio` for experiment concurrency — threads are sufficient for sklearn's GIL-releasing ops and simpler to debug
- No Bayesian hyperparameter search libraries (Optuna, Hyperopt) in v1 — fixed grids only, keep scope tight
- No MLflow / experiment-tracking infra in v1 — log to state/JSON, revisit later if needed
- No local LLM hosting (Ollama, etc.) — no GPU access; Groq only, but keep provider abstracted so swapping later is a config change, not a rewrite

## 2. Provider Abstraction (LLM)
All LLM calls go through `groq_client.py`. Agents never import the Groq SDK directly. This mirrors the CRCM project's abstraction pattern — if we swap providers later, only one file changes.

## 3. What the LLM Reasons About vs. What Stays Rule-Based
To avoid the system becoming an unpredictable black box (the exact thing we're building against):
- **Rule-based / deterministic:** task-type detection thresholds, preprocessing transformer selection, model zoo composition, metric selection, ranking logic.
- **LLM-assisted:** report narration, explaining *why* a decision was made in natural language, flagging ambiguous edge cases for the human checkpoint.
- The LLM should never be the sole decision-maker for something that affects correctness (e.g., don't let the LLM "decide" the task type from a text description alone — it can narrate the reasoning, but the classification logic must be rule-based and reproducible).

## 4. Error Handling Policy
| Failure | Behavior |
|---|---|
| File fails to load / corrupt CSV | Hard fail. Return to user immediately with the parser error. Do not attempt to guess or repair. |
| Column can't be preprocessed (e.g., unencodable free text) | Drop column, log it in `preprocessing_log`, continue. Recoverable. |
| Target column ambiguous or task-type confidence low | Route to human checkpoint. Never silently guess on a consequential decision. |
| A single experiment run throws an exception | Log the failure, skip that model/param combo, continue with the rest. Do not kill the whole orchestrator run. |
| All experiments fail | Hard fail with a diagnostic summary of why each failed. |
| LLM call (Groq) fails or times out | Retry once with backoff; on second failure, fall back to a template-based (non-LLM) report/reasoning string rather than blocking the pipeline. |

Every error, recoverable or not, gets appended to `state["errors"]` with `{agent, error_type, message, recoverable}` — nothing fails silently.

## 5. Explainability Requirement
Every agent that makes a decision (Problem Detection, Preprocessing, Evaluation) must write a human-readable reasoning string into the state, not just the output value. This is non-negotiable — it's the core differentiator from black-box AutoML per the PRD.

## 6. Human-in-the-Loop
Do not remove or bypass the three defined checkpoints (Problem Detection confirmation, Experiment scope, Best model confirmation) without explicit sign-off — they exist because CRCM's design pattern established this project values transparency over full autonomy.

## 7. Scope Discipline (v1)
- CSV files only. Do not add support for other formats "while we're in there" — that's a v2 decision.
- Fixed model zoo, fixed hyperparameter grids. Do not reach for search libraries or auto-scaling infra to "improve" results in v1.
- No deployment/serving code. This produces a report, not a deployed model endpoint.

## 8. Testing Expectations
- Every agent needs at least one unit test covering its core happy path and one covering a failure/edge case (e.g., Preprocessing Agent tested against an all-missing column; Problem Detection tested against an ambiguous numeric target).
- No agent is considered "done" until it has a test file in `tests/`.

## 9. Code Style
- Type-annotate all agent function signatures (state in, state out).
- No agent function should mutate state in place silently — return the updated fields explicitly so LangGraph's state merge behavior stays predictable.
- Keep agent files single-responsibility — if an agent file exceeds ~300 lines, it's doing too much and should be split (e.g., model zoo logic out of the Experiment Orchestrator file).
