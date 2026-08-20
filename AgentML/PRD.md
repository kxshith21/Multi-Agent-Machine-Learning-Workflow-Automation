# PRD.md — AgentML: Multi-Agent ML Experiment Orchestrator

## 1. Problem Statement
Machine learning experimentation (dataset analysis, preprocessing, model selection, training, evaluation, reporting) is manual, expertise-heavy, and hard to reproduce. Existing AutoML tools are black boxes with no visibility into *why* decisions were made. There's no collaboration between individual decision-making steps — you get an output, not a reasoning trail.

## 2. Solution
AgentML is a multi-agent system where each stage of the ML lifecycle is owned by a specialized, autonomous agent, coordinated by a central Orchestrator. Every decision (problem type detected, preprocessing applied, model chosen) is logged and explainable, and the user can intervene at key checkpoints instead of accepting a black-box result.

## 3. Scope (v1)
- **Input format:** CSV files only (tabular data). No images, text corpora, or streaming data in v1.
- **Tasks supported:** Classification, Regression, Clustering (auto-detected).
- **Not in scope for v1:** Deep learning models, distributed/multi-node training, non-tabular formats, multi-file joins, time-series-specific handling.

## 4. Target Users
- **Primary:** Students / early-career data scientists who want to run a fast, explainable baseline experiment on a new dataset without hand-writing the full pipeline.
- **Secondary:** Instructors/evaluators (e.g., academic assessment context) who want to see a transparent, auditable decision trail — not just a final accuracy number.
- **Tertiary:** Small teams prototyping before committing to a production ML pipeline.

## 5. Goals
1. Take a raw CSV to a ranked, evaluated model with zero manual pipeline code.
2. Make every agent decision explainable (profiling → preprocessing → task detection → model choice) via a reasoning log.
3. Support human-in-the-loop confirmation at points where automated decisions are ambiguous or consequential.
4. Produce a final report that documents the *why*, not just the *what*.

## 6. Non-Goals
- Not a replacement for a production MLOps pipeline (no deployment, monitoring, or drift detection in v1).
- Not trying to beat state-of-the-art AutoML benchmarks (H2O, AutoGluon) on raw performance — the differentiator is transparency and modularity, not leaderboard scores.

## 7. Core Features
| Feature | Description |
|---|---|
| Dataset Profiling | Structure, dtypes, missing values, duplicates, statistical summary |
| Auto Preprocessing | Missing value handling, dedup, encoding, scaling — fully logged |
| Problem Type Detection | Classification / Regression / Clustering, with confidence score + reasoning |
| Target Column Inference | Heuristic detection with human confirmation on low confidence |
| Experiment Orchestration | Runs a fixed model zoo concurrently, logs all runs (not just the winner) |
| Model Evaluation & Ranking | Task-appropriate metrics, ranked leaderboard |
| Human-in-the-Loop Checkpoints | Task/target confirmation, experiment scope confirmation, final model override |
| Explainable Report | Markdown (→ optional PDF) documenting the full decision trail |

## 8. Success Criteria (v1)
- Given any reasonably clean CSV with a clear target column, the system produces a completed report end-to-end without manual intervention (unless a checkpoint is triggered).
- Every agent's output is inspectable — a user can trace *why* a decision was made, not just see the result.
- Pipeline runs on a laptop with no GPU (Groq API for any LLM reasoning steps, sklearn/xgboost for modeling — no local heavy compute).

## 9. Open Questions
- Do we allow the user to manually override the detected task type before experiments run, or only after (via checkpoint)? → Recommend: before, as a confirm/edit checkpoint.
- Do we cap experiment runtime by default, or let it run to completion? → Recommend: default cap (e.g., 5 min or N experiments), user-adjustable at checkpoint.
