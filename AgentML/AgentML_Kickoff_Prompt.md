# AgentML — Build Kickoff Prompt

> Copy everything below this line into your vibe-coding tool (Antigravity, Cursor, etc.) as the first message for this project.

---

## Context

You are building **AgentML**, a multi-agent ML experiment orchestrator. Full specifications live in six documents that are attached/available in the project root — **read all of them fully before writing any code**:

- `PRD.md` — problem statement, scope, goals, success criteria
- `Architecture.md` — agent list, state schema, folder structure, high-level flow
- `Rules.md` — hard boundaries (libraries to use/avoid, error handling policy, explainability requirements, human-in-the-loop rules, scope discipline, testing/code-style expectations)
- `Phases.md` — the exact build order, phase-by-phase, with "done when" criteria for each
- `Design.md` — UI/UX spec for the Streamlit demo (colors, typography, layout, components, tone)
- `Memory.md` — a living status log template; you will update this file yourself at the end of every session

**These documents are the source of truth. Do not deviate from them.** If something you're asked to do conflicts with Rules.md, Architecture.md, or Phases.md, stop and flag the conflict instead of silently working around it.

## Your Task

Build AgentML strictly in the order defined in `Phases.md`, one phase at a time. Do not start Phase N+1 until Phase N's "Done when" criteria are met and passing. This sequencing is intentional — the biggest risk on this project is agent-boundary rework mid-build if phases are skipped or parallelized.

### Ground rules while you work

1. **Follow `Architecture.md` exactly** for folder structure, the `AgentMLState` schema, and the 7-agent flow (Orchestrator, Dataset Profiling, Data Preprocessing, Problem Detection, Experiment Orchestrator, Model Evaluation, Report Generation). Do not add, remove, merge, or rename agents.
2. **Follow `Rules.md` without exception**, especially:
   - Library allow-list (pandas, scikit-learn, xgboost, langgraph, `ThreadPoolExecutor`, Jinja2, streamlit) vs. the explicit avoid-list (no PyTorch/TF/Keras, no asyncio for concurrency, no Optuna/Hyperopt, no MLflow, no local LLM hosting).
   - All Groq calls go through `src/llm/groq_client.py` — never call the Groq SDK directly from an agent file.
   - Keep the LLM to narration/explanation only; task-type detection, preprocessing selection, model zoo composition, metric selection, and ranking must stay rule-based and reproducible.
   - Implement the error-handling table in §4 exactly (hard fail vs. log-and-continue vs. checkpoint, per failure type). Every error goes into `state["errors"]`.
   - Every decision-making agent must write a human-readable reasoning string into state — not optional.
   - Do not bypass or remove the three human checkpoints (Problem Detection confirmation, Experiment scope, Best model confirmation).
   - Keep agent files under ~300 lines; split if they grow past that.
   - Every agent needs a happy-path test and a failure/edge-case test in `tests/` before it's "done."
3. **Follow `Phases.md` "Done when" criteria literally** before moving on — these are acceptance criteria, not suggestions.
4. **UI work (Phase 7 onward)** should follow `Design.md` precisely: the color palette, Inter/JetBrains Mono type system, horizontal pipeline stepper, collapsible per-agent cards, amber-accented checkpoint cards, and the Markdown/PDF report pane. Copy tone should be plain and direct — no hedging language, per Design.md §6.
5. **Update `Memory.md` at the end of every session** — Current Phase, Last Updated, What's Done, What's In Progress, Known Issues/Blockers, Decisions Made Since the Design Docs (log any deviation with a one-line reason), Key File Locations, Next Steps. This is how future sessions/tools pick up context without re-reading the whole codebase.

### Start here

Begin with **Phase 0 — Foundation**:
- Set up the repo structure exactly as specified in `Architecture.md` §5.
- Define `AgentMLState` in `src/orchestrator/state.py`.
- Build the LangGraph skeleton in `src/orchestrator/graph.py` with 7 stub agents that just pass state through unchanged.
- Build the Groq client abstraction in `src/llm/groq_client.py` with a working "hello world" call.
- Add `.env.example` (with `GROQ_API_KEY` placeholder) and `requirements.txt`.
- Confirm a CSV can flow through all 7 stub agents end-to-end without errors, producing a placeholder report — that's Phase 0's done condition.

Once Phase 0 passes its done condition, stop, summarize what you built, update `Memory.md`, and wait for confirmation before starting Phase 1.

Do not skip ahead to later phases, do not add scope not listed in the docs (no new file formats, no deep learning, no hyperparameter search libraries, no deployment code — see Rules.md §7), and do not write UI code before Phase 7.
