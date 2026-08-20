# Memory.md — AgentML

> **Note:** This file is a template. Leave it mostly empty until coding actually starts, then update it after every meaningful session so a new chat/tool can pick up context without re-reading the whole codebase. Keep entries short and factual — this is a status log, not a design doc (that's what PRD/Architecture/Rules/Phases are for).

## Current Phase
`Phase 0 — Foundation` *(update as you progress through Phases.md)*

## Last Updated
`YYYY-MM-DD` *(update every session)*

## What's Done
- *(nothing yet — populate as phases complete, e.g.: "Phase 0: repo structure, state schema, LangGraph skeleton with stub agents all working end-to-end")*

## What's In Progress
- *(e.g.: "Phase 1: Dataset Profiling Agent — dtype detection done, missing-value stats in progress")*

## Known Issues / Blockers
- *(e.g.: "ydata-profiling is slow on >100k rows — may need to hand-roll summary stats instead")*

## Decisions Made Since the Design Docs
- *(log anything that deviates from or refines PRD/Architecture/Rules/Phases here, with a one-line reason — e.g.: "Switched target-column heuristic to check last 3 columns instead of just the last one, since sample datasets had trailing ID columns")*

## Key File Locations (update as structure evolves)
- State schema: `src/orchestrator/state.py`
- LangGraph definition: `src/orchestrator/graph.py`
- Agents: `src/agents/`
- Model zoo: `src/model_zoo/`

## Next Steps
- *(concrete next 1-3 actions — e.g.: "1) finish missing-value detection in Profiling Agent, 2) write its unit tests, 3) start Preprocessing Agent")*

---
**Usage pattern:** at the start of a new chat/session, paste this file (plus Rules.md and the current Phase from Phases.md) instead of the whole codebase. Update the "Last Updated," "What's Done," "In Progress," and "Next Steps" sections at the end of every session — this is the cheapest way to preserve context across tool switches.
