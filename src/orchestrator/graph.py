"""
AgentML — LangGraph pipeline graph.

Wires the 7 agents into a linear StateGraph.
Uses MemorySaver checkpointer to support human-in-the-loop checkpoints via interrupt().
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from src.agents.profiling_agent import profiling_agent
from src.agents.preprocessing_agent import preprocessing_agent
from src.agents.problem_detection_agent import problem_detection_agent
from src.agents.experiment_orchestrator_agent import experiment_orchestrator_agent
from src.agents.evaluation_agent import evaluation_agent
from src.agents.report_agent import report_agent
from src.orchestrator.state import AgentMLState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentMLState) -> dict:
    """
    Entry node: initialises cross-cutting state fields.
    """
    session_id = state.get("session_id") or str(uuid.uuid4())
    logger.info("[orchestrator] session=%s | starting pipeline", session_id)

    return {
        "session_id": session_id,
        "current_phase": "orchestrator",
        "errors": state.get("errors") or [],
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Feature Engineering Selection checkpoint node (Phase 10)
# ---------------------------------------------------------------------------

def feature_engineering_node(state: AgentMLState) -> dict:
    """
    4th human checkpoint (ALWAYS offered, like the Experiment Scope checkpoint).

    Sits after Problem Detection and before Data Preprocessing. Reads the
    feature suggestions detected by the Profiling Agent and offers the user a
    chance to select which engineered features to apply (and to add custom
    formulas). Default is to apply none — the user explicitly opts in.

    When invoked directly (outside a compiled graph, e.g. a unit test),
    LangGraph's interrupt() raises a RuntimeError ("outside of a runnable
    context"); we fall back to applying nothing, mirroring the Experiment Scope
    node's pattern.
    """
    suggestions = state.get("feature_suggestions") or []
    default_selected = []  # apply none by default — user opts in explicitly

    try:
        override = interrupt({
            "message": (
                "Feature Engineering Selection. Choose which suggested features "
                "to create, and/or add custom formulas. Select none to skip. "
                "Custom formulas only allow + - * / and existing column names."
            ),
            "feature_suggestions": suggestions,
            "default_selected": default_selected,
        })
        override = override or {}
        selected = [s for s in (override.get("selected_features") or []) if isinstance(s, str)]
        custom = list(override.get("custom_features") or [])
    except RuntimeError as exc:
        if "outside of a runnable context" in str(exc):
            # Direct call (unit test) — apply none.
            selected = list(default_selected)
            custom = []
        else:
            raise

    return {
        "selected_features": selected,
        "custom_features": custom,
        "current_phase": "feature_engineering",
    }


# ---------------------------------------------------------------------------
# Thin wrapper nodes
# ---------------------------------------------------------------------------

def _wrap(agent_fn, phase_name: str):
    """
    Return a LangGraph node function that:
      1. Sets current_phase before calling the agent.
      2. Calls agent_fn(state).
      3. Merges current_phase into the returned diff.
    """
    def node(state: AgentMLState) -> dict:
        logger.info("[graph] entering phase: %s", phase_name)
        # Verify if there was a non-recoverable error before executing
        hard_errors = [e for e in state.get("errors", []) if not e.get("recoverable", True)]
        if hard_errors:
            logger.error("[graph] hard error already present, skipping node execution")
            return {"status": "failed"}

        diff = agent_fn(state)
        # Ensure we return a dictionary
        if not isinstance(diff, dict):
            diff = {}
        diff["current_phase"] = phase_name
        return diff
    node.__name__ = f"{phase_name}_node"
    return node


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the 7-node AgentML StateGraph.
    Configures a MemorySaver checkpointer for interrupt support.
    """
    builder = StateGraph(AgentMLState)

    # Register nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("dataset_profiling",      _wrap(profiling_agent,              "dataset_profiling"))
    builder.add_node("problem_detection",      _wrap(problem_detection_agent,      "problem_detection"))
    builder.add_node("feature_engineering",    feature_engineering_node)
    builder.add_node("data_preprocessing",     _wrap(preprocessing_agent,          "data_preprocessing"))
    builder.add_node("experiment_orchestrator",_wrap(experiment_orchestrator_agent,"experiment_orchestrator"))
    builder.add_node("model_evaluation",       _wrap(evaluation_agent,             "model_evaluation"))
    builder.add_node("report_generation",      _wrap(report_agent,                 "report_generation"))

    # Wire linear edges
    builder.add_edge(START,                    "orchestrator")
    builder.add_edge("orchestrator",           "dataset_profiling")
    builder.add_edge("dataset_profiling",      "problem_detection")
    builder.add_edge("problem_detection",      "feature_engineering")
    builder.add_edge("feature_engineering",    "data_preprocessing")
    builder.add_edge("data_preprocessing",     "experiment_orchestrator")
    builder.add_edge("experiment_orchestrator","model_evaluation")
    builder.add_edge("model_evaluation",       "report_generation")
    builder.add_edge("report_generation",      END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_file_path: str,
    target_column: Optional[str] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> AgentMLState:
    """
    Build the graph and run it end-to-end.
    """
    session_id = (extra_state or {}).get("session_id") or str(uuid.uuid4())
    
    initial_state: Dict[str, Any] = {
        "session_id": session_id,
        "raw_file_path": raw_file_path,
        "target_column": target_column,
        "errors": [],
        "status": "running",
    }
    if extra_state:
        initial_state.update(extra_state)

    graph = build_graph()
    logger.info("AgentML pipeline starting | csv=%s | target=%s | session=%s", raw_file_path, target_column, session_id)

    config = {"configurable": {"thread_id": session_id}}
    
    # We invoke the graph. If it hits an interrupt(), it returns the state at the interrupt.
    final_state: AgentMLState = graph.invoke(initial_state, config)

    # Auto-accept every human-in-the-loop checkpoint with its default value so
    # the convenience runner completes end-to-end. (The Streamlit UI drives these
    # interactively; here we just keep moving.)
    for _ in range(15):
        if "__interrupt__" not in final_state:
            break
        payload = final_state["__interrupt__"][-1].value
        if "detected_task_type" in payload:
            # Phase 3 detection checkpoint — MUST resume with a non-empty payload
            # (resuming with {} re-triggers the interrupt indefinitely).
            resume_value = {
                "target_column": payload.get("detected_target_column"),
                "task_type": payload.get("detected_task_type"),
            }
        elif "default_scope" in payload:
            scope = payload["default_scope"]
            resume_value = {
                "max_experiments": scope.get("max_experiments", 6),
                "max_workers": scope.get("max_workers", 4),
                "time_cap_seconds": 0,
            }
        elif "default_best_model_id" in payload:
            resume_value = {"best_model_id": payload["default_best_model_id"]}
        elif "feature_suggestions" in payload:
            # Phase 10 feature-engineering checkpoint — always offered. The CLI
            # auto-runner applies NONE by default (user opts in via the UI).
            resume_value = {
                "selected_features": [],
                "custom_features": [],
            }
        else:
            resume_value = {}
        final_state = graph.invoke(Command(resume=resume_value), config)

    # Check for hard-fail errors
    hard_errors = [e for e in final_state.get("errors", []) if not e.get("recoverable", True)]
    if hard_errors:
        raise ValueError(
            f"Pipeline terminated with {len(hard_errors)} non-recoverable error(s): "
            + "; ".join(e["message"] for e in hard_errors)
        )

    logger.info(
        "AgentML pipeline complete or paused | session=%s | phase=%s",
        final_state.get("session_id"),
        final_state.get("current_phase"),
    )
    return final_state


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    csv = sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv"
    target = sys.argv[2] if len(sys.argv) > 2 else None

    result = run_pipeline(raw_file_path=csv, target_column=target)
    print("\n--- FINAL STATE SUMMARY ---")
    print(f"Session ID    : {result.get('session_id')}")
    print(f"Current Phase : {result.get('current_phase')}")
    print(f"Errors        : {len(result.get('errors', []))}")
    print(f"Report path   : {result.get('report_path')}")
