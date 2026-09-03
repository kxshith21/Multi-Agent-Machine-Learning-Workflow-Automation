"""
Agent 6: Model Evaluation Agent (Phase 5)

Ranks experiment_results by task-appropriate metrics, selects a best model,
and offers the user a chance to override the pick.

Per Rules.md §3 — every metric and ranking decision is rule-based.
Per Rules.md §5 — every decision agent writes a human-readable reasoning
string into state.
Per Architecture.md §6 checkpoint 3 — always offers a human-in-the-loop
override of the best-model pick.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langgraph.types import interrupt

from src.metrics.task_metrics import (
    PRIMARY_METRIC,
    TIEBREAKER_METRIC,
    rank_results,
)
from src.orchestrator.state import AgentMLState, ErrorEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_top(ranking: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the top entry (rank 1) or None if the list is empty."""
    if not ranking:
        return None
    return ranking[0]


def _compose_reasoning(
    task_type: str,
    top: Optional[Dict[str, Any]],
    ranking_size: int,
    successful_size: int,
    failed_size: int,
) -> str:
    """
    Build a human-readable explanation of the best-model pick (Rules.md §5).

    Deliberately rule-based / template — no LLM call. The LLM narration
    layer in Phase 6 (Report Generation) can rewrite this in prettier prose
    if desired; here we just want a deterministic, reproducible string.
    """
    if top is None:
        if successful_size == 0:
            return (
                f"No successful experiments were available for the {task_type} "
                f"task, so no best model can be selected."
            )
        return "The ranking is empty."

    primary_name = top["primary_metric"]
    primary_value = top["primary_value"]
    metric_display = _format_metric(primary_name, primary_value)

    tiebreaker_name = TIEBREAKER_METRIC.get(task_type)
    tiebreaker_clause = ""
    if tiebreaker_name and top.get("tiebreaker_value") is not None:
        tiebreaker_clause = (
            f" with a {tiebreaker_name} tiebreaker of "
            f"{_format_metric(tiebreaker_name, top['tiebreaker_value'])}"
        )

    return (
        f"Selected '{top['model_name']}' as the best model for the {task_type} "
        f"task. It ranked first out of {ranking_size} successful experiment"
        f"{'s' if ranking_size != 1 else ''} on the primary metric "
        f"{primary_name} = {metric_display}{tiebreaker_clause}. "
        f"{failed_size} experiment{'s' if failed_size != 1 else ''} "
        f"failed and {'were' if failed_size != 1 else 'was'} excluded from ranking."
    )


def _format_metric(name: str, value: float) -> str:
    """Format a metric for the reasoning string."""
    import math
    if math.isnan(value) or math.isinf(value):
        return "N/A"
    if name in ("accuracy", "f1", "precision", "recall", "r2", "silhouette"):
        return f"{value:.4f}"
    if name in ("rmse", "mae"):
        return f"{value:.3f}"
    return f"{value}"


# ---------------------------------------------------------------------------
# Public agent function
# ---------------------------------------------------------------------------

def evaluation_agent(state: AgentMLState) -> Dict[str, Any]:
    """
    Rank experiment results, select the best model, and offer override.

    Args:
        state: AgentMLState with `experiment_results` (Phase 4 output),
            `task_type`, and `errors` already populated.

    Returns:
        Dict containing:
          - ranking: list of ranked entries (best first)
          - best_model_id: model_id of the top-ranked entry (after optional override)
          - errors: updated error list (no new entries on the happy path)
          - status: "completed" | "needs_human_input" | "failed"

    Raises:
        ValueError: only if the orchestrator state is so malformed that we
            can't proceed (e.g. missing task_type). The pipeline should
            never reach Phase 5 in that state.
    """
    task_type = state.get("task_type")
    experiment_results: List[Dict[str, Any]] = list(state.get("experiment_results", []) or [])
    errors_to_report: List[ErrorEntry] = list(state.get("errors", []))

    if task_type not in PRIMARY_METRIC:
        raise ValueError(
            f"evaluation_agent: cannot rank for unknown task_type={task_type!r}"
        )

    successful = [r for r in experiment_results if r.get("success") is True]
    failed = [r for r in experiment_results if r.get("success") is not True]

    ranking = rank_results(experiment_results, task_type)
    top = _select_top(ranking)

    logger.info(
        "Model Evaluation | task=%s | successful=%d | failed=%d | ranking_size=%d",
        task_type, len(successful), len(failed), len(ranking),
    )

    # ---- 0. Edge case: nothing succeeded -----------------------------------------
    # We still offer the checkpoint (user can manually pick), but the
    # default pick is the literal string "none" so it's distinguishable
    # from a real model_id.
    if top is None:
        reasoning = _compose_reasoning(
            task_type=task_type,
            top=None,
            ranking_size=0,
            successful_size=len(successful),
            failed_size=len(failed),
        )
        # Still emit a checkpoint so the user can manually pick a model.
        # If they decline, best_model_id stays as the sentinel "none".
        best_model_id = "none"
        try:
            override = interrupt({
                "message": (
                    f"No models succeeded for task '{task_type}'. "
                    "Override the best-model pick or pass {} to accept 'none'."
                ),
                "default_best_model_id": best_model_id,
                "ranking": ranking,
                "task_type": task_type,
            })
            if isinstance(override, dict) and override.get("best_model_id"):
                best_model_id = str(override["best_model_id"])
                reasoning = (
                    f"Human override selected '{best_model_id}' even though no "
                    f"experiment succeeded."
                )
        except RuntimeError as exc:
            if "outside of a runnable context" not in str(exc):
                raise

        return {
            "ranking": ranking,
            "best_model_id": best_model_id,
            "evaluation_reasoning": reasoning,
            "errors": errors_to_report,
            "status": "completed",
        }

    # ---- 1. Architecture.md §6 checkpoint 3 — always-offered override -------------
    # The agent auto-picks the top-ranked model. The user can override it.
    default_best_model_id = top["model_id"]
    try:
        override = interrupt({
            "message": (
                f"Model Evaluation: '{default_best_model_id}' ranked first for "
                f"task '{task_type}' "
                f"(primary metric {top['primary_metric']} = "
                f"{_format_metric(top['primary_metric'], top['primary_value'])}). "
                "Pass {best_model_id: '<some_other_model_id>'} to override, or {} to accept."
            ),
            "default_best_model_id": default_best_model_id,
            "ranking": ranking,
            "task_type": task_type,
        })
        chosen_best_model_id = default_best_model_id
        overridden = False
        if isinstance(override, dict) and override.get("best_model_id"):
            chosen_best_model_id = str(override["best_model_id"])
            if chosen_best_model_id != default_best_model_id:
                overridden = True
    except RuntimeError as exc:
        # Only swallow the "outside a runnable context" error that unit
        # tests produce — propagate real RuntimeErrors (see Phase 4 decision #10).
        if "outside of a runnable context" not in str(exc):
            raise
        chosen_best_model_id = default_best_model_id
        overridden = False

    # ---- 2. Compose reasoning, including override note ---------------------------
    base_reasoning = _compose_reasoning(
        task_type=task_type,
        top=top,
        ranking_size=len(ranking),
        successful_size=len(successful),
        failed_size=len(failed),
    )
    if overridden:
        final_reasoning = (
            f"{base_reasoning} A manual override replaced this pick: the user "
            f"selected '{chosen_best_model_id}' instead of the automatic choice."
        )
    else:
        final_reasoning = base_reasoning

    return {
        "ranking": ranking,
        "best_model_id": chosen_best_model_id,
        "evaluation_reasoning": final_reasoning,
        "errors": errors_to_report,
        "status": "completed",
    }
