"""
Task-appropriate metric selection and ranking for the Model Evaluation Agent.

Per Rules.md §3, metric and ranking decisions are rule-based and reproducible.
The Experiment Orchestrator (Phase 4) already precomputes task-specific
metrics on every experiment record's `metrics` dict; this module
centralizes the "which metric matters most" mapping.

Hierarchy per task type:
  - Classification: primary = accuracy (higher better), tiebreaker = f1 (higher better)
  - Regression:     primary = r2      (higher better), tiebreaker = rmse (lower better)
  - Clustering:     primary = silhouette (higher better); NaN treated as worst

We also expose a small direction map (`METRIC_DIRECTION`) so the ranking
function stays generic — adding a new metric only requires a one-line
entry there.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Metric selection tables
# ---------------------------------------------------------------------------

#: Primary metric per task type — what the leaderboard ranks by.
PRIMARY_METRIC: Dict[str, str] = {
    "classification": "accuracy",
    "regression": "r2",
    "clustering": "silhouette",
}

#: Tiebreaker metric per task type (None = no tiebreaker).
TIEBREAKER_METRIC: Dict[str, Optional[str]] = {
    "classification": "f1",
    "regression": "rmse",   # NOTE: rmse is lower-is-better; ranking handles this.
    "clustering": "n_clusters",  # secondary signal: more clusters is not strictly better,
                                 # but we use it as a stable tiebreaker for ranking only.
}

#: Direction for every metric that may appear on a leaderboard.
#: Anything not in this map defaults to "higher_is_better".
METRIC_DIRECTION: Dict[str, str] = {
    "accuracy": "higher_is_better",
    "f1": "higher_is_better",
    "precision": "higher_is_better",
    "recall": "higher_is_better",
    "r2": "higher_is_better",
    "rmse": "lower_is_better",
    "mae": "lower_is_better",
    "silhouette": "higher_is_better",
    "n_clusters": "higher_is_better",   # for ranking only; not a quality claim
    "n_noise": "lower_is_better",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def direction_for(metric_name: str) -> str:
    """Return 'higher_is_better' or 'lower_is_better' for a metric."""
    return METRIC_DIRECTION.get(metric_name, "higher_is_better")


def metric_value(record: Dict[str, Any], metric_name: str) -> float:
    """
    Pull a metric out of an experiment record, coercing NaN/None to
    a sentinel that sorts to the END of the ranking regardless of
    direction.

    Returns:
      - the numeric value if present and finite
      - math.inf  if the metric is missing/None and direction is lower_is_better
                  (i.e. we treat absence as "infinitely bad", which sorts last)
      - -math.inf if the metric is missing/None and direction is higher_is_better

    NaN values are coerced the same way missing values are — they should
    never be considered "as good as" a real measurement.
    """
    metrics = record.get("metrics") or {}
    val = metrics.get(metric_name)
    if val is None:
        return math.inf if direction_for(metric_name) == "lower_is_better" else -math.inf
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return math.inf if direction_for(metric_name) == "lower_is_better" else -math.inf
    if math.isnan(fval):
        return math.inf if direction_for(metric_name) == "lower_is_better" else -math.inf
    return fval


def rank_results(
    experiment_results: List[Dict[str, Any]],
    task_type: str,
) -> List[Dict[str, Any]]:
    """
    Produce the leaderboard ranking for a task type.

    Drops any record where success=False (per Rules.md §4 — those are
    already isolated and must never be re-raised as a winner).

    Args:
        experiment_results: list of records from
            `state["experiment_results"]` produced by the experiment
            orchestrator.
        task_type: one of "classification", "regression", "clustering".

    Returns:
        A list of ranking entries, ordered best → worst, each shaped:
            {
                "rank":            int (1-based),
                "model_id":        str,
                "model_name":      str,
                "primary_metric":  str (metric name),
                "primary_value":   float,
                "tiebreaker_value": float | None,
                "metrics":         dict (verbatim from record),
                "runtime_seconds": float,
            }
    """
    if task_type not in PRIMARY_METRIC:
        raise ValueError(f"rank_results: unknown task_type={task_type!r}")

    primary = PRIMARY_METRIC[task_type]
    tiebreaker = TIEBREAKER_METRIC.get(task_type)
    primary_dir = direction_for(primary)
    tiebreaker_dir = direction_for(tiebreaker) if tiebreaker else None

    successful = [r for r in experiment_results if r.get("success") is True]

    def _sort_key(r: Dict[str, Any]):
        # Primary: sort ascending; we want "best" at the top, so we negate
        # lower_is_better metrics so ascending = best-first.
        pval = metric_value(r, primary)
        if primary_dir == "lower_is_better":
            pval_key = -pval
        else:
            pval_key = pval

        if tiebreaker is None or tiebreaker_dir is None:
            return (pval_key, r.get("model_name") or r.get("model_id") or "")

        tval = metric_value(r, tiebreaker)
        if tiebreaker_dir == "lower_is_better":
            tval_key = -tval
        else:
            tval_key = tval

        return (pval_key, tval_key, r.get("model_name") or r.get("model_id") or "")

    ordered = sorted(successful, key=_sort_key)

    ranking: List[Dict[str, Any]] = []
    for idx, r in enumerate(ordered, start=1):
        ranking.append({
            "rank": idx,
            "model_id": r.get("model_id") or r.get("model_name"),
            "model_name": r.get("model_name"),
            "primary_metric": primary,
            "primary_value": metric_value(r, primary),
            "tiebreaker_value": (
                metric_value(r, tiebreaker) if tiebreaker else None
            ),
            "metrics": r.get("metrics") or {},
            "runtime_seconds": r.get("runtime_seconds", 0.0),
        })
    return ranking
