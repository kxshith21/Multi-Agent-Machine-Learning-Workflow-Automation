"""
src.metrics — task-appropriate metric selection and ranking helpers.

Per Architecture.md §4 + Phases.md Phase 5:
- Metric selection is rule-based (Rules.md §3 — no LLM decisions on correctness).
- All metrics are precomputed by the experiment orchestrator (Phase 4)
  and re-used here; this module is the single source of truth for
  which metric is the "primary" one a task type is ranked by.

Public API:
    PRIMARY_METRIC       — dict: task_type -> primary metric name
    TIEBREAKER_METRIC    — dict: task_type -> tiebreaker metric name (or None)
    METRIC_DIRECTION      — dict: metric name -> "higher_is_better" or "lower_is_better"
    metric_value()       — helper that returns the metric value for a record (NaN -> -inf or +inf)
    rank_results()       — produce the ranking list given experiment_results + task_type
"""

from __future__ import annotations
