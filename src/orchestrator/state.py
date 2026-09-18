"""
AgentMLState — the single shared state object passed through every node
in the LangGraph pipeline.

Matches Architecture.md §3 exactly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict


class ErrorEntry(TypedDict):
    """One entry in state["errors"]."""
    phase: str          # which agent/phase raised the error
    error_type: str     # e.g. "csv_parse_error", "groq_timeout", "model_fit_error"
    message: str        # human-readable description
    recoverable: bool   # True → log-and-continue; False → hard fail


class AgentMLState(TypedDict, total=False):
    """
    Shared state for the AgentML 7-agent LangGraph pipeline.
    Matches Architecture.md §3 exactly.
    """
    raw_file_path: str
    """Absolute or relative path to the uploaded CSV file."""

    dataset_profile: dict
    """Structure/dtype/missing/duplicate analysis computed by Profiling."""

    clean_dataset_path: str
    """Path to the preprocessed dataset file."""

    preprocessing_log: list[dict]
    """Log of all preprocessing operations applied and why."""

    task_type: Literal["classification", "regression", "clustering"]
    """Rule-based detected task type."""

    target_column: str | None
    """Prediction target column name (None for unsupervised/clustering)."""

    detection_confidence: float
    """Confidence score of the problem detection."""

    detection_reasoning: str
    """Human-readable explanation of why task_type was detected."""

    class_balance: dict
    """Normalized class distribution of the target column for classification
    tasks, e.g. {"0": 0.95, "1": 0.05} (keys are str(class label)). Present
    only for classification; used to surface imbalance before training."""

    experiment_results: list[dict]
    """Summary of model evaluation results."""

    best_model_id: str
    """Identifier of the top-performing model."""

    ranking: list[dict]
    """Leaderboard ranking of evaluated models."""

    report_path: str
    """Path to the generated report file."""

    errors: list[ErrorEntry]
    """Accumulated error log."""

    status: Literal["running", "needs_human_input", "completed", "failed"]
    """Orchestrator pipeline status."""

    # Helpful orchestration fields (can be present under total=False)
    session_id: str
    """Unique ID for this pipeline run (UUID4 string)."""

    current_phase: str
    """Name of the currently executing phase."""

    evaluation_reasoning: str
    """Human-readable explanation of why the best model was picked (Phase 5)."""

    user_instruction: str
    """Natural-language instruction from the chat bar describing what to predict.

    Optional. When present, Problem Detection resolves it (via LLM grounded in the
    actual CSV columns) into target_column + task_type. Blank/absent → auto-detect."""

    instruction_parsed: dict
    """Cached LLM parse of user_instruction: {"target_column": ..., "task_type": ...}.

    Stored so re-runs/resumes don't re-call the LLM for the same instruction."""

    # ------------------------------------------------------------------
    # Feature Engineering Selection (4th human checkpoint, Phase 10)
    # ------------------------------------------------------------------
    feature_suggestions: list[dict]
    """Plain-language candidate feature ideas detected by the Profiling Agent.
    Each dict: {"name", "type", "description", "columns"}. Populated by Profiling,
    never applied by it (detect & suggest only)."""

    selected_features: list[str]
    """Names of suggested features the user opted in to at the checkpoint.
    Populated via the human checkpoint. These are created deterministically by
    Data Preprocessing after the checkpoint resolves."""

    custom_features: list[dict]
    """User-defined columns, e.g. [{"name": "price_per_sqft", "formula": "price / sqft"}].
    Formulas are always evaluated with the restricted safe_formula parser (never eval).
    Populated via the human checkpoint."""

    feature_engineering_log: list[dict]
    """Log of every engineered column actually created and why (source: suggested
    vs. custom, plus the formula/recipe). Same logging pattern as preprocessing_log."""
