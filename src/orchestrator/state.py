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
