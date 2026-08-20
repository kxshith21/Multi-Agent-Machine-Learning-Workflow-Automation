"""
Agent 1: Dataset Profiling Agent

Performs structure, dtype, missing value, duplicates, and per-column statistical analysis.
Matches Architecture.md §2 and Phases.md Phase 1 specifications.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Tuple
import pandas as pd
import numpy as np

from src.orchestrator.state import AgentMLState, ErrorEntry

logger = logging.getLogger(__name__)


def _sanitize_value(val: Any) -> Any:
    """
    Helper to convert numpy types to standard Python types for JSON serializability.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, pd.Timedelta)):
        return str(val)
    return val


def profiling_agent(state: AgentMLState) -> dict[str, Any]:
    """
    Loads the CSV dataset, performs profiling, and returns the dataset profile.

    Args:
        state: The current AgentMLState.

    Returns:
        A dict containing the updated dataset_profile and errors.

    Raises:
        ValueError/Exception: If the CSV is corrupt or unreadable, per Rules.md §4.
    """
    raw_file_path = state.get("raw_file_path")
    if not raw_file_path:
        error_msg = "No raw_file_path provided in the state."
        logger.error(error_msg)
        error_entry: ErrorEntry = {
            "phase": "dataset_profiling",
            "error_type": "missing_file_path",
            "message": error_msg,
            "recoverable": False
        }
        # Hard fail immediately
        raise ValueError(error_msg)

    logger.info(f"Profiling agent starting for file: {raw_file_path}")

    # Load the CSV
    try:
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        error_msg = f"Failed to load CSV '{raw_file_path}': {str(e)}"
        logger.error(error_msg)
        # Note: Rules.md §4: Hard fail. Return to user immediately with the parser error.
        # We raise the exception directly to trigger a hard fail.
        # We can still construct the error entry to be appended if caught.
        error_entry: ErrorEntry = {
            "phase": "dataset_profiling",
            "error_type": "csv_load_error",
            "message": error_msg,
            "recoverable": False
        }
        # Before raising, we can ensure the error is logged.
        raise type(e)(error_msg) from e

    # Check for empty dataframe
    if df.empty:
        error_msg = f"The loaded dataset '{raw_file_path}' is empty (has 0 rows or columns)."
        logger.error(error_msg)
        error_entry: ErrorEntry = {
            "phase": "dataset_profiling",
            "error_type": "empty_dataset",
            "message": error_msg,
            "recoverable": False
        }
        raise ValueError(error_msg)

    # Basic shape
    num_rows, num_cols = df.shape
    duplicate_count = int(df.duplicated().sum())

    # Column typing
    dtypes: dict[str, str] = {}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    missing_pct: dict[str, float] = {}
    column_stats: dict[str, dict[str, Any]] = {}
    errors_to_report: list[ErrorEntry] = list(state.get("errors", []))

    for col in df.columns:
        # Missing percentage
        missing_count = df[col].isna().sum()
        pct = float((missing_count / num_rows) * 100.0)
        missing_pct[col] = round(pct, 2)

        # Recoverable error warning if column is >50% missing values
        if pct > 50.0:
            warn_msg = f"Column '{col}' has high missing values ({pct:.2f}%)."
            logger.warning(warn_msg)
            errors_to_report.append({
                "phase": "dataset_profiling",
                "error_type": "high_missing_values",
                "message": warn_msg,
                "recoverable": True
            })

        # Identify numeric vs categorical
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
            dtypes[str(col)] = str(df[col].dtype)
            
            # Numeric stats
            desc = df[col].describe()
            column_stats[str(col)] = {
                "mean": _sanitize_value(desc.get("mean")),
                "std": _sanitize_value(desc.get("std")),
                "min": _sanitize_value(desc.get("min")),
                "max": _sanitize_value(desc.get("max")),
                "median": _sanitize_value(df[col].median()),
                "25%": _sanitize_value(desc.get("25%")),
                "75%": _sanitize_value(desc.get("75%")),
                "unique_count": int(df[col].nunique())
            }
        else:
            categorical_cols.append(str(col))
            dtypes[str(col)] = str(df[col].dtype)
            
            # Categorical stats
            desc = df[col].describe()
            column_stats[str(col)] = {
                "unique_count": int(df[col].nunique()),
                "top_value": _sanitize_value(desc.get("top")),
                "top_freq": _sanitize_value(desc.get("freq"))
            }

    # Assemble dataset profile
    profile = {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "duplicate_count": duplicate_count,
        "dtypes": dtypes,
        "missing_pct": missing_pct,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "column_stats": column_stats
    }

    logger.info(f"Dataset profiling completed successfully. Profile summary: {num_rows} rows, {num_cols} columns.")

    return {
        "dataset_profile": profile,
        "errors": errors_to_report
    }
