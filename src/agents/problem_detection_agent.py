"""
Agent 3: Problem Detection Agent

Infers the machine learning task type (classification, regression, clustering) and target column.
If detection confidence is low, it triggers a human-in-the-loop checkpoint via LangGraph's interrupt().
Matches Architecture.md §2, §6 and Phases.md Phase 3 specifications.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal
import pandas as pd
from langgraph.types import interrupt

from src.orchestrator.state import AgentMLState, ErrorEntry
from src.llm import groq_client

logger = logging.getLogger(__name__)


def problem_detection_agent(state: AgentMLState) -> dict[str, Any]:
    """
    Detects the target column and task type. Pauses for human confirmation if confidence is low.

    Args:
        state: The current AgentMLState.

    Returns:
        A dict containing the updated task_type, target_column, detection_confidence,
        detection_reasoning, and errors.
    """
    raw_file_path = state.get("raw_file_path")
    target_col = state.get("target_column")
    errors_to_report = list(state.get("errors", []))

    if not raw_file_path:
        error_msg = "No raw_file_path provided for problem detection."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Load dataset to inspect target
    try:
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        error_msg = f"Problem detection failed to read CSV: {str(e)}"
        logger.error(error_msg)
        raise type(e)(error_msg) from e

    # 1. Target Column Heuristic Detection
    # If target column is already in the state and exists in df.columns, we keep it.
    # If the user explicitly sets target_column to None, we skip target detection (unsupervised/clustering).
    # If target_column is not set in the state at all (or is empty), we perform heuristic detection.
    is_target_provided = ("target_column" in state) and (state["target_column"] is not None)

    if not is_target_provided and state.get("target_column") != None:
        # Heuristic 1: name match
        matches = ["target", "label", "class", "y", "output", "response"]
        detected_col = None
        for col in df.columns:
            if col.lower() in matches:
                detected_col = col
                break
        
        # Heuristic 2: last column
        if not detected_col and len(df.columns) > 0:
            detected_col = df.columns[-1]
            
        target_col = detected_col
        logger.info(f"Heuristically detected target column: {target_col}")

    # 2. Rule-Based Task Type Classification
    task_type: Literal["classification", "regression", "clustering"] = "clustering"
    confidence = 1.0
    nunique = 0

    if target_col is None or target_col not in df.columns:
        task_type = "clustering"
        confidence = 1.0
        logger.info("No target column detected or target not in columns. Set task type to clustering.")
    else:
        nunique = df[target_col].nunique()
        dtype = df[target_col].dtype

        if pd.api.types.is_categorical_dtype(df[target_col]) or pd.api.types.is_object_dtype(df[target_col]) or isinstance(dtype, pd.CategoricalDtype) or dtype == bool:
            task_type = "classification"
            confidence = 1.0
        elif pd.api.types.is_numeric_dtype(df[target_col]):
            if pd.api.types.is_float_dtype(df[target_col]):
                # If float, check if they are all integer-like (e.g. 1.0, 2.0) and unique count is small
                is_all_int = all(float(x).is_integer() for x in df[target_col].dropna()) if len(df[target_col].dropna()) > 0 else False
                if nunique <= 10 and is_all_int:
                    task_type = "classification"
                    confidence = 0.5
                else:
                    task_type = "regression"
                    confidence = 1.0
            else:
                # Integer target column
                if nunique <= 2:
                    task_type = "classification"
                    confidence = 1.0
                elif 2 < nunique <= 10:
                    task_type = "classification"
                    confidence = 0.5
                else:
                    task_type = "regression"
                    confidence = 0.95

    # 3. Formulate Reasoning Template
    reasoning_template = f"Detected task '{task_type}' for target column '{target_col}' because "
    if task_type == "clustering":
        reasoning_template += "no target column was provided, which is typical for unsupervised clustering tasks."
    elif task_type == "classification":
        reasoning_template += f"the target column has categorical or low cardinality numeric values ({nunique} unique values)."
    else:
        reasoning_template += f"the target column has numeric values with high cardinality ({nunique} unique values), typical for continuous regression tasks."

    # Groq LLM narration (optional narration / explanation)
    detection_reasoning = reasoning_template
    if target_col and target_col in df.columns:
        prompt = (
            f"We detected the machine learning task type as '{task_type}' and the target column as '{target_col}'. "
            f"The target column has data type '{df[target_col].dtype}' and {nunique} unique values. "
            f"Please narrate this decision in a professional, concise tone. Explain why this task type makes sense based on these data properties."
        )
        try:
            narration, groq_errors = groq_client.chat(
                prompt=prompt,
                system="You are a concise, plain-spoken ML experiment narrator. Explain decisions in 2-4 sentences. No hedging language."
            )
            if groq_errors:
                errors_to_report.extend(groq_errors)
            if narration:
                detection_reasoning = narration
        except Exception as e:
            logger.warning(f"Groq call failed, falling back to template: {str(e)}")

    # 4. Human Checkpoint Interruption (fires when confidence is below 0.7)
    if confidence < 0.7:
        logger.info(f"Ambiguous task detection (confidence={confidence}). Raising interrupt for human confirmation.")
        
        # Trigger the interrupt
        override = interrupt({
            "message": f"Task type detection is ambiguous (confidence={confidence}). Please confirm/override target column and task type.",
            "detected_target_column": target_col,
            "detected_task_type": task_type,
            "confidence": confidence
        })
        
        # Resumed: override contains the payload passed during resume
        logger.info(f"Resumed from checkpoint. Human override received: {override}")
        
        confirmed_target = override.get("target_column", target_col)
        confirmed_task = override.get("task_type", task_type)
        
        target_col = confirmed_target
        task_type = confirmed_task
        confidence = 1.0
        detection_reasoning = f"Human override confirmed. Target column: '{target_col}', Task type: '{task_type}'."
        logger.info(f"Updated task: {task_type}, target: {target_col}")

    return {
        "task_type": task_type,
        "target_column": target_col,
        "detection_confidence": confidence,
        "detection_reasoning": detection_reasoning,
        "errors": errors_to_report
    }
