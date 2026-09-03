"""
Agent 3: Problem Detection Agent

Infers the machine learning task type (classification, regression, clustering) and target column.
If detection confidence is low, it triggers a human-in-the-loop checkpoint via LangGraph's interrupt().
Matches Architecture.md §2, §6 and Phases.md Phase 3 specifications.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import pandas as pd
from langgraph.types import interrupt

from src.orchestrator.state import AgentMLState, ErrorEntry
from src.llm import groq_client

logger = logging.getLogger(__name__)

# Common target-column names used by the auto-detect heuristic (only when no
# explicit column or instruction is given).
_TARGET_NAME_MATCHES = [
    "target", "label", "class", "y", "output", "response",
    "outcome", "result", "prediction", "survived", "churn", "default",
]


def _nearest_column(name: Any, columns: list[str]) -> Optional[str]:
    """Resolve a (possibly imperfect) column name from the LLM to a real column."""
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None
    if name in columns:
        return name
    low = name.lower()
    for c in columns:
        if c.lower() == low:
            return c
    for c in columns:
        if low in c.lower() or c.lower() in low:
            return c
    return None


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply (tolerates code fences/text)."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Find the first balanced {...} block
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def infer_target_from_instruction(
    df: pd.DataFrame,
    instruction: str,
    cached: Optional[dict] = None,
) -> Tuple[Optional[str], Optional[str], str, List[ErrorEntry]]:
    """
    Map a user's natural-language instruction to a target column + task type,
    grounded in the dataset's actual column names.

    Returns (target_col, task_type, reason, errors):
      - target_col None, task_type 'clustering' → user asked for unsupervised.
      - Falls back to the heuristic naming that no target was confidently found.
    """
    errors: List[ErrorEntry] = []
    reason = ""

    if cached and cached.get("target_column") is not None:
        tcol = _nearest_column(cached.get("target_column"), list(df.columns))
        if tcol:
            task = cached.get("task_type")
            reason = (
                f"User instruction parsed (cached): target column "
                f"'{tcol}' ({task})."
            )
            return tcol, task, reason, errors

    cols_meta = ", ".join(f"{c}({df[c].dtype})" for c in df.columns[:40])
    prompt = (
        "A user uploaded a dataset and below describes what they want the model to "
        "predict. Map the request to EXACTLY ONE of the dataset's columns as the "
        "prediction target, and choose its task type.\n"
        f"Dataset columns (name(dtype)): {cols_meta}\n"
        "Rules:\n"
        "- 'target_column' must be one of the exact column names above; use null if the "
        "  user wants clustering / no specific prediction.\n"
        "- 'task_type' must be exactly one of: classification, regression, clustering.\n"
        "- If the description clearly implies a column (e.g. 'survived', 'price', "
        "  'churn'), pick the closest matching column name.\n"
        "Reply with ONLY a JSON object, no prose, like:\n"
        '{"target_column": "<name|null>", "task_type": "<classification|regression|clustering>"}'
        f"\n\nUser instruction: {instruction}"
    )
    system = (
        "You resolve ML prediction targets from a user's natural-language description. "
        "Return strictly valid JSON with keys target_column and task_type."
    )

    try:
        text, call_errors = groq_client.chat(prompt=prompt, system=system)
        errors.extend(call_errors)
        data = _extract_json(text)
        if not data:
            reason = "Could not parse LLM target response; falling back to auto-detect."
            return None, None, reason, errors

        task = data.get("task_type")
        if isinstance(task, str):
            task = task.strip().lower()
        if task not in ("classification", "regression", "clustering"):
            task = None

        raw_col = data.get("target_column")
        tcol = _nearest_column(raw_col, list(df.columns))

        # Explicit unsupervised request: user asked for clustering / no target.
        if task == "clustering":
            reason = "User instruction parsed: no clear target → clustering."
            return None, "clustering", reason, errors
        # Supervised task named, but target column couldn't be resolved to a real
        # column → tell the caller to fall back to heuristic auto-detection.
        if tcol is None:
            reason = "Could not resolve the instruction's target to a dataset column; falling back to auto-detect."
            return None, None, reason, errors
        if task:
            reason = (
                f"User instruction parsed via LLM: predict '{tcol}' "
                f"({task}). Confidence grounded in the dataset columns."
            )
            return tcol, task, reason, errors

        # Task type unspecified → let rule-based classification decide later.
        reason = f"User instruction parsed via LLM: predict '{tcol}'."
        return tcol, None, reason, errors

    except Exception as exc:  # noqa: BLE001
        logger.warning("[problem_detection] instruction parsing failed: %s", exc)
        errors.append(ErrorEntry(
            phase="problem_detection",
            error_type="instruction_parse_error",
            message=f"Could not resolve prediction from instruction: {exc}",
            recoverable=True,
        ))
        return None, None, "Instruction parsing failed; falling back to auto-detect.", errors


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

    # 1. Resolve the prediction target.
    # Precedence:
    #   1. Natural-language user instruction (chat bar) → LLM-parsed target/task.
    #   2. Explicitly provided target_column in state.
    #   3. Auto-detect heuristic (name match → last column).
    #   4. Clustering (no target).
    # A user who provides NO instruction and NO explicit target gets auto-detection
    # (previously a blank/None target silently forced clustering — bug fixed).
    user_instruction = (state.get("user_instruction") or "").strip()
    instruction_target = None
    instruction_task = None
    instruction_reason_extra = ""
    is_clustering_request = False

    if user_instruction:
        instruction_target, instruction_task, instruction_reason_extra, instr_errors = (
            infer_target_from_instruction(df, user_instruction, state.get("instruction_parsed"))
        )
        errors_to_report.extend(instr_errors)
        # Only an explicit 'clustering' resolution means unsupervised. A None task
        # means the helper couldn't resolve a target → we fall back to heuristic.
        if instruction_task == "clustering":
            is_clustering_request = True

    is_target_provided = (target_col is not None)

    if is_clustering_request:
        # Explicit "no target" via the instruction → unsupervised.
        target_col = None
    elif instruction_target is not None:
        target_col = instruction_target
    elif is_target_provided and target_col in df.columns:
        # Explicit column given — already resolved.
        pass
    else:
        # Auto-detect: name match, then last column.
        detected_col = None
        for col in df.columns:
            if col.lower() in _TARGET_NAME_MATCHES:
                detected_col = col
                break
        if not detected_col and len(df.columns) > 0:
            detected_col = df.columns[-1]
            logger.info(f"Heuristically detected target column: {detected_col}")
        target_col = detected_col

    logger.info(f"[problem_detection] resolved target column: {target_col} | "
                f"instruction_target={instruction_target}")

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

    # 2b. Prefer an LLM-suggested task type when the user provided an instruction.
    # The explicit user hint (e.g. "predict who survived" → classification) is treated
    # as authoritative over the statistical rule above, but only for a real target.
    if (not is_clustering_request and instruction_task and
            instruction_task in ("classification", "regression") and
            target_col and target_col in df.columns):
        task_type = instruction_task
        confidence = max(confidence, 0.8)
        logger.info(f"[problem_detection] using instruction task type: {task_type}")

    # 3. Formulate Reasoning Template
    if instruction_reason_extra:
        reasoning_template = instruction_reason_extra + " "
    else:
        reasoning_template = ""
    reasoning_template += f"Detected task '{task_type}' for target column '{target_col}' because "
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
