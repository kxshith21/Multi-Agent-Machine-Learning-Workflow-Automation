"""
Agent 4: Experiment Orchestrator Agent (Phase 4)

Runs the fixed model zoo for the detected task type, concurrently, with
fixed hyperparameter grids (no Optuna/Hyperopt — Rules.md §1, §7).
Logs every experiment to state["experiment_results"] and obeys
Rules.md §4's error policy exactly:

  - Single experiment throws an exception → log, skip, continue.
  - ALL experiments fail → hard fail with diagnostic summary.

Architecture.md §6 checkpoint 2 is offered (always, not forced): the user
can confirm/adjust scope (max experiment count, time cap) before the zoo
runs.

Concurrency: concurrent.futures.ThreadPoolExecutor — not asyncio (Rules.md §1).
"""

from __future__ import annotations

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langgraph.types import interrupt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from src.model_zoo import ModelSpec
from src.model_zoo.classification_models import get_models as get_classification_models
from src.model_zoo.regression_models import get_models as get_regression_models
from src.model_zoo.clustering_models import get_models as get_clustering_models
from src.orchestrator.state import AgentMLState, ErrorEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_zoo(task_type: str) -> List[ModelSpec]:
    """Return the fixed model zoo for the detected task type."""
    if task_type == "classification":
        return get_classification_models()
    if task_type == "regression":
        return get_regression_models()
    if task_type == "clustering":
        return get_clustering_models()
    raise ValueError(f"Unknown task_type for experiment orchestration: {task_type!r}")


def _split_features_target(
    df: pd.DataFrame, target_column: Optional[str]
) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    """Return (X, y) from the cleaned dataframe."""
    if target_column is None or target_column not in df.columns:
        return df, None
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def _compute_metrics(
    task_type: str,
    y_true: Any,
    y_pred: Any,
    X_for_silhouette: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute task-appropriate metrics.

    Classification: accuracy, f1 (weighted), precision, recall.
    Regression:     RMSE, MAE, R².
    Clustering:     silhouette (only if ≥2 clusters were formed and X given),
                    n_clusters, n_noise_points.
    """
    metrics: Dict[str, float] = {}
    try:
        if task_type == "classification":
            labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
            # Some metrics require at least 2 unique labels in y_true
            average = "weighted" if len(labels) > 2 else "binary"
            pos_label = labels[0] if len(labels) == 2 else 1
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            try:
                metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
            except Exception:
                metrics["f1"] = float("nan")
            try:
                metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
            except Exception:
                metrics["precision"] = float("nan")
            try:
                metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            except Exception:
                metrics["recall"] = float("nan")

        elif task_type == "regression":
            mse = mean_squared_error(y_true, y_pred)
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))

        elif task_type == "clustering":
            labels = np.asarray(y_pred)
            n_clusters = int(len(set(labels) - {-1}))
            metrics["n_clusters"] = float(n_clusters)
            metrics["n_noise"] = float(int((labels == -1).sum()))
            if X_for_silhouette is not None and n_clusters >= 2 and n_clusters < len(labels):
                try:
                    metrics["silhouette"] = float(silhouette_score(X_for_silhouette, labels))
                except Exception:
                    metrics["silhouette"] = float("nan")
            else:
                metrics["silhouette"] = float("nan")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Metric computation partially failed: %s", exc)
    return metrics


def _run_single_experiment(
    spec: ModelSpec,
    task_type: str,
    X: pd.DataFrame,
    y: Optional[pd.Series],
    random_state: int,
) -> Dict[str, Any]:
    """
    Run one experiment (model fit + predict + metric compute).

    Returns a serializable dict ready to be appended to
    state["experiment_results"]. ANY exception raised here is caught
    by the caller (the agent function) and converted into a failure
    record — never raised past this function.

    This function is the unit that runs in a worker thread.
    """
    started = time.perf_counter()
    run_id = f"{spec['name']}"
    record: Dict[str, Any] = {
        "model_id": run_id,
        "model_name": spec["name"],
        "family": spec.get("family", "unknown"),
        "params": dict(spec.get("param_grid") or {}),
        "task_type": task_type,
        "success": False,
        "error_type": None,
        "error_message": None,
        "metrics": {},
        "runtime_seconds": 0.0,
        "n_train_samples": 0,
        "n_test_samples": 0,
    }

    try:
        # Clone the estimator so each thread gets its own (sklearn estimators
        # are not always thread-safe to share; cloning is cheap and safe).
        from sklearn.base import clone
        estimator = clone(spec["estimator"])

        # Apply fixed param grid (v1: as-is, no cartesian expansion)
        if spec.get("param_grid"):
            estimator.set_params(**spec["param_grid"])

        if task_type == "clustering":
            # Unsupervised — fit on the whole feature matrix.
            X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
            estimator.fit(X_arr)
            y_pred = estimator.labels_ if hasattr(estimator, "labels_") else estimator.predict(X_arr)
            metrics = _compute_metrics(
                task_type, y_true=None, y_pred=y_pred, X_for_silhouette=X_arr
            )
            record["n_train_samples"] = int(X_arr.shape[0])
            record["n_test_samples"] = 0
        else:
            # Supervised — train/test split with stratification when possible.
            assert y is not None, "Supervised experiment requires y"
            stratify = y if task_type == "classification" and y.nunique() > 1 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=random_state, stratify=stratify
                )
            except ValueError:
                # Stratification can fail with very small class counts
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=random_state
                )

            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            metrics = _compute_metrics(task_type, y_true=y_test, y_pred=y_pred)
            record["n_train_samples"] = int(len(X_train))
            record["n_test_samples"] = int(len(X_test))

        record["metrics"] = metrics
        record["success"] = True

    except Exception as exc:  # noqa: BLE001
        record["success"] = False
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)[:500]
        # Verbose traceback lives only in the logger, not the state.
        logger.warning(
            "Experiment %s failed: %s\n%s", run_id, exc, traceback.format_exc()
        )

    finally:
        record["runtime_seconds"] = round(time.perf_counter() - started, 4)

    return record


# ---------------------------------------------------------------------------
# Public agent function
# ---------------------------------------------------------------------------

def experiment_orchestrator_agent(state: AgentMLState) -> Dict[str, Any]:
    """
    Run the model zoo concurrently and log every result.

    Args:
        state: AgentMLState with task_type, clean_dataset_path,
               target_column, and (optional) session_id already set.

    Returns:
        A dict containing:
          - experiment_results: list of per-experiment records (success and failure)
          - errors: updated error list (with appended recoverable errors for
                    individual experiment failures; possibly non-recoverable
                    if ALL experiments failed)
          - status: "running" | "completed" | "failed" | "needs_human_input"

    Raises:
        ValueError: hard-fail (Rules.md §4) — only if EVERY experiment
                    failed AND there was no checkpoint override. The
                    diagnostic summary is in state["errors"] before the raise.
    """
    task_type = state.get("task_type")
    clean_path = state.get("clean_dataset_path")
    target_column = state.get("target_column")
    session_id = state.get("session_id") or "default"
    errors_to_report: List[ErrorEntry] = list(state.get("errors", []))
    prior_results: List[Dict[str, Any]] = list(state.get("experiment_results", []) or [])

    # ---- 0. Pre-flight validation -------------------------------------------------
    if task_type not in ("classification", "regression", "clustering"):
        raise ValueError(f"experiment_orchestrator_agent: unknown task_type={task_type!r}")

    if not clean_path:
        raise ValueError("experiment_orchestrator_agent: clean_dataset_path is empty")

    try:
        zoo: List[ModelSpec] = _select_zoo(task_type)
    except ValueError:
        raise

    # ---- 1. Optional human checkpoint for scope (Architecture.md §6 #2) -----------
    # ALWAYS offered, not forced (Rules.md §6 + Architecture.md §6).
    default_scope = {
        "max_experiments": len(zoo),
        "time_cap_seconds": 0,            # 0 = no cap
        "max_workers": min(4, len(zoo)),  # thread pool size
    }
    # `interrupt()` raises a GraphInterrupt when invoked inside a compiled
    # LangGraph (this is what causes the graph to pause and surface
    # __interrupt__ to the caller). When invoked OUTSIDE a graph context
    # — e.g. by a direct unit-test call — it raises a plain RuntimeError
    # ("Called get_config outside of a runnable context"). We want to
    # ONLY swallow that specific fallback case; we must NOT catch the
    # real GraphInterrupt or the checkpoint will silently no-op.
    try:
        override = interrupt({
            "message": (
                "Experiment Orchestrator scope check. Confirm/adjust max "
                "experiments, per-run time cap (seconds), and worker count. "
                "Pass empty dict {} to accept defaults."
            ),
            "default_scope": default_scope,
            "task_type": task_type,
            "available_models": [m["name"] for m in zoo],
        })
        scope = {**default_scope, **(override or {})}
    except RuntimeError as exc:
        msg = str(exc)
        if "outside of a runnable context" in msg:
            # Called outside a compiled graph — fall back to defaults.
            scope = default_scope
        else:
            # Real RuntimeError — propagate so it's not silently swallowed.
            raise

    max_workers = max(1, int(scope.get("max_workers", default_scope["max_workers"])))
    max_experiments = int(scope.get("max_experiments", default_scope["max_experiments"]))
    time_cap = float(scope.get("time_cap_seconds", 0))

    zoo_to_run = zoo[:max_experiments]

    # ---- 2. Load clean dataset ----------------------------------------------------
    try:
        df = pd.read_csv(clean_path)
    except Exception as exc:
        msg = f"experiment_orchestrator_agent: failed to read clean dataset {clean_path}: {exc}"
        logger.error(msg)
        errors_to_report.append(ErrorEntry(
            phase="experiment_orchestrator",
            error_type="dataset_load_error",
            message=msg,
            recoverable=False,
        ))
        return {"experiment_results": prior_results, "errors": errors_to_report, "status": "failed"}

    X, y = _split_features_target(df, target_column)

    # Sanity: need at least 2 rows for any meaningful split/clustering
    if len(df) < 2:
        msg = f"experiment_orchestrator_agent: dataset has only {len(df)} rows — nothing to run."
        logger.error(msg)
        errors_to_report.append(ErrorEntry(
            phase="experiment_orchestrator",
            error_type="dataset_too_small",
            message=msg,
            recoverable=False,
        ))
        return {"experiment_results": prior_results, "errors": errors_to_report, "status": "failed"}

    # ---- 3. Run zoo concurrently with ThreadPoolExecutor -------------------------
    logger.info(
        "Experiment Orchestrator | task=%s | models=%d | workers=%d | session=%s",
        task_type, len(zoo_to_run), max_workers, session_id,
    )

    results: List[Dict[str, Any]] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for spec in zoo_to_run:
            fut = pool.submit(
                _run_single_experiment, spec, task_type, X, y, 42
            )
            futures[fut] = spec["name"]

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                record = fut.result(timeout=max(time_cap, 1.0) if time_cap > 0 else None)
            except Exception as exc:
                # The runner itself already swallows exceptions, but if a
                # worker somehow raises (e.g. a future-level error from the
                # executor), we still log-and-skip per Rules.md §4.
                logger.error("Future-level failure for %s: %s", name, exc)
                record = {
                    "model_id": name,
                    "model_name": name,
                    "family": "unknown",
                    "params": {},
                    "task_type": task_type,
                    "success": False,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "metrics": {},
                    "runtime_seconds": 0.0,
                    "n_train_samples": 0,
                    "n_test_samples": 0,
                }

            results.append(record)

            if not record["success"]:
                errors_to_report.append(ErrorEntry(
                    phase="experiment_orchestrator",
                    error_type="experiment_failure",
                    message=f"{name} failed: {record['error_type']}: {record['error_message']}",
                    recoverable=True,
                ))

    # ---- 4. Order results by model name for deterministic logs --------------------
    results.sort(key=lambda r: r["model_name"])

    # ---- 5. Hard-fail if EVERY experiment failed (Rules.md §4) --------------------
    successful = [r for r in results if r["success"]]
    if results and not successful:
        summary_lines = [
            f"- {r['model_name']}: {r['error_type']}: {r['error_message']}"
            for r in results
        ]
        summary = "All experiments failed. Diagnostic summary:\n" + "\n".join(summary_lines)
        errors_to_report.append(ErrorEntry(
            phase="experiment_orchestrator",
            error_type="all_experiments_failed",
            message=summary,
            recoverable=False,
        ))
        return {
            "experiment_results": prior_results + results,
            "errors": errors_to_report,
            "status": "failed",
        }

    # ---- 6. Append to state, return diff -----------------------------------------
    return {
        "experiment_results": prior_results + results,
        "errors": errors_to_report,
        "status": "completed",
    }
