"""
Unit tests for the Experiment Orchestrator Agent
(src/agents/experiment_orchestrator_agent.py).

Phase 4 spec (Phases.md + Rules.md §4):

  - Run the full model zoo for each task type and log correct results.
  - A single deliberately-broken experiment is isolated — its failure
    is logged, the rest of the zoo continues, and the orchestrator
    still returns useful results.
  - When ALL experiments fail, the orchestrator hard-fails with a
    diagnostic summary in state["errors"].
  - Architecture.md §6 checkpoint 2 (scope) is offered — when the
    orchestrator is invoked inside a compiled LangGraph, interrupt()
    fires; resuming with an override applies it.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_classification, make_regression
from langgraph.types import Command

from src.agents.experiment_orchestrator_agent import (
    experiment_orchestrator_agent,
    _run_single_experiment,
)
from src.model_zoo.classification_models import get_models as get_class_models
from src.model_zoo.regression_models import get_models as get_reg_models
from src.model_zoo.clustering_models import get_models as get_clust_models
from src.model_zoo import ModelSpec
from src.orchestrator.state import AgentMLState
from src.orchestrator.graph import build_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_clean_csv(path: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def _make_classification_dataframe(n: int = 120, seed: int = 0) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n, n_features=6, n_informative=4, n_redundant=1,
        n_classes=2, random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def _make_regression_dataframe(n: int = 120, seed: int = 0) -> pd.DataFrame:
    X, y = make_regression(
        n_samples=n, n_features=6, n_informative=4, noise=0.5, random_state=seed
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["target"] = y
    return df


def _make_clustering_dataframe(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # 3 well-separated blobs so silhouette is meaningful
    centers = [(0, 0), (5, 5), (-5, 5)]
    parts = []
    for cx, cy in centers:
        parts.append(rng.normal(loc=(cx, cy), scale=0.5, size=(n // 3, 2)))
    arr = np.vstack(parts)
    df = pd.DataFrame(arr, columns=["feat_0", "feat_1"])
    return df


# A deliberately-broken estimator that ALWAYS raises during fit. Used to
# verify error-isolation per Rules.md §4.
class _BrokenClassifier(BaseEstimator, ClassifierMixin):
    """Estimator that throws a known exception on every fit()."""
    def fit(self, X, y=None):
        raise RuntimeError("deliberately broken estimator — fit() always fails")

    def predict(self, X):
        return np.zeros(len(X))


def _make_broken_spec(name: str = "BrokenClassifier_always_fails") -> ModelSpec:
    return ModelSpec(
        name=name,
        family="broken",
        estimator=_BrokenClassifier(),
        param_grid={},
    )


# ---------------------------------------------------------------------------
# 1. Full-zoo happy paths for each task type
# ---------------------------------------------------------------------------

def test_experiment_orchestrator_classification_zoo_runs(tmp_path):
    """The classification zoo runs against a real numeric dataset and logs results."""
    csv_path = str(tmp_path / "clean_class.csv")
    df = _make_classification_dataframe()
    _write_clean_csv(csv_path, df)

    state: AgentMLState = {
        "session_id": "test_cls",
        "task_type": "classification",
        "target_column": "target",
        "clean_dataset_path": csv_path,
        "errors": [],
    }

    result = experiment_orchestrator_agent(state)

    assert result["status"] == "completed"
    assert "experiment_results" in result
    results = result["experiment_results"]
    assert len(results) == len(get_class_models())
    # Every entry must have a model_name, success flag, runtime, metrics
    for r in results:
        assert "model_name" in r
        assert "success" in r
        assert "runtime_seconds" in r
        assert "metrics" in r
        if r["success"]:
            assert "accuracy" in r["metrics"]
            assert 0.0 <= r["metrics"]["accuracy"] <= 1.0
    # At least one model should beat the dummy baseline (it's the leaderboard's job)
    successful = [r for r in results if r["success"]]
    assert len(successful) >= 4, f"Too few classification models succeeded: {len(successful)}"


def test_experiment_orchestrator_regression_zoo_runs(tmp_path):
    """The regression zoo runs against a real numeric dataset and logs results."""
    csv_path = str(tmp_path / "clean_reg.csv")
    df = _make_regression_dataframe()
    _write_clean_csv(csv_path, df)

    state: AgentMLState = {
        "session_id": "test_reg",
        "task_type": "regression",
        "target_column": "target",
        "clean_dataset_path": csv_path,
        "errors": [],
    }

    result = experiment_orchestrator_agent(state)

    assert result["status"] == "completed"
    results = result["experiment_results"]
    assert len(results) == len(get_reg_models())
    successful = [r for r in results if r["success"]]
    assert len(successful) >= 5, f"Too few regression models succeeded: {len(successful)}"
    # Successful regressors should produce r2 in metrics
    r2_record = next(r for r in successful if "r2" in r["metrics"])
    assert -1.0 <= r2_record["metrics"]["r2"] <= 1.0


def test_experiment_orchestrator_clustering_zoo_runs(tmp_path):
    """The clustering zoo runs and logs results (no target column)."""
    csv_path = str(tmp_path / "clean_clust.csv")
    df = _make_clustering_dataframe()
    _write_clean_csv(csv_path, df)

    state: AgentMLState = {
        "session_id": "test_clust",
        "task_type": "clustering",
        "target_column": None,
        "clean_dataset_path": csv_path,
        "errors": [],
    }

    result = experiment_orchestrator_agent(state)

    assert result["status"] == "completed"
    results = result["experiment_results"]
    assert len(results) == len(get_clust_models())
    # At least KMeans + GaussianMixture + Agglomerative should succeed.
    successful = [r for r in results if r["success"]]
    assert len(successful) >= 3, f"Too few clustering models succeeded: {len(successful)}"


# ---------------------------------------------------------------------------
# 2. Failure isolation — Rules.md §4
# ---------------------------------------------------------------------------

def test_experiment_orchestrator_isolates_single_broken_model(monkeypatch, tmp_path):
    """
    Inject one broken estimator into the classification zoo. Verify:
      - The broken entry's failure is logged in experiment_results.
      - Other models still run and succeed.
      - state["errors"] gets a recoverable error entry for the broken model.
      - The orchestrator's overall status is still 'completed' (not 'failed').
    """
    csv_path = str(tmp_path / "clean_class.csv")
    df = _make_classification_dataframe()
    _write_clean_csv(csv_path, df)

    broken = _make_broken_spec()
    real_zoo = get_class_models()

    # Build a zoo with the broken estimator inserted in the middle
    mixed_zoo = real_zoo[:2] + [broken] + real_zoo[2:]

    monkeypatch.setattr(
        "src.agents.experiment_orchestrator_agent._select_zoo",
        lambda task_type: mixed_zoo,
    )

    state: AgentMLState = {
        "session_id": "test_iso",
        "task_type": "classification",
        "target_column": "target",
        "clean_dataset_path": csv_path,
        "errors": [],
    }

    result = experiment_orchestrator_agent(state)

    results = result["experiment_results"]
    # Find the broken entry specifically (this is the entry the prompt
    # asks to be shown at the end of the run).
    broken_entries = [r for r in results if r["model_name"] == broken["name"]]
    assert len(broken_entries) == 1, f"Expected exactly one broken entry, got {broken_entries}"
    broken_record = broken_entries[0]
    assert broken_record["success"] is False
    assert broken_record["error_type"] == "RuntimeError"
    assert "deliberately broken" in broken_record["error_message"]

    # Other (non-broken) models should still have run and succeeded.
    other_entries = [r for r in results if r["model_name"] != broken["name"]]
    other_successful = [r for r in other_entries if r["success"]]
    assert len(other_successful) >= 4, (
        f"Failure isolation broke: only {len(other_successful)} non-broken models succeeded"
    )

    # A recoverable error must have been logged for the broken model.
    recoverable_errors = [
        e for e in result["errors"]
        if e.get("phase") == "experiment_orchestrator"
        and e.get("recoverable") is True
        and broken["name"] in e.get("message", "")
    ]
    assert len(recoverable_errors) == 1, (
        f"Expected exactly one recoverable error mentioning {broken['name']!r}, got {recoverable_errors}"
    )

    # Status remains 'completed' — the failure did NOT kill the orchestrator.
    assert result["status"] == "completed"

    # Print the broken entry so it's visible in pytest output. The prompt
    # requires the failed run's log entry to be shown specifically.
    print("\n--- DELIBERATELY-BROKEN EXPERIMENT LOG ENTRY ---")
    for k, v in broken_record.items():
        print(f"  {k}: {v!r}")
    print("--- END BROKEN LOG ENTRY ---")


# ---------------------------------------------------------------------------
# 3. All-fail case — hard fail with diagnostic summary
# ---------------------------------------------------------------------------

def test_experiment_orchestrator_all_fail_hard_fails(monkeypatch, tmp_path):
    """
    If EVERY experiment fails, the orchestrator hard-fails with a diagnostic
    summary in state["errors"] and status='failed' (Rules.md §4).
    """
    csv_path = str(tmp_path / "clean_class.csv")
    df = _make_classification_dataframe()
    _write_clean_csv(csv_path, df)

    broken_zoo = [
        _make_broken_spec(f"BrokenClassifier_{i}") for i in range(3)
    ]
    monkeypatch.setattr(
        "src.agents.experiment_orchestrator_agent._select_zoo",
        lambda task_type: broken_zoo,
    )

    state: AgentMLState = {
        "session_id": "test_all_fail",
        "task_type": "classification",
        "target_column": "target",
        "clean_dataset_path": csv_path,
        "errors": [],
    }

    result = experiment_orchestrator_agent(state)

    assert result["status"] == "failed"
    assert len(result["experiment_results"]) == 3
    assert all(not r["success"] for r in result["experiment_results"])

    # A non-recoverable 'all_experiments_failed' error must be present
    # and must include a diagnostic summary mentioning every model.
    fatal = [
        e for e in result["errors"]
        if e.get("phase") == "experiment_orchestrator"
        and e.get("recoverable") is False
        and e.get("error_type") == "all_experiments_failed"
    ]
    assert len(fatal) == 1
    summary = fatal[0]["message"]
    assert "All experiments failed" in summary
    for spec in broken_zoo:
        assert spec["name"] in summary


# ---------------------------------------------------------------------------
# 4. Scope checkpoint — Architecture.md §6 #2
# ---------------------------------------------------------------------------

def test_experiment_orchestrator_scope_checkpoint_is_offered(tmp_path):
    """
    Inside a compiled LangGraph, the orchestrator's interrupt() fires
    on entry (Architecture.md §6 checkpoint 2). Resuming with an override
    applies the override (verified by the orchestrator only running the
    requested number of experiments).
    """
    csv_path = str(tmp_path / "clean_class.csv")
    df = _make_classification_dataframe(n=200)
    _write_clean_csv(csv_path, df)

    graph = build_graph()
    session_id = "test_scope"
    config = {"configurable": {"thread_id": session_id}}

    # Drive state through profiling → preprocessing → problem_detection
    # so the orchestrator sees all the fields it expects. Easiest: invoke
    # the graph from scratch and let it pause at interrupt().
    initial_state = {
        "session_id": session_id,
        "raw_file_path": csv_path,
        "target_column": "target",
        "errors": [],
        "status": "running",
    }

    paused = graph.invoke(initial_state, config)
    # If profiling/preprocessing didn't interrupt (they shouldn't), the
    # pause should be the experiment orchestrator's scope checkpoint.
    assert "__interrupt__" in paused, "Graph did not pause for any checkpoint"
    # The most recent interrupt should be from experiment_orchestrator
    last_interrupt_payload = paused["__interrupt__"][-1].value
    assert "scope" in last_interrupt_payload["message"].lower() or "experiment" in last_interrupt_payload["message"].lower()
    assert "default_scope" in last_interrupt_payload

    # Resume with an override that limits to 2 experiments.
    override = {
        "max_experiments": 2,
        "max_workers": 2,
        "time_cap_seconds": 0,
    }
    final = graph.invoke(Command(resume=override), config)

    # Verify orchestrator ran exactly 2 experiments
    exp_results = final.get("experiment_results", [])
    assert len(exp_results) == 2, (
        f"Expected exactly 2 experiment runs after override, got {len(exp_results)}: "
        f"{[r['model_name'] for r in exp_results]}"
    )


# ---------------------------------------------------------------------------
# 5. Direct runner sanity check
# ---------------------------------------------------------------------------

def test_run_single_experiment_success_record_shape():
    """_run_single_experiment returns a record with the expected keys."""
    spec = get_class_models()[1]  # LogisticRegression
    df = _make_classification_dataframe()
    X = df.drop(columns=["target"])
    y = df["target"]
    record = _run_single_experiment(spec, "classification", X, y, random_state=42)
    assert record["success"] is True
    assert record["model_name"] == spec["name"]
    assert record["task_type"] == "classification"
    assert "accuracy" in record["metrics"]


def test_run_single_experiment_captures_broken_failure():
    """_run_single_experiment converts a fit-time exception into a failure record."""
    spec = _make_broken_spec()
    df = _make_classification_dataframe()
    X = df.drop(columns=["target"])
    y = df["target"]
    record = _run_single_experiment(spec, "classification", X, y, random_state=42)
    assert record["success"] is False
    assert record["error_type"] == "RuntimeError"
    assert "deliberately broken" in record["error_message"]
    assert record["runtime_seconds"] >= 0
