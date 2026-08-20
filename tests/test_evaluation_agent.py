"""
Unit tests for the Model Evaluation Agent
(src/agents/evaluation_agent.py + src/metrics/task_metrics.py).

Phase 5 spec (Phases.md + Architecture.md §6 #3):

  - Produces a correct ranked leaderboard for each task type.
  - Override checkpoint actually changes best_model_id when invoked
    inside a compiled LangGraph.
  - Failed models (success=False from Phase 4) are excluded.
  - Reasoning string is human-readable and mentions the top model by name.
"""

from __future__ import annotations

import math
import pytest
from langgraph.types import Command

from src.agents.evaluation_agent import evaluation_agent
from src.metrics.task_metrics import (
    PRIMARY_METRIC,
    TIEBREAKER_METRIC,
    rank_results,
    metric_value,
    direction_for,
)
from src.orchestrator.state import AgentMLState
from src.orchestrator.graph import build_graph


# ---------------------------------------------------------------------------
# Helpers — fabricate experiment_results for unit tests
# ---------------------------------------------------------------------------

def _class_record(name: str, accuracy: float, f1: float, success: bool = True) -> dict:
    return {
        "model_id": name,
        "model_name": name,
        "family": "tree",
        "params": {},
        "task_type": "classification",
        "success": success,
        "error_type": None,
        "error_message": None,
        "metrics": {"accuracy": accuracy, "f1": f1, "precision": accuracy, "recall": f1},
        "runtime_seconds": 0.5,
        "n_train_samples": 100,
        "n_test_samples": 50,
    }


def _reg_record(name: str, r2: float, rmse: float, mae: float, success: bool = True) -> dict:
    return {
        "model_id": name,
        "model_name": name,
        "family": "tree",
        "params": {},
        "task_type": "regression",
        "success": success,
        "error_type": None,
        "error_message": None,
        "metrics": {"r2": r2, "rmse": rmse, "mae": mae},
        "runtime_seconds": 0.5,
        "n_train_samples": 100,
        "n_test_samples": 50,
    }


def _clust_record(
    name: str,
    silhouette: float,
    n_clusters: int = 3,
    n_noise: int = 0,
    success: bool = True,
) -> dict:
    return {
        "model_id": name,
        "model_name": name,
        "family": "centroid",
        "params": {},
        "task_type": "clustering",
        "success": success,
        "error_type": None,
        "error_message": None,
        "metrics": {
            "silhouette": silhouette,
            "n_clusters": float(n_clusters),
            "n_noise": float(n_noise),
        },
        "runtime_seconds": 0.5,
        "n_train_samples": 100,
        "n_test_samples": 0,
    }


# ---------------------------------------------------------------------------
# 1. task_metrics primitives
# ---------------------------------------------------------------------------

def test_metric_direction_defaults_higher_is_better():
    assert direction_for("accuracy") == "higher_is_better"
    assert direction_for("rmse") == "lower_is_better"
    assert direction_for("mae") == "lower_is_better"
    assert direction_for("r2") == "higher_is_better"
    assert direction_for("silhouette") == "higher_is_better"
    assert direction_for("unknown_metric") == "higher_is_better"


def test_metric_value_handles_nan_and_missing():
    """NaN/missing should sort to the end regardless of direction."""
    rec_ok = {"metrics": {"rmse": 1.0, "accuracy": 0.9}}
    rec_missing = {"metrics": {}}
    rec_nan = {"metrics": {"rmse": float("nan")}}

    assert metric_value(rec_ok, "rmse") == 1.0
    assert metric_value(rec_ok, "accuracy") == 0.9
    # missing for lower_is_better -> +inf (worst)
    assert math.isinf(metric_value(rec_missing, "rmse")) and metric_value(rec_missing, "rmse") > 0
    # missing for higher_is_better -> -inf (worst)
    assert math.isinf(metric_value(rec_missing, "accuracy")) and metric_value(rec_missing, "accuracy") < 0
    # NaN treated the same as missing
    assert math.isinf(metric_value(rec_nan, "rmse")) and metric_value(rec_nan, "rmse") > 0


# ---------------------------------------------------------------------------
# 2. Ranking correctness — classification
# ---------------------------------------------------------------------------

def test_classification_ranking_picks_highest_accuracy():
    results = [
        _class_record("A_low",  accuracy=0.60, f1=0.55),
        _class_record("B_high", accuracy=0.92, f1=0.90),
        _class_record("C_mid",  accuracy=0.78, f1=0.70),
    ]
    ranking = rank_results(results, "classification")
    assert [r["model_name"] for r in ranking] == ["B_high", "C_mid", "A_low"]
    assert ranking[0]["primary_metric"] == "accuracy"
    assert ranking[0]["primary_value"] == pytest.approx(0.92)
    assert [r["rank"] for r in ranking] == [1, 2, 3]


def test_classification_tiebreaker_on_accuracy_uses_f1():
    """Two models tied on accuracy — higher f1 wins."""
    results = [
        _class_record("AA", accuracy=0.85, f1=0.70),
        _class_record("BB", accuracy=0.85, f1=0.80),
    ]
    ranking = rank_results(results, "classification")
    assert ranking[0]["model_name"] == "BB"
    assert ranking[1]["model_name"] == "AA"


# ---------------------------------------------------------------------------
# 3. Ranking correctness — regression
# ---------------------------------------------------------------------------

def test_regression_ranking_picks_highest_r2():
    results = [
        _reg_record("Reg_low",  r2=0.30, rmse=10.0, mae=8.0),
        _reg_record("Reg_high", r2=0.95, rmse=1.5,  mae=1.2),
        _reg_record("Reg_mid",  r2=0.70, rmse=5.0,  mae=4.0),
    ]
    ranking = rank_results(results, "regression")
    assert [r["model_name"] for r in ranking] == ["Reg_high", "Reg_mid", "Reg_low"]
    assert ranking[0]["primary_metric"] == "r2"


def test_regression_tiebreaker_on_r2_uses_rmse_ascending():
    """Two models tied on R² — lower rmse wins."""
    results = [
        _reg_record("AA", r2=0.90, rmse=2.0, mae=1.5),
        _reg_record("BB", r2=0.90, rmse=1.0, mae=1.5),
    ]
    ranking = rank_results(results, "regression")
    assert ranking[0]["model_name"] == "BB"


# ---------------------------------------------------------------------------
# 4. Ranking correctness — clustering
# ---------------------------------------------------------------------------

def test_clustering_ranking_picks_highest_silhouette():
    results = [
        _clust_record("Clust_bad",  silhouette=0.10),
        _clust_record("Clust_best", silhouette=0.72),
        _clust_record("Clust_mid",  silhouette=0.45),
    ]
    ranking = rank_results(results, "clustering")
    assert [r["model_name"] for r in ranking] == ["Clust_best", "Clust_mid", "Clust_bad"]


def test_clustering_ranking_handles_nan_silhouette():
    """NaN silhouette must sort to the END (never appear as a winner)."""
    results = [
        _clust_record("Clust_nan",   silhouette=float("nan")),
        _clust_record("Clust_good",  silhouette=0.40),
    ]
    ranking = rank_results(results, "clustering")
    assert ranking[0]["model_name"] == "Clust_good"
    assert ranking[1]["model_name"] == "Clust_nan"


# ---------------------------------------------------------------------------
# 5. Mixed success / failure — failed models excluded
# ---------------------------------------------------------------------------

def test_ranking_excludes_failed_models():
    results = [
        _class_record("Good_A", accuracy=0.80, f1=0.75),
        _class_record("Failed", accuracy=1.00, f1=1.00, success=False),
        _class_record("Good_B", accuracy=0.85, f1=0.80),
    ]
    ranking = rank_results(results, "classification")
    names = [r["model_name"] for r in ranking]
    assert "Failed" not in names
    assert names == ["Good_B", "Good_A"]


# ---------------------------------------------------------------------------
# 6. End-to-end evaluation_agent — happy paths
# ---------------------------------------------------------------------------

def test_evaluation_agent_classification_happy_path():
    results = [
        _class_record("A", accuracy=0.60, f1=0.55),
        _class_record("B", accuracy=0.92, f1=0.90),
        _class_record("C", accuracy=0.78, f1=0.70),
    ]
    state: AgentMLState = {
        "session_id": "t1",
        "task_type": "classification",
        "experiment_results": results,
        "errors": [],
    }
    out = evaluation_agent(state)
    assert out["best_model_id"] == "B"
    assert [r["model_name"] for r in out["ranking"]] == ["B", "C", "A"]
    # Reasoning string mentions the winner
    assert "B" in out["evaluation_reasoning"]
    assert "classification" in out["evaluation_reasoning"]
    assert out["status"] == "completed"


def test_evaluation_agent_regression_happy_path():
    results = [
        _reg_record("R1", r2=0.40, rmse=8.0, mae=6.0),
        _reg_record("R2", r2=0.95, rmse=1.5, mae=1.2),
    ]
    state: AgentMLState = {
        "session_id": "t2",
        "task_type": "regression",
        "experiment_results": results,
        "errors": [],
    }
    out = evaluation_agent(state)
    assert out["best_model_id"] == "R2"


def test_evaluation_agent_excludes_failed_models():
    results = [
        _class_record("Good_A", accuracy=0.80, f1=0.75),
        _class_record("Broken", accuracy=0.99, f1=0.99, success=False),
    ]
    state: AgentMLState = {
        "session_id": "t3",
        "task_type": "classification",
        "experiment_results": results,
        "errors": [],
    }
    out = evaluation_agent(state)
    assert out["best_model_id"] == "Good_A"


def test_evaluation_agent_no_successful_results():
    """If every experiment failed, best_model_id='none' and reasoning is set."""
    results = [
        _class_record("Broken1", accuracy=0.99, f1=0.99, success=False),
        _class_record("Broken2", accuracy=0.99, f1=0.99, success=False),
    ]
    state: AgentMLState = {
        "session_id": "t4",
        "task_type": "classification",
        "experiment_results": results,
        "errors": [],
    }
    out = evaluation_agent(state)
    assert out["best_model_id"] == "none"
    assert out["status"] == "completed"
    assert "no successful" in out["evaluation_reasoning"].lower() or \
           "no best model" in out["evaluation_reasoning"].lower()


# ---------------------------------------------------------------------------
# 7. Override checkpoint — Architecture.md §6 #3
# ---------------------------------------------------------------------------

def test_evaluation_agent_override_changes_best_model_id(tmp_path):
    """
    Inside a compiled LangGraph, the evaluation agent's interrupt() fires.
    Resuming with an override payload actually changes best_model_id.

    Uses a real LangGraph run: profiling/preprocessing/problem_detection/
    experiment_orchestrator all run, then evaluation pauses at its
    checkpoint. We resume with an override pointing to a non-winning
    model and assert best_model_id changes.
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    # Build a synthetic clean CSV that the orchestrator will succeed on.
    X, y = make_classification(
        n_samples=200, n_features=6, n_informative=4, n_classes=2, random_state=0
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df["target"] = y
    csv_path = str(tmp_path / "clean.csv")
    df.to_csv(csv_path, index=False)

    graph = build_graph()
    session_id = "test_eval_override"
    config = {"configurable": {"thread_id": session_id}}
    initial_state = {
        "session_id": session_id,
        "raw_file_path": csv_path,
        "target_column": "target",
        "errors": [],
        "status": "running",
    }

    # First invoke: should pause at the experiment orchestrator's scope
    # checkpoint (Phase 4 §6 #2).
    paused = graph.invoke(initial_state, config)
    assert "__interrupt__" in paused
    # Resume Phase 4's scope checkpoint with full defaults
    after_scope = graph.invoke(Command(resume={"max_experiments": 6, "max_workers": 4}), config)
    # Should now be paused at Phase 5's evaluation checkpoint
    assert "__interrupt__" in after_scope
    last_payload = after_scope["__interrupt__"][-1].value
    assert "ranking" in last_payload
    assert "default_best_model_id" in last_payload

    automatic_best = last_payload["default_best_model_id"]
    ranking = last_payload["ranking"]
    # Pick a non-winner (last in the ranking) as our override target.
    override_target = ranking[-1]["model_id"]
    assert override_target != automatic_best, "Test data should yield a non-trivial ranking"

    print(f"\n--- AUTOMATIC RANKING (top 3) ---")
    for entry in ranking[:3]:
        print(
            f"  rank {entry['rank']}: {entry['model_name']} "
            f"({entry['primary_metric']}={entry['primary_value']:.4f})"
        )
    print(f"  -> automatic best_model_id: {automatic_best}")
    print(f"--- OVERRIDE REQUESTED -> best_model_id: {override_target} ---")

    # Resume with the override
    final = graph.invoke(
        Command(resume={"best_model_id": override_target}),
        config,
    )

    assert final["best_model_id"] == override_target, (
        f"Override did not take effect: best_model_id={final['best_model_id']!r} "
        f"expected={override_target!r}"
    )
    # Reasoning string should mention both the override and the original top pick
    assert "override" in final["evaluation_reasoning"].lower()
    assert override_target in final["evaluation_reasoning"]

    print(f"--- FINAL evaluation_reasoning ---")
    print(final["evaluation_reasoning"])
    print(f"--- FINAL best_model_id: {final['best_model_id']} ---")
