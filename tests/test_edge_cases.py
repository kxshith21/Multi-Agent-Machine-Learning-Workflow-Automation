"""
Unit and integration tests for ML pipeline edge cases (Phase 7).

Tests:
  - Empty CSV (should fail profiling)
  - Single-column CSV (target-only / feature-only, should fail on 0 features in preprocessing)
  - All-missing column (should be dropped in preprocessing)
  - Huge-cardinality categorical column (should be dropped in preprocessing)
  - No numeric columns at all (should run categorical pipeline only)
"""

from __future__ import annotations

import os
import pytest
import pandas as pd
import numpy as np

from src.agents.profiling_agent import profiling_agent
from src.agents.preprocessing_agent import preprocessing_agent
from src.orchestrator.state import AgentMLState


def test_edge_case_empty_csv(tmp_path):
    """
    Empty CSV should raise ValueError during profiling (hard fail).
    """
    csv_file = tmp_path / "empty.csv"
    # Create an empty file
    csv_file.touch()

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "errors": []
    }

    # profiling_agent should raise ValueError or ParserError
    with pytest.raises((ValueError, pd.errors.EmptyDataError)):
        profiling_agent(state)


def test_edge_case_single_column_target_only(tmp_path):
    """
    Single-column CSV that is the target only (0 features) should fail gracefully in preprocessing.
    """
    csv_file = tmp_path / "target_only.csv"
    df = pd.DataFrame({"target": [1, 0, 1, 0, 1]})
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)
    assert result["status"] == "failed"
    assert any(e["error_type"] == "zero_features_error" for e in result["errors"])


def test_edge_case_all_missing_column(tmp_path):
    """
    Columns with 100% missing values should be dropped and warning logged, rather than raising exceptions.
    """
    csv_file = tmp_path / "all_missing.csv"
    df = pd.DataFrame({
        "all_missing_num": [np.nan, np.nan, np.nan, np.nan],
        "all_missing_cat": [np.nan, np.nan, np.nan, np.nan],
        "good_feature": [1.0, 2.0, 3.0, 4.0],
        "target": [0, 1, 0, 1]
    })
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)
    assert result["clean_dataset_path"] != ""
    
    # Read clean dataset
    df_clean = pd.read_csv(result["clean_dataset_path"])
    assert "all_missing_num" not in df_clean.columns
    assert "all_missing_cat" not in df_clean.columns
    assert "good_feature" in df_clean.columns

    # Verify log entry
    log = result["preprocessing_log"]
    dropped_cols = [entry["column"] for entry in log if entry["operation"] == "drop"]
    assert "all_missing_num" in dropped_cols
    assert "all_missing_cat" in dropped_cols


def test_edge_case_no_numeric_features(tmp_path):
    """
    If there are no numeric columns (only categorical), the pipeline should run encoding only.
    """
    csv_file = tmp_path / "no_numeric.csv"
    df = pd.DataFrame({
        "cat1": ["A", "B", "A", "B"],
        "cat2": ["X", "X", "Y", "Y"],
        "target": [0, 1, 0, 1]
    })
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)
    assert result["clean_dataset_path"] != ""
    
    df_clean = pd.read_csv(result["clean_dataset_path"])
    # cat1 -> cat1_A, cat1_B or similar
    # target should be present
    assert "target" in df_clean.columns
    assert not any(col.startswith("numeric__") for col in df_clean.columns)


def test_edge_case_huge_cardinality(tmp_path):
    """
    If a categorical column is extremely high cardinality, it should be dropped.
    """
    csv_file = tmp_path / "huge_cardinality.csv"
    # 30 unique string values for 30 rows
    df = pd.DataFrame({
        "huge_card": [f"val_{i}" for i in range(30)],
        "good_feat": [1, 2, 3] * 10,
        "target": [0, 1] * 15
    })
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])
    assert "huge_card" not in df_clean.columns
    assert "good_feat" in df_clean.columns


def _run_graph_auto_resume(csv, target, tmp_path):
    """
    Run the full graph, auto-accepting every interrupt checkpoint.
    The detection checkpoint MUST be resumed with an explicit non-empty payload
    — resuming it with {} makes LangGraph re-trigger the interrupt indefinitely.
    """
    from src.orchestrator.graph import build_graph
    from langgraph.types import Command
    import uuid

    graph = build_graph()
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    initial = {
        "session_id": session_id,
        "raw_file_path": str(csv),
        "target_column": target,
        "errors": [],
        "status": "running",
    }

    result = graph.invoke(initial, config)
    for _ in range(15):
        if "__interrupt__" not in result:
            break
        payload = result["__interrupt__"][-1].value
        if "detected_task_type" in payload:
            resume = {
                "target_column": payload.get("detected_target_column"),
                "task_type": payload.get("detected_task_type"),
            }
        elif "default_scope" in payload:
            ds = payload["default_scope"]
            resume = {
                "max_experiments": ds.get("max_experiments", 6),
                "max_workers": ds.get("max_workers", 4),
                "time_cap_seconds": 0,
            }
        elif "default_best_model_id" in payload:
            resume = {"best_model_id": payload["default_best_model_id"]}
        elif "feature_suggestions" in payload:
            # Phase 10 feature-engineering checkpoint — apply none by default.
            resume = {"selected_features": [], "custom_features": []}
        else:
            resume = {}
        result = graph.invoke(Command(resume=resume), config)
    return result


def test_edge_case_detection_checkpoint_resume_does_not_loop(tmp_path):
    """
    A low-confidence float-target dataset triggers the Phase 3 detection interrupt.
    Auto-resuming with the detected target/task must complete the pipeline, not loop.
    """
    csv_file = tmp_path / "float_target.csv"
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "feat1": rng.normal(size=40),
        "feat2": rng.normal(size=40),
        "target": [1.0, 0.0] * 20,  # float, 2 unique values -> confidence 0.5 -> interrupt
    })
    df.to_csv(csv_file, index=False)

    result = _run_graph_auto_resume(csv_file, "target", tmp_path)
    assert result.get("status") == "completed"
    assert result.get("task_type") == "classification"
    assert result.get("report_path")
    hard = [e for e in result.get("errors", []) if not e.get("recoverable", True)]
    assert len(hard) == 0


def test_edge_case_binary_target_not_scaled_by_preprocessing(tmp_path):
    """
    Regression: a binary target must NOT be StandardScaled to [-1, 1] by
    preprocessing. The graph runs Problem Detection BEFORE Preprocessing so the
    target is known and excluded from feature scaling. Otherwise XGBoost fails
    with 'Invalid classes inferred from unique values of y. Expected: [0 1], got [-1. 1.]'.
    """
    csv_file = tmp_path / "titanic.csv"
    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame({
        "PassengerId": list(range(1, n + 1)),
        "Survived": [0, 1] * (n // 2),
        "Pclass": rng.integers(1, 4, n),
        "Age": rng.normal(loc=30, scale=12, size=n).round(1),
        "Fare": rng.normal(loc=30, scale=20, size=n).round(2),
    })
    df.to_csv(csv_file, index=False)

    # target=None -> Problem Detection auto-detects 'Survived' via name match,
    # then Preprocessing must preserve {0,1} (not scale to {-1,1}).
    result = _run_graph_auto_resume(csv_file, None, tmp_path)

    assert result.get("status") == "completed"
    assert result.get("task_type") == "classification"
    assert result.get("target_column") == "Survived"
    hard = [e for e in result.get("errors", []) if not e.get("recoverable", True)]
    assert len(hard) == 0

    clean_path = result.get("clean_dataset_path")
    assert clean_path and os.path.exists(clean_path)
    clean = pd.read_csv(clean_path)
    survived = clean["Survived"].dropna().unique()
    assert sorted(survived.tolist()) == [0, 1], (
        f"Target was scaled during preprocessing: got {survived.tolist()!r}"
    )


def test_edge_case_feature_engineering_checkpoint_is_always_offered(tmp_path):
    """
    Phase 10: the Feature Engineering Selection checkpoint fires ALWAYS (not
    confidence-gated). A correlated-pair dataset triggers suggestions, the graph
    pauses at the checkpoint before preprocessing, and resuming with a selection
    plus a custom formula applies both and logs them in feature_engineering_log.
    """
    from src.orchestrator.graph import build_graph
    from langgraph.types import Command
    import uuid

    csv_file = tmp_path / "fe_graph.csv"
    rng = np.random.default_rng(3)
    sqft = 1000.0 + rng.random(40) * 500.0
    price = 200.0 * sqft + rng.normal(0, 5.0, 40)
    df = pd.DataFrame({"sqft": sqft, "price": price, "target": [0, 1] * 20})
    df.to_csv(csv_file, index=False)

    graph = build_graph()
    session_id = "fe_graph_test"
    config = {"configurable": {"thread_id": session_id}}
    initial = {
        "session_id": session_id,
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": [],
        "status": "running",
    }

    paused = graph.invoke(initial, config)
    # The graph pauses at the feature-engineering checkpoint (always offered).
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][-1].value
    assert "feature_suggestions" in payload
    assert any(s["type"] == "ratio" for s in payload["feature_suggestions"])

    chosen = [s["name"] for s in payload["feature_suggestions"] if s["type"] == "ratio"]
    assert len(chosen) == 1
    ratio_name = chosen[0]

    # Loop over every checkpoint: apply our feature selection at the
    # feature-engineering checkpoint, and auto-accept the rest.
    result = paused
    for _ in range(15):
        if "__interrupt__" not in result:
            break
        p = result["__interrupt__"][-1].value
        if "feature_suggestions" in p:
            resume = {
                "selected_features": chosen,
                "custom_features": [{"name": "luxury_score", "formula": "price / sqft"}],
            }
        elif "detected_task_type" in p:
            resume = {"target_column": p.get("detected_target_column"), "task_type": p.get("detected_task_type")}
        elif "default_scope" in p:
            ds = p["default_scope"]
            resume = {"max_experiments": ds.get("max_experiments", 6), "max_workers": ds.get("max_workers", 4), "time_cap_seconds": 0}
        elif "default_best_model_id" in p:
            resume = {"best_model_id": p["default_best_model_id"]}
        else:
            resume = {}
        result = graph.invoke(Command(resume=resume), config)

    assert result.get("status") == "completed"
    assert ratio_name in (result.get("selected_features") or [])
    assert any(c.get("name") == "luxury_score" for c in (result.get("custom_features") or []))

    clean_path = result.get("clean_dataset_path")
    assert clean_path and os.path.exists(clean_path)
    clean = pd.read_csv(clean_path)
    assert ratio_name in clean.columns
    assert "luxury_score" in clean.columns

    fe_log = result.get("feature_engineering_log") or []
    assert any(e.get("column") == "luxury_score" and e.get("source") == "custom" and e.get("operation") == "create" for e in fe_log)
    assert any(e.get("column") == ratio_name and e.get("source") == "suggested" and e.get("operation") == "create" for e in fe_log)

    hard = [e for e in result.get("errors", []) if not e.get("recoverable", True)]
    assert len(hard) == 0
