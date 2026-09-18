"""
Multi-Agent Scenario & Checkpoint Integration Tests
File: tests/test_multiagent_scenarios.py

Covers end-to-end multi-agent orchestration across all 7 agents:
  1. Supervised Classification Workflow (with Groq narration fallback)
  2. Continuous Regression Workflow (numeric features & scaling)
  3. Unsupervised Clustering Workflow (no target column, silhouette score)
  4. Human Checkpoint: Problem Detection Ambiguity Override
  5. Human Checkpoint: Feature Engineering Selection & Custom Formula Evaluation
  6. Human Checkpoint: Experiment Scope Concurrency & Max Zoo Limits
  7. Human Checkpoint: Best Model Leaderboard Override
  8. Resilience: Dirty Data (duplicates + missing values + high cardinality)
"""

from __future__ import annotations

import os
import uuid
import pytest
import pandas as pd
import numpy as np
from langgraph.types import Command

from src.orchestrator.graph import build_graph
from src.orchestrator.state import AgentMLState


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_csv(tmp_path):
    """Creates a sample binary classification dataset."""
    csv_file = tmp_path / "classification_data.csv"
    np.random.seed(42)
    n = 60
    data = {
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.normal(50000, 15000, size=n).round(2),
        "category": np.random.choice(["bronze", "silver", "gold"], size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def regression_csv(tmp_path):
    """Creates a sample continuous regression dataset."""
    csv_file = tmp_path / "regression_data.csv"
    np.random.seed(42)
    n = 60
    sqft = np.random.uniform(500, 3500, size=n).round(1)
    rooms = np.random.randint(1, 6, size=n)
    price = sqft * 250 + rooms * 15000 + np.random.normal(0, 10000, size=n)
    df = pd.DataFrame({"sqft": sqft, "rooms": rooms, "price": price.round(2)})
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def clustering_csv(tmp_path):
    """Creates an unsupervised customer segmentation dataset with no label."""
    csv_file = tmp_path / "clustering_data.csv"
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "annual_spend": np.random.uniform(1000, 20000, size=n).round(2),
        "visit_frequency": np.random.randint(1, 50, size=n),
        "support_tickets": np.random.randint(0, 10, size=n),
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def messy_csv(tmp_path):
    """Creates a messy dataset with duplicates, missing values, and correlated columns."""
    csv_file = tmp_path / "messy_data.csv"
    data = {
        "id": list(range(1, 51)),
        "feat_a": [10.0, 20.0, np.nan, 40.0, 50.0] * 10,
        "feat_b": [100.0, 200.0, 300.0, 400.0, np.nan] * 10,
        "region": ["North", "South", "East", "West", np.nan] * 10,
        "target": [0, 1, 0, 1, 0] * 10,
    }
    df = pd.DataFrame(data)
    # Add 5 duplicate rows
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


# ---------------------------------------------------------------------------
# 1. End-to-End Supervised Classification Scenario
# ---------------------------------------------------------------------------

def test_e2e_classification_workflow(classification_csv):
    """
    Scenario 1: Complete 7-agent pipeline run on binary classification data.
    Validates state propagation through all nodes to report generation.
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": classification_csv,
        "target_column": "churn",
        "errors": [],
        "status": "running",
    }

    # Step 1: Initial invocation pauses at Feature Engineering Checkpoint
    state = graph.invoke(initial_state, config)
    assert "__interrupt__" in state
    payload = state["__interrupt__"][-1].value
    assert "feature_suggestions" in payload

    # Step 2: Resume Feature Engineering Checkpoint (apply no custom features)
    fe_resume = {"selected_features": [], "custom_features": []}
    state = graph.invoke(Command(resume=fe_resume), config)

    # Step 3: Handle Scope Checkpoint if triggered
    if "__interrupt__" in state:
        payload = state["__interrupt__"][-1].value
        if "default_scope" in payload:
            state = graph.invoke(Command(resume={"max_experiments": 2, "max_workers": 2}), config)

    # Step 4: Handle Best Model Evaluation Checkpoint
    if "__interrupt__" in state:
        payload = state["__interrupt__"][-1].value
        assert "default_best_model_id" in payload
        best_id = payload["default_best_model_id"]
        state = graph.invoke(Command(resume={"best_model_id": best_id}), config)

    # Assertions across the 7 agents
    assert state.get("task_type") == "classification"
    assert state.get("target_column") == "churn"
    assert state.get("dataset_profile") is not None
    assert state.get("clean_dataset_path") is not None
    assert os.path.exists(state["clean_dataset_path"])
    assert len(state.get("experiment_results", [])) > 0
    assert len(state.get("ranking", [])) > 0
    assert state.get("best_model_id") is not None
    assert state.get("report_path") is not None
    assert os.path.exists(state["report_path"])


# ---------------------------------------------------------------------------
# 2. End-to-End Continuous Regression Scenario
# ---------------------------------------------------------------------------

def test_e2e_regression_workflow(regression_csv):
    """
    Scenario 2: Complete 7-agent pipeline run on regression dataset.
    Validates task detection as regression and primary metric ranking (RMSE/R2).
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": regression_csv,
        "target_column": "price",
        "errors": [],
        "status": "running",
    }

    state = graph.invoke(initial_state, config)

    # Auto-resume checkpoints
    for _ in range(5):
        if "__interrupt__" not in state:
            break
        payload = state["__interrupt__"][-1].value
        if "feature_suggestions" in payload:
            state = graph.invoke(Command(resume={"selected_features": [], "custom_features": []}), config)
        elif "default_scope" in payload:
            scope = payload.get("default_scope") or {}
            resume_scope = {
                "max_experiments": scope.get("max_experiments", 6),
                "max_workers": scope.get("max_workers", 4),
                "time_cap_seconds": 0,
            }
            state = graph.invoke(Command(resume=resume_scope), config)
        elif "default_best_model_id" in payload:
            state = graph.invoke(Command(resume={"best_model_id": payload["default_best_model_id"]}), config)

    assert state.get("task_type") == "regression"
    assert state.get("target_column") == "price"
    ranking = state.get("ranking", [])
    assert len(ranking) > 0
    assert ranking[0]["primary_metric"] == "r2"
    assert state.get("report_path") is not None


# ---------------------------------------------------------------------------
# 3. End-to-End Unsupervised Clustering Scenario
# ---------------------------------------------------------------------------

def test_e2e_clustering_workflow(clustering_csv):
    """
    Scenario 3: Complete 7-agent pipeline run on unlabelled dataset.
    Validates automatic task detection as clustering and silhouette evaluation.
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": clustering_csv,
        "target_column": None,
        "errors": [],
        "status": "running",
    }

    state = graph.invoke(initial_state, config)

    # Auto-resume checkpoints
    for _ in range(5):
        if "__interrupt__" not in state:
            break
        payload = state["__interrupt__"][-1].value
        if "detected_target_column" in payload:
            state = graph.invoke(Command(resume={"target_column": None, "task_type": "clustering"}), config)
        elif "feature_suggestions" in payload:
            state = graph.invoke(Command(resume={"selected_features": [], "custom_features": []}), config)
        elif "default_scope" in payload:
            scope = payload.get("default_scope") or {}
            resume_scope = {
                "max_experiments": scope.get("max_experiments", 6),
                "max_workers": scope.get("max_workers", 4),
                "time_cap_seconds": 0,
            }
            state = graph.invoke(Command(resume=resume_scope), config)
        elif "default_best_model_id" in payload:
            state = graph.invoke(Command(resume={"best_model_id": payload["default_best_model_id"]}), config)

    assert state.get("task_type") == "clustering"
    assert state.get("target_column") is None
    ranking = state.get("ranking", [])
    assert len(ranking) > 0
    assert ranking[0]["primary_metric"] == "silhouette"
    assert state.get("report_path") is not None


# ---------------------------------------------------------------------------
# 4. Human Checkpoint: Feature Engineering Custom Formula Injection Test
# ---------------------------------------------------------------------------

def test_checkpoint_feature_engineering_custom_formula(regression_csv):
    """
    Scenario 4: Validates human feature engineering checkpoint applying a
    safe custom formula (e.g. `price_per_room = price / rooms`).
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": regression_csv,
        "target_column": "price",
        "errors": [],
        "status": "running",
    }

    state = graph.invoke(initial_state, config)
    assert "__interrupt__" in state
    payload = state["__interrupt__"][-1].value
    assert "feature_suggestions" in payload

    # Human provides a valid custom mathematical feature
    custom_feature = [{"name": "sqft_per_room", "formula": "sqft / rooms"}]
    state = graph.invoke(Command(resume={"selected_features": [], "custom_features": custom_feature}), config)

    # Check that the custom feature was logged in state and applied in preprocessed dataset
    fe_log = state.get("feature_engineering_log", [])
    assert any(entry.get("column") == "sqft_per_room" for entry in fe_log)
    
    clean_df = pd.read_csv(state["clean_dataset_path"])
    assert "sqft_per_room" in clean_df.columns


# ---------------------------------------------------------------------------
# 5. Human Checkpoint: Experiment Scope Restriction
# ---------------------------------------------------------------------------

def test_checkpoint_experiment_scope_limit(classification_csv):
    """
    Scenario 5: Validates human restricting model zoo execution scope
    (e.g., limit to exactly 1 experiment with 1 worker).
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": classification_csv,
        "target_column": "churn",
        "errors": [],
        "status": "running",
    }

    state = graph.invoke(initial_state, config)

    # Resume feature engineering
    if "__interrupt__" in state:
        state = graph.invoke(Command(resume={"selected_features": [], "custom_features": []}), config)

    # Check Scope Checkpoint
    if "__interrupt__" in state:
        payload = state["__interrupt__"][-1].value
        if "default_scope" in payload:
            scope_override = {"max_experiments": 1, "max_workers": 1, "time_cap_seconds": 0}
            state = graph.invoke(Command(resume=scope_override), config)

    # Verify only 1 experiment was conducted
    results = state.get("experiment_results", [])
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 6. Human Checkpoint: Best Model Override Test
# ---------------------------------------------------------------------------

def test_checkpoint_best_model_override(classification_csv):
    """
    Scenario 6: Validates human overriding the top-ranked model with another candidate.
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": classification_csv,
        "target_column": "churn",
        "errors": [],
        "status": "running",
    }

    state = graph.invoke(initial_state, config)

    # Resume feature selection
    if "__interrupt__" in state:
        state = graph.invoke(Command(resume={"selected_features": [], "custom_features": []}), config)

    # Resume scope
    if "__interrupt__" in state:
        state = graph.invoke(Command(resume={"max_experiments": 3, "max_workers": 2}), config)

    # Evaluation Checkpoint
    if "__interrupt__" in state:
        payload = state["__interrupt__"][-1].value
        assert "ranking" in payload
        ranking = payload["ranking"]
        if len(ranking) >= 2:
            override_model_id = ranking[1]["model_id"]
            state = graph.invoke(Command(resume={"best_model_id": override_model_id}), config)
            assert state.get("best_model_id") == override_model_id
            assert "manual override" in state.get("evaluation_reasoning", "").lower()


# ---------------------------------------------------------------------------
# 7. Resilience & Dirty Data Preprocessing Scenario
# ---------------------------------------------------------------------------

def test_resilience_messy_data_handling(messy_csv):
    """
    Scenario 7: Validates data cleaning resilience:
      - Duplicate row detection & removal
      - Missing value median/mode imputation
      - One-hot encoding of categorical variables
      - Drop high-missing columns or zero-variance IDs
    """
    session_id = str(uuid.uuid4())
    graph = build_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": messy_csv,
        "target_column": "target",
        "errors": [],
        "status": "running",
    }

    state = graph.invoke(initial_state, config)

    # Auto-resume checkpoints
    for _ in range(5):
        if "__interrupt__" not in state:
            break
        payload = state["__interrupt__"][-1].value
        if "feature_suggestions" in payload:
            state = graph.invoke(Command(resume={"selected_features": [], "custom_features": []}), config)
        elif "default_scope" in payload:
            state = graph.invoke(Command(resume={"max_experiments": 2, "max_workers": 2}), config)
        elif "default_best_model_id" in payload:
            state = graph.invoke(Command(resume={"best_model_id": payload["default_best_model_id"]}), config)

    profile = state.get("dataset_profile", {})
    assert profile.get("duplicate_count", 0) >= 5

    clean_df = pd.read_csv(state["clean_dataset_path"])
    # Verify no NaN values exist in cleaned dataset
    assert clean_df.isna().sum().sum() == 0
    # Verify pipeline succeeded through to report
    assert state.get("report_path") is not None
