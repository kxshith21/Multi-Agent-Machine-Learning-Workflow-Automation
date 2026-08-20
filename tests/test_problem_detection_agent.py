"""
Unit tests for Problem Detection Agent (src/agents/problem_detection_agent.py)
"""

from __future__ import annotations

import os
import pytest
import pandas as pd
from langgraph.types import Command

from src.orchestrator.graph import build_graph, run_pipeline
from src.agents.problem_detection_agent import problem_detection_agent
from src.orchestrator.state import AgentMLState


def test_problem_detection_clear_classification(tmp_path):
    """
    Test problem detection with categorical target column (clear classification, confidence=1.0).
    """
    csv_file = tmp_path / "class.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "target": ["A", "B", "A", "B"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = problem_detection_agent(state)
    assert result["task_type"] == "classification"
    assert result["target_column"] == "target"
    assert result["detection_confidence"] == 1.0
    assert "categorical" in result["detection_reasoning"].lower()


def test_problem_detection_clear_regression(tmp_path):
    """
    Test problem detection with continuous target column (clear regression, confidence=1.0).
    """
    csv_file = tmp_path / "reg.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        # Floats, not all integers, unique count = 10. This must go to regression directly.
        "target": [1.2, 3.4, 5.1, 7.8, 9.0, 11.2, 13.5, 15.6, 17.8, 19.9]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = problem_detection_agent(state)
    assert result["task_type"] == "regression"
    assert result["target_column"] == "target"
    assert result["detection_confidence"] == 1.0
    assert "regression" in result["detection_reasoning"].lower()


def test_problem_detection_clustering(tmp_path):
    """
    Test problem detection when target column is explicitly None (unsupervised / clustering, confidence=1.0).
    """
    csv_file = tmp_path / "cluster.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "feat2": [5.0, 6.0, 7.0, 8.0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "errors": []
    }

    result = problem_detection_agent(state)
    assert result["task_type"] == "clustering"
    assert result["target_column"] is None
    assert result["detection_confidence"] == 1.0
    assert "clustering" in result["detection_reasoning"].lower()


def test_problem_detection_ambiguous_target_and_interrupt(tmp_path):
    """
    Test that an ambiguous numeric target causes an interrupt, and when resumed
    with a human override, the override is correctly applied.
    """
    csv_file = tmp_path / "ambiguous.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # Integer target with 5 unique values (ambiguous)
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    # Compile graph with memory saver
    graph = build_graph()
    session_id = "test_ambiguous_thread"
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "session_id": session_id,
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": [],
        "status": "running"
    }

    # Run the graph. It should pause and return the state containing __interrupt__.
    res = graph.invoke(initial_state, config)
    
    assert "__interrupt__" in res
    assert len(res["__interrupt__"]) > 0
    
    interrupt_payload = res["__interrupt__"][0].value
    assert "ambiguous" in interrupt_payload["message"].lower()
    assert interrupt_payload["detected_target_column"] == "target"
    assert interrupt_payload["detected_task_type"] == "classification"
    assert interrupt_payload["confidence"] == 0.5

    # Resume the graph with a user override (e.g. override to regression)
    resume_command = Command(resume={"target_column": "target", "task_type": "regression"})
    final_state = graph.invoke(resume_command, config)

    # Verify target and task type after resume
    assert final_state["task_type"] == "regression"
    assert final_state["target_column"] == "target"
    assert final_state["detection_confidence"] == 1.0
    assert "human override" in final_state["detection_reasoning"].lower()
