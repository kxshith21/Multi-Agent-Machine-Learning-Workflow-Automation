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


def test_problem_detection_clear_classification(tmp_path, monkeypatch):
    """
    Test problem detection with categorical target column (clear classification, confidence=1.0).
    The Groq narration is mocked so the assertion is deterministic (no live network).
    """
    csv_file = tmp_path / "class.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "target": ["A", "B", "A", "B"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat",
        lambda prompt, system=None, model=None, temperature=None: (
            "This is a categorical classification problem.", []
        )
    )

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


def test_problem_detection_clear_regression(tmp_path, monkeypatch):
    """
    Test problem detection with continuous target column (clear regression, confidence=1.0).
    The Groq narration is mocked so the assertion is deterministic (no live network).
    """
    csv_file = tmp_path / "reg.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "target": [1.2, 3.4, 5.1, 7.8, 9.0, 11.2, 13.5, 15.6, 17.8, 19.9]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat",
        lambda prompt, system=None, model=None, temperature=None: (
            "This is a regression problem.", []
        )
    )

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


def test_problem_detection_clustering(tmp_path, monkeypatch):
    """
    Test explicit unsupervised / clustering when the user requests 'no target' via
    the instruction (LLM resolves to clustering), which is the new contract for
    expressing genuine clustering (blank/None now means auto-detect instead).
    """
    csv_file = tmp_path / "cluster.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "feat2": [5.0, 6.0, 7.0, 8.0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    def _chat(prompt, system=None, model=None, temperature=None):
        if "Map the request" in prompt:
            return '{"target_column": null, "task_type": "clustering"}', []
        return "Unsupervised clustering.", []

    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat", _chat
    )

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "user_instruction": "This is unlabeled data, just cluster it.",
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


# --- New: natural-language instruction parsing ---------------------------------

def _mock_chat_factory(temp_path, csv_data, instruction_json=None):
    """
    Build a groq_client.chat mock. Returns (response_text, errors).
    - For the instruction-parsing call, return instruction_json.
    - For the narration call, return a plain text explanation.
    """
    def _chat(prompt, system=None, model=None, temperature=None):
        if "Map the request" in prompt:
            return instruction_json, []
        return "Narration text.", []
    return _chat


def test_problem_detection_instruction_classification(tmp_path, monkeypatch):
    """User says 'predict who survived' → LLM maps to Survived / classification."""
    csv_file = tmp_path / "titanic.csv"
    data = {
        "PassengerId": list(range(1, 6)),
        "Survived": [0, 1, 0, 1, 1],
        "Pclass": [3, 1, 3, 1, 2],
        "Fare": [7.25, 71.28, 8.05, 53.1, 8.46],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    mock = _mock_chat_factory(
        csv_file, data,
        instruction_json='{"target_column": "Survived", "task_type": "classification"}'
    )
    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat", mock
    )

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "user_instruction": "This is the Titanic csv, predict who survived or not.",
        "errors": [],
    }

    result = problem_detection_agent(state)
    assert result["target_column"] == "Survived"
    assert result["task_type"] == "classification"


def test_problem_detection_instruction_clustering(tmp_path, monkeypatch):
    """User explicitly asks for clustering (no target) via the instruction."""
    csv_file = tmp_path / "unlabeled.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": [5.0, 4.0, 3.0, 2.0, 1.0],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    mock = _mock_chat_factory(
        csv_file, data,
        instruction_json='{"target_column": null, "task_type": "clustering"}'
    )
    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat", mock
    )

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "user_instruction": "This is unlabeled data, just cluster it.",
        "errors": [],
    }

    result = problem_detection_agent(state)
    assert result["task_type"] == "clustering"
    assert result["target_column"] is None


def test_problem_detection_instruction_unknown_column_falls_back(tmp_path, monkeypatch):
    """
    LLM names a column that does not exist → must not crash; falls back to auto-detect.
    With a clearly named 'survived' column, the heuristic should recover to it.
    """
    csv_file = tmp_path / "titanic2.csv"
    data = {
        "PassengerId": list(range(1, 6)),
        "Survived": [0, 1, 0, 1, 1],
        "Age": [22.0, 38.0, 26.0, 35.0, 3.0],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    # LLM hallucinates a column name that isn't present.
    mock = _mock_chat_factory(
        csv_file, data,
        instruction_json='{"target_column": "died", "task_type": "classification"}'
    )
    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat", mock
    )

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "user_instruction": "predict who survived or not",
        "errors": [],
    }

    result = problem_detection_agent(state)
    # Heuristic should pick 'Survived' by name match; task classification by rules.
    assert result["target_column"] == "Survived"
    assert result["task_type"] == "classification"


def test_problem_detection_no_instruction_auto_detect(tmp_path):
    """
    Regression test: blank/no instruction must NOT force clustering.
    A dataset with a matching 'survived' column should be auto-detected.
    """
    csv_file = tmp_path / "autodetect.csv"
    data = {
        "PassengerId": list(range(1, 6)),
        "Survived": [0, 1, 0, 1, 1],
        "Age": [22.0, 38.0, 26.0, 35.0, 3.0],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "user_instruction": "",
        "errors": [],
    }

    # No mocks: if the heuristic were skipped, this would default to clustering
    # (the old bug). It must instead detect Survived.
    result = problem_detection_agent(state)
    assert result["target_column"] == "Survived"
    assert result["task_type"] == "classification"


def test_problem_detection_instruction_parse_error_falls_back(tmp_path, monkeypatch):
    """Groq raises on the instruction call → graceful fallback to auto-detect."""
    csv_file = tmp_path / "err.csv"
    data = {
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "target": ["A", "B", "A", "B"],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    def _boom(prompt, system=None, model=None, temperature=None):
        raise RuntimeError("groq down")

    monkeypatch.setattr(
        "src.agents.problem_detection_agent.groq_client.chat", _boom
    )

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "user_instruction": "predict who survived",
        "errors": [],
    }

    result = problem_detection_agent(state)
    # Heuristic recovers to 'target' column.
    assert result["target_column"] == "target"
    assert result["task_type"] == "classification"
    assert any(e["error_type"] == "instruction_parse_error" for e in result["errors"])
