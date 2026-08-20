"""
Unit tests for Dataset Profiling Agent (src/agents/profiling_agent.py)
"""

from __future__ import annotations

import os
import pytest
import pandas as pd
from src.agents.profiling_agent import profiling_agent
from src.orchestrator.state import AgentMLState


def test_profiling_agent_clean_dataset(tmp_path):
    """
    Test profiling_agent on a clean, valid dataset.
    """
    csv_file = tmp_path / "clean.csv"
    data = {
        "age": [25, 30, 35, 40],
        "salary": [50000.0, 60000.0, 70000.0, 80000.0],
        "city": ["New York", "Chicago", "Boston", "New York"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "errors": []
    }

    result = profiling_agent(state)

    assert "dataset_profile" in result
    profile = result["dataset_profile"]

    # Verify shape
    assert profile["num_rows"] == 4
    assert profile["num_cols"] == 3
    assert profile["duplicate_count"] == 0

    # Verify column division
    assert "age" in profile["numeric_cols"]
    assert "salary" in profile["numeric_cols"]
    assert "city" in profile["categorical_cols"]

    # Verify statistics
    assert profile["column_stats"]["age"]["mean"] == 32.5
    assert profile["column_stats"]["age"]["unique_count"] == 4
    assert profile["column_stats"]["city"]["unique_count"] == 3
    assert profile["column_stats"]["city"]["top_value"] == "New York"
    assert profile["column_stats"]["city"]["top_freq"] == 2

    # Verify error list is empty
    assert len(result.get("errors", [])) == 0


def test_profiling_agent_missing_values(tmp_path):
    """
    Test profiling_agent on a dataset with missing values, including one column > 50% missing.
    """
    csv_file = tmp_path / "missing.csv"
    # age has 0 missing, salary has 1/4 missing (25%), code has 3/4 missing (75%)
    data = {
        "age": [25, 30, 35, 40],
        "salary": [50000.0, None, 70000.0, 80000.0],
        "code": [None, None, "A", None]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "errors": []
    }

    result = profiling_agent(state)

    assert "dataset_profile" in result
    profile = result["dataset_profile"]

    # Verify missing percentages
    assert profile["missing_pct"]["age"] == 0.0
    assert profile["missing_pct"]["salary"] == 25.0
    assert profile["missing_pct"]["code"] == 75.0

    # Verify warning generated for >50% missing values
    errors = result.get("errors", [])
    assert len(errors) == 1
    assert errors[0]["error_type"] == "high_missing_values"
    assert "code" in errors[0]["message"]
    assert errors[0]["recoverable"] is True


def test_profiling_agent_all_duplicates(tmp_path):
    """
    Test profiling_agent on a dataset with all duplicate rows.
    """
    csv_file = tmp_path / "duplicates.csv"
    data = {
        "item": ["apple", "apple", "apple", "apple"],
        "price": [1.0, 1.0, 1.0, 1.0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "errors": []
    }

    result = profiling_agent(state)

    assert "dataset_profile" in result
    profile = result["dataset_profile"]

    # 4 rows, but 3 are duplicates of the first row
    assert profile["num_rows"] == 4
    assert profile["duplicate_count"] == 3


def test_profiling_agent_corrupt_csv_hard_fail(tmp_path):
    """
    Verify that an unparseable or corrupt CSV causes a hard fail.
    """
    csv_file = tmp_path / "corrupt.csv"
    with open(csv_file, "w") as f:
        f.write("column1,column2\nval1,val2,extra_val_causing_unbalance\nextra,extra2\n")

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "errors": []
    }

    # Rules.md §4: a corrupt/unloadable CSV is a hard fail.
    # An unbalanced/corrupt file might be parsed by pandas with a warning/error depending on settings.
    # Let's write binary garbage or an invalid path to guarantee pd.read_csv fails.
    bad_file = tmp_path / "does_not_exist.csv"
    state_no_file: AgentMLState = {
        "raw_file_path": str(bad_file),
        "errors": []
    }
    with pytest.raises(Exception):
        profiling_agent(state_no_file)
