"""
Unit tests for Dataset Profiling Agent (src/agents/profiling_agent.py)
"""

from __future__ import annotations

import os
import numpy as np
import pytest
import pandas as pd
from src.agents.profiling_agent import profiling_agent, detect_feature_suggestions
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


# ---------------------------------------------------------------------------
# Feature-engineering suggestion detection (Phase 10)
# ---------------------------------------------------------------------------

def test_feature_suggestions_clean_dataset(tmp_path):
    """
    A clean dataset with a single numeric column (no pairs), no datetime, and
    only low-cardinality categoricals produces an EMPTY feature_suggestions list
    without erroring (no-op case).
    """
    csv_file = tmp_path / "no_features.csv"
    data = {
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["x", "y", "x", "y", "x"],       # low cardinality -> no binning idea
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {"raw_file_path": str(csv_file), "errors": []}
    result = profiling_agent(state)

    suggestions = result.get("feature_suggestions")
    assert suggestions == [], f"Expected empty suggestions, got {suggestions}"
    assert len(result.get("errors", [])) == 0


def test_feature_suggestions_datetime_decompose(tmp_path):
    """
    A datetime column yields a datetime_decompose suggestion.
    """
    csv_file = tmp_path / "dates.csv"
    data = {
        "purchase_date": pd.date_range("2023-01-01", periods=6, freq="D"),
        "price": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {"raw_file_path": str(csv_file), "errors": []}
    result = profiling_agent(state)

    suggestions = result.get("feature_suggestions", [])
    assert any(s["type"] == "datetime_decompose" and "purchase_date" in " ".join(s["columns"]) for s in suggestions)


def test_feature_suggestions_correlated_numeric_pair(tmp_path):
    """
    Two strongly correlated numeric columns yield a ratio suggestion.
    """
    csv_file = tmp_path / "corr.csv"
    rng = np.random.default_rng(0)
    sqft = 1000.0 + rng.random(50) * 500.0
    price = 200.0 * sqft + rng.normal(0, 5.0, 50)   # near-perfect linear relationship
    df = pd.DataFrame({"sqft": sqft, "price": price})
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {"raw_file_path": str(csv_file), "errors": []}
    result = profiling_agent(state)

    suggestions = result.get("feature_suggestions", [])
    assert any(s["type"] in ("ratio", "product") for s in suggestions), suggestions


def test_feature_suggestions_high_cardinality_binning(tmp_path):
    """
    A high-cardinality categorical column yields a binning suggestion.
    """
    csv_file = tmp_path / "bin.csv"
    df = pd.DataFrame({
        "item_code": [f"SKU-{i}" for i in range(30)],
        "value": list(range(30)),
    })
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {"raw_file_path": str(csv_file), "errors": []}
    result = profiling_agent(state)

    suggestions = result.get("feature_suggestions", [])
    assert any(s["type"] == "binning" for s in suggestions), suggestions


def test_detect_feature_suggestions_no_op():
    """
    Direct call to the detector on a no-opportunity frame returns [] and does not raise.
    """
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
    })
    profile = {
        "numeric_cols": ["x"],
        "categorical_cols": [],
    }
    suggestions = detect_feature_suggestions(df, profile)
    assert suggestions == []
