"""
Unit tests for Data Preprocessing Agent (src/agents/preprocessing_agent.py)
"""

from __future__ import annotations

import os
import pytest
import pandas as pd
import numpy as np

from src.agents.preprocessing_agent import preprocessing_agent
from src.orchestrator.state import AgentMLState


def test_preprocessing_agent_unencodable_free_text(tmp_path):
    """
    Test that high-cardinality categorical columns (free text) are dropped and logged.
    """
    csv_file = tmp_path / "free_text.csv"
    data = {
        "id_col": [f"ID_{i}" for i in range(30)],
        "description": [f"Free text description of item number {i} with unique phrases." for i in range(30)],
        "numeric_feat": np.random.randn(30),
        "target": [i % 2 for i in range(30)]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)

    assert "clean_dataset_path" in result
    assert "preprocessing_log" in result

    # Read processed dataset
    df_clean = pd.read_csv(result["clean_dataset_path"])
    
    # "id_col" and "description" should be dropped
    assert "id_col" not in df_clean.columns
    assert "description" not in df_clean.columns
    assert "numeric_feat" in df_clean.columns
    assert "target" in df_clean.columns

    # Verify dropped logging entries
    log = result["preprocessing_log"]
    dropped_cols = [entry["column"] for entry in log if entry["operation"] == "drop"]
    assert "id_col" in dropped_cols
    assert "description" in dropped_cols


def test_preprocessing_agent_all_numeric(tmp_path):
    """
    Test that an all-numeric dataset gets scaled and imputed.
    """
    csv_file = tmp_path / "numeric.csv"
    data = {
        "feat1": [1.0, 2.0, None, 4.0, 5.0],
        "feat2": [10.0, 20.0, 30.0, None, 50.0],
        "target": [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)

    # Read processed dataset
    df_clean = pd.read_csv(result["clean_dataset_path"])

    # Verify imputation: no null values remain
    assert df_clean["feat1"].isna().sum() == 0
    assert df_clean["feat2"].isna().sum() == 0

    # Imputed value for feat1 should be the median (3.0)
    # Scaled check: standard deviation should be close to 1.0, mean close to 0.0 (or at least scaled)
    assert np.allclose(df_clean["feat1"].mean(), 0.0, atol=1e-5)

    # Check log contains imputation and scaling
    log = result["preprocessing_log"]
    assert any(e["column"] == "feat1" and e["operation"] == "impute" for e in log)
    assert any(e["column"] == "feat1" and e["operation"] == "scale" for e in log)


def test_preprocessing_agent_mixed(tmp_path):
    """
    Test that mixed numerical and categorical datasets are properly scaled, encoded, and imputed.
    """
    csv_file = tmp_path / "mixed.csv"
    data = {
        "age": [25, None, 35, 40],
        "city": ["NY", "LA", None, "NY"],
        "target": [0.5, 1.5, 2.5, 3.5]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": []
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    # Columns expected: age, city_LA, city_NY (since it is one-hot encoded, mode mode imputed missing)
    # The mode for city is "NY". The None will be imputed to "NY".
    # Therefore, we have city values: "NY", "LA", "NY", "NY".
    # One-hot encoded feature names out: city_LA, city_NY.
    assert "age" in df_clean.columns
    assert "city_LA" in df_clean.columns
    assert "city_NY" in df_clean.columns
    assert "target" in df_clean.columns

    # Verify imputation occurred
    assert df_clean.isna().sum().sum() == 0

    log = result["preprocessing_log"]
    assert any(e["column"] == "city" and e["operation"] == "onehot_encode" for e in log)
