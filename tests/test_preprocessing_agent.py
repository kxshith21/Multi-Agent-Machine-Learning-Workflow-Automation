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
from src.utils.safe_formula import FormulaValidationError, validate_formula, evaluate_formula


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


# ---------------------------------------------------------------------------
# Feature Engineering Selection application (Phase 10)
# ---------------------------------------------------------------------------

def test_preprocessing_applies_selected_and_custom_features(tmp_path):
    """
    A checked suggestion AND a user-supplied custom formula are both created and
    logged into feature_engineering_log (source: suggested / custom).
    """
    csv_file = tmp_path / "fe.csv"
    rng = np.random.default_rng(1)
    sqft = 1000.0 + rng.random(20) * 500.0
    price = 200.0 * sqft + rng.normal(0, 5.0, 20)
    df = pd.DataFrame({"sqft": sqft, "price": price, "target": [0, 1] * 10})
    df.to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": [],
        "feature_suggestions": [
            {
                "name": "price_per_sqft",
                "type": "ratio",
                "description": "Ratio suggestion",
                "columns": ["price", "sqft"],
            }
        ],
        "selected_features": ["price_per_sqft"],
        "custom_features": [
            {"name": "luxury_score", "formula": "price / (sqft + 1)"}
        ],
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    # Suggested feature created (ratio)
    assert "price_per_sqft" in df_clean.columns
    # Custom feature created
    assert "luxury_score" in df_clean.columns

    fe_log = result.get("feature_engineering_log", [])
    suggested_entry = [e for e in fe_log if e.get("source") == "suggested" and e.get("column") == "price_per_sqft"]
    custom_entry = [e for e in fe_log if e.get("source") == "custom" and e.get("column") == "luxury_score"]
    assert len(suggested_entry) == 1
    assert suggested_entry[0]["operation"] == "create"
    assert len(custom_entry) == 1
    assert custom_entry[0]["operation"] == "create"
    assert custom_entry[0]["formula"] == "price / (sqft + 1)"


def test_custom_formula_malicious_input_is_rejected_and_not_executed(tmp_path):
    """
    CRITICAL SAFETY: a malicious custom formula is rejected by the restricted
    evaluator and NEVER executed. The eval path must raise FormulaValidationError
    for function calls / imports / attribute access.
    """
    csv_file = tmp_path / "mal.csv"
    df = pd.DataFrame({
        "price": [100.0, 110.0, 120.0],
        "sqft": [10.0, 11.0, 12.0],
        "target": [0, 1, 0],
    })
    df.to_csv(csv_file, index=False)

    # 1. The dangerous input is rejected by validate_formula directly.
    malicious = "__import__('os').system('ls')"
    with pytest.raises(FormulaValidationError):
        validate_formula(malicious, {"price", "sqft"})

    # 2. A benign arithmetic formula over real columns is accepted and evaluated.
    validate_formula("price / sqft", {"price", "sqft"})
    series = evaluate_formula("price / sqft", {"price", "sqft"}, df)
    assert abs(series.iloc[0] - 10.0) < 1e-9

    # 3. Feeding the malicious formula through preprocessing must produce a
    #    recoverable error and NOT create the column, and must NOT execute.
    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "target",
        "errors": [],
        "custom_features": [{"name": "evil_col", "formula": malicious}],
    }
    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])
    assert "evil_col" not in df_clean.columns

    fe_log = result.get("feature_engineering_log", [])
    assert any(e.get("column") == "evil_col" and e.get("operation") == "skip" for e in fe_log)
    assert any(
        e.get("error_type") == "custom_formula_rejected"
        for e in result.get("errors", [])
    )

    # 4. Additional malicious variants must also be rejected.
    for variant in [
        "__import__('os').system('rm -rf /')",
        "price.__class__",
        "eval('1')",
        "open('x')",
        "price if True else sqft",   # conditional (unsupported node)
    ]:
        with pytest.raises(FormulaValidationError):
            validate_formula(variant, {"price", "sqft"})
