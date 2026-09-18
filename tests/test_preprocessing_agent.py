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
# Dimensionality Reduction (clustering only — Option 1)
# ---------------------------------------------------------------------------

def _write_wide_clustering_csv(tmp_path, n_features=80, n_rows=60):
    """Wide numeric dataset for clustering (no target column present)."""
    csv_file = tmp_path / "wide_clustering.csv"
    rng = np.random.default_rng(7)
    data = {
        f"feat_{i}": rng.normal(0, 1, size=n_rows) + (2.0 * i / n_features)
        for i in range(n_features)
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return str(csv_file)


def test_preprocessing_dim_reduction_applied_for_clustering(tmp_path, monkeypatch):
    """
    Clustering task → features are reduced to <= min(50, n_features) components,
    columns renamed pc_N, and a dimension_reduction entry is logged.
    """
    monkeypatch.delenv("PREPROCESS_DIM_REDUCTION_COMPONENTS", raising=False)
    monkeypatch.delenv("PREPROCESS_DIM_REDUCTION", raising=False)

    state: AgentMLState = {
        "raw_file_path": _write_wide_clustering_csv(tmp_path),
        "target_column": None,
        "task_type": "clustering",
        "errors": [],
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    pc_cols = [c for c in df_clean.columns if c.startswith("pc_")]
    assert len(pc_cols) > 0
    assert len(pc_cols) <= 50
    assert len(pc_cols) < 80  # reduced vs original 80 engineered features

    log = result["preprocessing_log"]
    red = [e for e in log if e["operation"] == "dimension_reduction"]
    assert len(red) == 1
    assert "80 features" in red[0]["details"]
    assert f"→ {len(pc_cols)} components" in red[0]["details"]


def test_preprocessing_dim_reduction_uses_env_component_override(tmp_path, monkeypatch):
    """
    PREPROCESS_DIM_REDUCTION_COMPONENTS overrides the default component count.
    """
    monkeypatch.setenv("PREPROCESS_DIM_REDUCTION_COMPONENTS", "5")
    monkeypatch.delenv("PREPROCESS_DIM_REDUCTION", raising=False)

    state: AgentMLState = {
        "raw_file_path": _write_wide_clustering_csv(tmp_path),
        "target_column": None,
        "task_type": "clustering",
        "errors": [],
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    pc_cols = [c for c in df_clean.columns if c.startswith("pc_")]
    assert len(pc_cols) == 5

    red = [e for e in result["preprocessing_log"] if e["operation"] == "dimension_reduction"]
    assert len(red) == 1
    assert "5 components" in red[0]["details"]


def test_preprocessing_dim_reduction_off_env(tmp_path, monkeypatch):
    """
    PREPROCESS_DIM_REDUCTION=off disables reduction entirely — raw features kept.
    """
    monkeypatch.setenv("PREPROCESS_DIM_REDUCTION", "off")

    state: AgentMLState = {
        "raw_file_path": _write_wide_clustering_csv(tmp_path),
        "target_column": None,
        "task_type": "clustering",
        "errors": [],
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    assert not any(c.startswith("pc_") for c in df_clean.columns)
    assert len(df_clean.columns) == 80
    assert not any(
        e["operation"] == "dimension_reduction"
        for e in result["preprocessing_log"]
    )


def test_preprocessing_dim_reduction_untouched_for_supervised(tmp_path, monkeypatch):
    """
    Supervised (classification/regression) tasks are NEVER reduced — raw,
    interpretable feature columns must be preserved.
    """
    monkeypatch.delenv("PREPROCESS_DIM_REDUCTION_COMPONENTS", raising=False)
    monkeypatch.delenv("PREPROCESS_DIM_REDUCTION", raising=False)

    csv_file = tmp_path / "supervised.csv"
    rng = np.random.default_rng(3)
    sqft = rng.uniform(500, 3500, size=30)
    price = sqft * 250 + rng.normal(0, 10000, size=30)
    pd.DataFrame({"sqft": sqft, "rooms": rng.integers(1, 6, size=30), "price": price}).to_csv(
        csv_file, index=False
    )

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": "price",
        "task_type": "regression",
        "errors": [],
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    assert "sqft" in df_clean.columns
    assert "rooms" in df_clean.columns
    assert "price" in df_clean.columns
    assert not any(c.startswith("pc_") for c in df_clean.columns)
    assert not any(
        e["operation"] == "dimension_reduction"
        for e in result["preprocessing_log"]
    )

    # Also: an existing supervised call that omits task_type entirely (older
    # direct-call tests) must behave identically — no reduction.
    del state["task_type"]
    result2 = preprocessing_agent(state)
    df_clean2 = pd.read_csv(result2["clean_dataset_path"])
    assert "sqft" in df_clean2.columns
    assert not any(c.startswith("pc_") for c in df_clean2.columns)


def test_preprocessing_dim_reduction_skipped_for_small_feature_sets(tmp_path, monkeypatch):
    """
    Clustering with only 1 feature (or 1 row) skips reduction without erroring.
    """
    monkeypatch.delenv("PREPROCESS_DIM_REDUCTION_COMPONENTS", raising=False)

    csv_file = tmp_path / "single.csv"
    pd.DataFrame({"only_feature": np.arange(20.0)}).to_csv(csv_file, index=False)

    state: AgentMLState = {
        "raw_file_path": str(csv_file),
        "target_column": None,
        "task_type": "clustering",
        "errors": [],
    }

    result = preprocessing_agent(state)
    df_clean = pd.read_csv(result["clean_dataset_path"])

    assert "only_feature" in df_clean.columns
    assert not any(c.startswith("pc_") for c in df_clean.columns)
    assert not any(e.get("recoverable") is False for e in result["errors"])

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
