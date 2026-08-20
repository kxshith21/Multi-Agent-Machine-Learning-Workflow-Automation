"""
Agent 2: Data Preprocessing Agent

Cleans dataset by deduplicating, dropping unencodable free-text columns,
imputing missing values, encoding categorical variables, and scaling numeric variables.
Matches Architecture.md §2 and Phases.md Phase 2 specifications.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Literal
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.orchestrator.state import AgentMLState, ErrorEntry

logger = logging.getLogger(__name__)


def preprocessing_agent(state: AgentMLState) -> dict[str, Any]:
    """
    Cleans the raw CSV file, applies transformations, logs details, and saves the cleaned dataset.

    Args:
        state: The current AgentMLState.

    Returns:
        A dict containing the updated clean_dataset_path, preprocessing_log, and errors.
    """
    raw_file_path = state.get("raw_file_path")
    target_col = state.get("target_column")
    errors_to_report = list(state.get("errors", []))
    preprocessing_log: list[dict[str, Any]] = []

    if not raw_file_path:
        error_msg = "No raw_file_path provided for preprocessing."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Preprocessing starting for: {raw_file_path}")

    # Load dataset
    try:
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        error_msg = f"Preprocessing failed to read CSV: {str(e)}"
        logger.error(error_msg)
        raise type(e)(error_msg) from e

    # 1. Deduplication
    initial_row_count = len(df)
    df = df.drop_duplicates()
    final_row_count = len(df)
    duplicates_removed = initial_row_count - final_row_count

    if duplicates_removed > 0:
        log_entry = {
            "column": "dataset",
            "operation": "deduplication",
            "details": f"Removed {duplicates_removed} duplicate rows.",
            "reasoning": "Duplicate rows can bias model training and result in overly optimistic evaluation metrics."
        }
        preprocessing_log.append(log_entry)
        logger.info(f"Deduplication: removed {duplicates_removed} rows.")

    # Get profiling info from state, or compute dynamically if missing
    profile = state.get("dataset_profile") or {}
    numeric_cols = list(profile.get("numeric_cols") or [])
    categorical_cols = list(profile.get("categorical_cols") or [])

    # If profile is missing, detect types dynamically (excluding target_col)
    if not numeric_cols and not categorical_cols:
        for col in df.columns:
            if col == target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(str(col))
            else:
                categorical_cols.append(str(col))

    # Keep target column separate
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # 2. Drop unencodable free-text / high-cardinality columns
    remaining_categorical_cols = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        nunique = df[col].nunique()
        # High cardinality threshold: cardinality > 20 and unique values / total rows > 0.4
        if nunique > 20 and (nunique / len(df)) > 0.4:
            log_entry = {
                "column": col,
                "operation": "drop",
                "details": f"Dropped high-cardinality column '{col}' with {nunique} unique values.",
                "reasoning": (
                    "This column contains free text or unique identifiers rather than structured categorical values. "
                    "One-hot encoding this would produce a sparse matrix with too many features, causing overfitting."
                )
            }
            preprocessing_log.append(log_entry)
            errors_to_report.append({
                "phase": "data_preprocessing",
                "error_type": "column_dropped",
                "message": f"High-cardinality column '{col}' was dropped because it is unencodable.",
                "recoverable": True
            })
            logger.info(f"Dropped unencodable column: {col}")
        else:
            remaining_categorical_cols.append(col)

    # 3. Handle missing values, scaling and categorical encoding
    # Set up pipeline transformers
    transformers = []
    remaining_numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Impute and Scale numeric columns
    if remaining_numeric_cols:
        # Determine which columns have missing values to explain reasoning in log
        for col in remaining_numeric_cols:
            missing_count = df[col].isna().sum()
            impute_reasoning = ""
            if missing_count > 0:
                pct = (missing_count / len(df)) * 100.0
                impute_reason = f"Imputed {missing_count} missing values ({pct:.1f}%) using the column median."
                preprocessing_log.append({
                    "column": col,
                    "operation": "impute",
                    "details": impute_reason,
                    "reasoning": "Median imputation preserves the central tendency of numeric columns and is robust to outliers."
                })
            preprocessing_log.append({
                "column": col,
                "operation": "scale",
                "details": "Standardized numeric feature to zero mean and unit variance.",
                "reasoning": "Standard scaling ensures that numerical features contribute equally to models and improves optimization stability."
            })

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", num_pipeline, remaining_numeric_cols))

    # Impute and Encode categorical columns
    if remaining_categorical_cols:
        for col in remaining_categorical_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                pct = (missing_count / len(df)) * 100.0
                preprocessing_log.append({
                    "column": col,
                    "operation": "impute",
                    "details": f"Imputed {missing_count} missing values ({pct:.1f}%) using the column mode (most frequent).",
                    "reasoning": "Mode imputation fills categorical missingness using the most representative class."
                })
            preprocessing_log.append({
                "column": col,
                "operation": "onehot_encode",
                "details": f"One-hot encoded categorical levels (unique count: {df[col].nunique()}).",
                "reasoning": "Translates string category levels into numerical columns suitable for machine learning models."
            })

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("categorical", cat_pipeline, remaining_categorical_cols))

    # Fit and transform features
    X = df[remaining_numeric_cols + remaining_categorical_cols]
    
    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        processed_arr = preprocessor.fit_transform(X)
        feature_names_out = list(preprocessor.get_feature_names_out())
        
        # Clean up output feature names (remove numeric__ and categorical__)
        clean_feature_names = []
        for name in feature_names_out:
            if name.startswith("numeric__"):
                clean_feature_names.append(name[9:])
            elif name.startswith("categorical__"):
                clean_feature_names.append(name[13:])
            else:
                clean_feature_names.append(name)
                
        df_processed = pd.DataFrame(processed_arr, columns=clean_feature_names)
    else:
        # No features to preprocess
        df_processed = pd.DataFrame(index=df.index)

    # Re-attach target column (preserving values but resetting index to align with processed df)
    if target_col and target_col in df.columns:
        target_series = df[target_col].reset_index(drop=True)
        df_processed[target_col] = target_series

        # Check target missing values
        target_missing = target_series.isna().sum()
        if target_missing > 0:
            warn_msg = f"Target column '{target_col}' has {target_missing} missing values. These will be handled during modeling."
            logger.warning(warn_msg)
            preprocessing_log.append({
                "column": target_col,
                "operation": "impute",
                "details": f"Target column '{target_col}' has missing values.",
                "reasoning": "Target missingness cannot be safely imputed deterministically without introducing label bias; remaining steps must handle it."
            })

    # Ensure output directories exist
    os.makedirs("data", exist_ok=True)
    session_id = state.get("session_id") or "default"
    clean_dataset_path = f"data/clean_dataset_{session_id}.csv"
    
    # Save cleaned dataset
    try:
        df_processed.to_csv(clean_dataset_path, index=False)
        logger.info(f"Clean dataset saved to: {clean_dataset_path}")
    except Exception as e:
        error_msg = f"Failed to save clean dataset CSV: {str(e)}"
        logger.error(error_msg)
        raise type(e)(error_msg) from e

    return {
        "clean_dataset_path": clean_dataset_path,
        "preprocessing_log": preprocessing_log,
        "errors": errors_to_report
    }
