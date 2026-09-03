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
from src.utils.safe_formula import FormulaValidationError, evaluate_formula

logger = logging.getLogger(__name__)


def _apply_selected_features(
    df: pd.DataFrame,
    selected_features: list,
    suggestions_by_name: dict,
    feature_engineering_log: list,
) -> None:
    """
    Deterministically create each user-selected suggested feature from its recipe.

    Recipes (from profiling's suggestions):
      - datetime_decompose : split a datetime column into year/month/day/weekday
      - ratio / product     : new = col_a / col_b  (ratio)  or  col_a * col_b
      - binning             : bucket a high-cardinality categorical column

    Each created column is logged with source="suggested" and its reasoning.
    Skipped/unknown suggestions are logged as recoverable, non-blocking.
    """
    for name in selected_features:
        sug = suggestions_by_name.get(name)
        if sug is None:
            feature_engineering_log.append({
                "column": str(name),
                "operation": "skip",
                "source": "suggested",
                "details": f"Suggestion '{name}' was not found among detected suggestions.",
                "reasoning": "Unknown suggested feature — nothing was created.",
            })
            continue

        ftype = sug.get("type")
        cols = sug.get("columns") or []
        try:
            if ftype == "datetime_decompose":
                col = cols[0]
                parsed = pd.to_datetime(df[col], errors="coerce")
                for part in ("year", "month", "day", "weekday"):
                    new_col = f"{col}_{part}"
                    df[new_col] = getattr(parsed.dt, part)
                    feature_engineering_log.append({
                        "column": new_col,
                        "operation": "create",
                        "source": "suggested",
                        "details": f"Decomposed datetime '{col}' into '{new_col}'.",
                        "reasoning": f"Exposing the {part} of a datetime helps the model learn time-based patterns.",
                    })
            elif ftype in ("ratio", "product"):
                a, b = cols[0], cols[1]
                if ftype == "ratio":
                    df[name] = df[a] / df[b]
                    expr = f"{a} / {b}"
                else:
                    df[name] = df[a] * df[b]
                    expr = f"{a} * {b}"
                feature_engineering_log.append({
                    "column": name,
                    "operation": "create",
                    "source": "suggested",
                    "details": f"Created '{name}' as '{expr}'.",
                    "reasoning": (
                        f"'{a}' and '{b}' are strongly correlated; combining them "
                        "captures their relationship as a single feature."
                    ),
                })
            elif ftype == "binning":
                col = cols[0]
                nunique = df[col].nunique(dropna=True)
                # Bucket into a modest number of ordered groups by rank.
                df[name] = pd.qcut(
                    df[col].astype("category").cat.codes,
                    q=min(10, nunique),
                    labels=False,
                    duplicates="drop",
                )
                feature_engineering_log.append({
                    "column": name,
                    "operation": "create",
                    "source": "suggested",
                    "details": f"Binned categorical '{col}' into ordered groups.",
                    "reasoning": (
                        f"'{col}' has high cardinality ({nunique} values); binning "
                        "reduces sparsity relative to one-hot encoding every level."
                    ),
                })
            else:
                feature_engineering_log.append({
                    "column": str(name),
                    "operation": "skip",
                    "source": "suggested",
                    "details": f"Suggestion type '{ftype}' is not supported.",
                    "reasoning": "Unsupported recipe — nothing was created.",
                })
        except Exception as exc:  # noqa: BLE001
            feature_engineering_log.append({
                "column": str(name),
                "operation": "skip",
                "source": "suggested",
                "details": f"Failed to create suggested feature '{name}': {exc}",
                "reasoning": "Suggested feature could not be computed from the data.",
            })


def _apply_custom_features(
    df: pd.DataFrame,
    custom_features: list,
    feature_engineering_log: list,
    errors_to_report: list,
) -> None:
    """
    Deterministically evaluate each user-supplied custom formula with the SAFE
    restricted evaluator — never eval(). Malicious/unsafe formulas are rejected
    (recoverable error) and never executed.
    """
    columns = set(df.columns)
    for cf in custom_features or []:
        name = cf.get("name") if isinstance(cf, dict) else None
        formula = cf.get("formula") if isinstance(cf, dict) else None
        if not name or not formula:
            feature_engineering_log.append({
                "column": str(name or "?"),
                "operation": "skip",
                "source": "custom",
                "details": "Custom feature missing a name or formula.",
                "reasoning": "Incomplete custom feature definition — nothing was created.",
            })
            continue
        if str(name) in df.columns:
            feature_engineering_log.append({
                "column": str(name),
                "operation": "skip",
                "source": "custom",
                "details": f"Custom column '{name}' already exists.",
                "reasoning": "Refusing to overwrite an existing column.",
            })
            continue
        try:
            series = evaluate_formula(formula, columns, df)
            df[str(name)] = series
            feature_engineering_log.append({
                "column": str(name),
                "operation": "create",
                "source": "custom",
                "formula": formula,
                "details": f"Created custom column '{name}' from formula '{formula}'.",
                "reasoning": "Custom feature requested by the user at the feature-engineering checkpoint.",
            })
        except FormulaValidationError as exc:
            feature_engineering_log.append({
                "column": str(name),
                "operation": "skip",
                "source": "custom",
                "formula": formula,
                "details": f"Custom formula for '{name}' was REJECTED: {exc}",
                "reasoning": "Formula was rejected by the restricted safe evaluator. Nothing was executed.",
            })
            errors_to_report.append({
                "phase": "data_preprocessing",
                "error_type": "custom_formula_rejected",
                "message": f"Custom column '{name}': {exc}",
                "recoverable": True,
            })
        except Exception as exc:  # noqa: BLE001
            feature_engineering_log.append({
                "column": str(name),
                "operation": "skip",
                "source": "custom",
                "formula": formula,
                "details": f"Failed to evaluate custom formula for '{name}': {exc}",
                "reasoning": "Custom feature could not be computed from the data.",
            })


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
    feature_engineering_log: list[dict[str, Any]] = list(state.get("feature_engineering_log") or [])

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

    # 1.5 Feature Engineering Selection (Phase 10)
    # Deterministically apply the features the user selected at the checkpoint.
    # Suggestions are applied from their recipe; custom features are evaluated
    # through the restricted safe_formula parser (never raw eval()).
    selected_features = state.get("selected_features") or []
    custom_features = state.get("custom_features") or []
    if selected_features or custom_features:
        suggestions_by_name = {}
        for sug in state.get("feature_suggestions") or []:
            suggestions_by_name[sug.get("name")] = sug
        _apply_selected_features(df, selected_features, suggestions_by_name, feature_engineering_log)
        _apply_custom_features(df, custom_features, feature_engineering_log, errors_to_report)
        logger.info(
            f"Feature engineering applied: {len(selected_features)} selected, "
            f"{len(custom_features)} custom → {len(feature_engineering_log)} log entries"
        )

    # Get profiling info from state, or compute dynamically if missing
    profile = state.get("dataset_profile") or {}
    numeric_cols = list(profile.get("numeric_cols") or [])
    categorical_cols = list(profile.get("categorical_cols") or [])

    # Any columns in the (post-engineering) dataframe that aren't already covered
    # by the profile lists are newly-created feature columns — classify them.
    known = set(numeric_cols) | set(categorical_cols) | ({target_col} if target_col else set())
    for col in df.columns:
        if col in known or col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    # If profile is missing, detect types dynamically (excluding target_col)
    if not numeric_cols and not categorical_cols:
        # Detect over ALL columns present now (including features we just added).
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

    # 1.5 Drop columns with 100% missing values
    non_missing_numeric = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        missing_count = df[col].isna().sum()
        if missing_count == len(df):
            preprocessing_log.append({
                "column": col,
                "operation": "drop",
                "details": f"Dropped column '{col}' because it is 100% missing values.",
                "reasoning": "Columns with no observed values cannot be imputed or scaled, and provide no information to machine learning models."
            })
            errors_to_report.append({
                "phase": "data_preprocessing",
                "error_type": "column_all_missing",
                "message": f"Column '{col}' was dropped because it contains only missing values.",
                "recoverable": True
            })
            logger.info(f"Dropped 100% missing column: {col}")
        else:
            non_missing_numeric.append(col)
    numeric_cols = non_missing_numeric

    non_missing_categorical = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        missing_count = df[col].isna().sum()
        if missing_count == len(df):
            preprocessing_log.append({
                "column": col,
                "operation": "drop",
                "details": f"Dropped column '{col}' because it is 100% missing values.",
                "reasoning": "Columns with no observed values cannot be imputed or scaled, and provide no information to machine learning models."
            })
            errors_to_report.append({
                "phase": "data_preprocessing",
                "error_type": "column_all_missing",
                "message": f"Column '{col}' was dropped because it contains only missing values.",
                "recoverable": True
            })
            logger.info(f"Dropped 100% missing column: {col}")
        else:
            non_missing_categorical.append(col)
    categorical_cols = non_missing_categorical


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

    # Ensure we actually have at least 1 feature left for training
    feature_cols = [c for c in df_processed.columns if c != target_col]
    if not feature_cols:
        error_msg = "Preprocessing resulted in a dataset with 0 features. Cannot proceed with model training."
        logger.error(error_msg)
        errors_to_report.append({
            "phase": "data_preprocessing",
            "error_type": "zero_features_error",
            "message": error_msg,
            "recoverable": False
        })
        return {
            "clean_dataset_path": "",
            "preprocessing_log": preprocessing_log,
            "feature_engineering_log": feature_engineering_log,
            "errors": errors_to_report,
            "status": "failed"
        }

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
        "feature_engineering_log": feature_engineering_log,
        "errors": errors_to_report
    }
