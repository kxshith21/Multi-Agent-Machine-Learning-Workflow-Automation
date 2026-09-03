"""
Safe restricted formula evaluator for user-supplied feature-engineering formulas.

CRITICAL SAFETY REQUIREMENT: user-provided custom formulas are NEVER evaluated
with Python's raw eval()/exec(). This module implements a restricted parser that
accepts only:

  - arithmetic operators:  +  -  *  /
  - parentheses  (  )
  - numeric literals
  - whitespace
  - references to existing dataframe column names (identifiers must resolve to
    a column known to exist)

Everything else — function calls, imports, attribute access (dots), brackets,
string literals, assignments, semicolons, comments, and identifiers that do not
match a known column — is rejected with a FormulaValidationError. Nothing is
ever executed unless it passes validation.

The safe evaluation walks only a fixed subset of Python's AST node types
(BinOp, UnaryOp, Constant, Name) after validating that every Name resolves to a
real column. This is a provably-safe subset and does not touch eval().
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any, Callable, Optional, Set

import pandas as pd


class FormulaValidationError(ValueError):
    """Raised when a formula is unsafe or references an unknown column."""


# Token regexes used by the lightweight pre-flight scan. Anything that isn't
# matched here is rejected outright before we ever build an AST.
_ALLOWED_ATOM = re.compile(
    r"^\s*(?:[0-9]+(?:\.[0-9]*)?|\.?[0-9]+)\s*$"  # numeric literal
)
_ALLOWED_OPERATOR = set("+-*/()")


def _is_numeric_literal(tok: str) -> bool:
    return bool(_ALLOWED_ATOM.match(tok))


def validate_formula(formula: str, columns: Set[str]) -> None:
    """
    Validate that ``formula`` is safe to evaluate against the given column set.

    Raises FormulaValidationError if the formula references unknown columns,
    uses disallowed syntax (function calls, attribute access, imports, etc.),
    or is otherwise not a plain arithmetic expression.

    Args:
        formula: The raw user-supplied formula string.
        columns: Set of existing dataframe column names that may be referenced.

    Raises:
        FormulaValidationError: If the formula is unsafe or invalid.
    """
    if not isinstance(formula, str) or not formula.strip():
        raise FormulaValidationError("Formula is empty. Provide an expression like 'price / sqft'.")

    text = formula.strip()

    # ---- 1. Character-level whitelist ---------------------------------------
    # Disallow structural characters that enable code execution.
    for ch in text:
        if ch.isalnum() or ch.isspace() or ch in _ALLOWED_OPERATOR or ch in "._":
            continue
        raise FormulaValidationError(
            f"Formula contains a disallowed character {ch!r}. Only + - * / ( ) and column names are allowed."
        )

    # ---- 2. Parse into an AST but ONLY to inspect/gate it (never eval) ------
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise FormulaValidationError(f"Formula is not a valid arithmetic expression: {exc}") from exc

    # ---- 3. Walk the tree with a strict allow-list of node types -------------
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
    )
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise FormulaValidationError(
                f"Formula uses unsupported syntax ({type(node).__name__}). "
                "Only + - * /, parentheses, numbers, and existing column names are allowed."
            )
        if isinstance(node, ast.BinOp) and type(node.op) not in allowed_operators:
            raise FormulaValidationError("Formula uses an unsupported operator. Only + - * / are allowed.")
        if isinstance(node, ast.Name):
            name = node.id
            if name not in columns:
                raise FormulaValidationError(
                    f"Formula references unknown column '{name}'. "
                    f"Available columns: {sorted(columns)}"
                )

    return None


def evaluate_formula(formula: str, columns: Set[str], df: pd.DataFrame) -> pd.Series:
    """
    Validate ``formula`` and evaluate it safely against ``df``.

    Raises FormulaValidationError for unsafe/invalid formulas; on success returns
    a pandas Series of the computed values.

    Args:
        formula: The user-supplied formula string.
        columns: Set of existing dataframe column names (used for validation).
        df: The dataframe whose columns the formula may reference.

    Returns:
        A pandas Series with one value per row of ``df``.

    Raises:
        FormulaValidationError: If the formula is unsafe or references unknown
            columns, or if evaluation fails for a non-arithmetic reason.
    """
    validate_formula(formula, columns)

    try:
        tree = ast.parse(formula.strip(), mode="eval")
    except SyntaxError as exc:
        raise FormulaValidationError(f"Formula is not a valid arithmetic expression: {exc}") from exc

    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def _safe_eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant):
            # Only allow numeric constants.
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise FormulaValidationError("Formula constant must be a number.")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in df.columns:
                raise FormulaValidationError(f"Formula references unknown column '{node.id}'.")
            return df[node.id]
        if isinstance(node, ast.BinOp):
            op_fn = ops.get(type(node.op))
            if op_fn is None:
                raise FormulaValidationError("Formula uses an unsupported operator. Only + - * / are allowed.")
            left = _safe_eval(node.left)
            right = _safe_eval(node.right)
            try:
                return op_fn(left, right)
            except ZeroDivisionError:
                raise FormulaValidationError("Formula divides by zero.")
        if isinstance(node, ast.UnaryOp):
            operand = _safe_eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise FormulaValidationError("Formula uses an unsupported unary operator.")
        raise FormulaValidationError(
            f"Formula uses unsupported syntax ({type(node).__name__}). Only + - * / are allowed."
        )

    result = _safe_eval(tree)
    if isinstance(result, pd.Series):
        return result
    # Scalar result (no column references) — broadcast to a full-length series.
    return pd.Series([result] * len(df), index=df.index)
