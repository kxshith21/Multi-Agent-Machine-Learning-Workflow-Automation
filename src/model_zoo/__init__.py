"""
src.model_zoo — fixed model zoos per task type.

Per Architecture.md §4 and Rules.md §1:
- scikit-learn + xgboost only.
- Fixed hyperparameter grids (no Optuna/Hyperopt).
- Uniform interface across task types so the experiment orchestrator
  can iterate without knowing which zoo it's calling.

Each zoo module exposes `get_models()` returning a list of
`ModelSpec` TypedDict objects with the shape:

    {
        "name": str,                 # human-readable model identifier
        "family": str,               # "linear" | "tree" | "boosting" | ...
        "estimator": BaseEstimator,  # scikit-learn-compatible estimator
        "param_grid": dict,          # fixed grid; one entry == one experiment run
    }

The orchestrator iterates the list and runs every (estimator, param_grid) combo.
A combo with N dict-keys expands to cartesian(N) runs only if `expand_grid=True`;
otherwise the param_grid is applied as-is for a single run. For v1 we treat each
param_grid as a single experiment (no cartesian expansion) to keep scope tight
(Phases.md Phase 4 "fixed grids").
"""

from __future__ import annotations

from typing_extensions import TypedDict


class ModelSpec(TypedDict):
    """
    Uniform model-zoo entry consumed by the experiment orchestrator.

    Attributes:
        name:       Human-readable model identifier; used in `experiment_results`.
        family:     Loose family tag ("linear" | "tree" | "boosting" |
                    "centroid" | "density" | ...). Useful for grouping in
                    the final report; not used by the orchestrator for
                    dispatch logic.
        estimator:  scikit-learn-compatible estimator (must implement
                    `.fit()` and `.predict()`). XGBoost classifiers/
                    regressors satisfy this contract.
        param_grid: Fixed hyperparameter set for THIS run (dict). v1
                    applies the grid as-is for one experiment; cartesian
                    expansion is a v2 concern (Rules.md §7).
    """
    name: str
    family: str
    estimator: object
    param_grid: dict
