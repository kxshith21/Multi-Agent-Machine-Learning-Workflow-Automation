"""
Regression model zoo.

Fixed zoo (Rules.md §1, §7) — scikit-learn regressors + XGBoost.
No Optuna/Hyperopt, no deep learning, no auto-scaling infra.

Each ModelSpec expands to ONE experiment run (param_grid applied as-is).
"""

from __future__ import annotations

from typing import List

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.model_zoo import ModelSpec


def get_models() -> List[ModelSpec]:
    """
    Return the fixed regression zoo.

    Includes a baseline (DummyRegressor — predicts mean) so the leaderboard
    always has something to beat.
    """
    return [
        # 1. Baseline
        ModelSpec(
            name="DummyRegressor_mean",
            family="baseline",
            estimator=DummyRegressor(strategy="mean"),
            param_grid={},
        ),
        # 2. Linear
        ModelSpec(
            name="LinearRegression_default",
            family="linear",
            estimator=LinearRegression(),
            param_grid={},
        ),
        # 3. Ridge
        ModelSpec(
            name="Ridge_alpha1",
            family="linear",
            estimator=Ridge(alpha=1.0, random_state=42),
            param_grid={},
        ),
        # 4. KNN
        ModelSpec(
            name="KNeighborsRegressor_k5",
            family="instance",
            estimator=KNeighborsRegressor(n_neighbors=5),
            param_grid={},
        ),
        # 5. Random Forest
        ModelSpec(
            name="RandomForestRegressor_n100",
            family="tree",
            estimator=RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=1
            ),
            param_grid={},
        ),
        # 6. Gradient Boosting (sklearn)
        ModelSpec(
            name="GradientBoostingRegressor_n100",
            family="boosting",
            estimator=GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=42
            ),
            param_grid={},
        ),
        # 7. XGBoost
        ModelSpec(
            name="XGBRegressor_n100",
            family="boosting",
            estimator=XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
            ),
            param_grid={},
        ),
    ]
