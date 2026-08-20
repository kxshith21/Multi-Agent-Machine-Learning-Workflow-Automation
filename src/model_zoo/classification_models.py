"""
Classification model zoo.

Fixed zoo (Rules.md §1, §7) — scikit-learn classifiers + XGBoost.
No Optuna/Hyperopt, no deep learning, no auto-scaling infra.

Each ModelSpec expands to ONE experiment run (param_grid applied as-is).
"""

from __future__ import annotations

from typing import List

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.model_zoo import ModelSpec


def get_models() -> List[ModelSpec]:
    """
    Return the fixed classification zoo.

    Includes a baseline (DummyClassifier — predicts majority class) so the
    leaderboard always has something to beat, even on pathological datasets.
    """
    return [
        # 1. Baseline
        ModelSpec(
            name="DummyClassifier_baseline",
            family="baseline",
            estimator=DummyClassifier(strategy="most_frequent"),
            param_grid={},
        ),
        # 2. Linear
        ModelSpec(
            name="LogisticRegression_default",
            family="linear",
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            param_grid={},
        ),
        # 3. KNN
        ModelSpec(
            name="KNeighborsClassifier_k5",
            family="instance",
            estimator=KNeighborsClassifier(n_neighbors=5),
            param_grid={},
        ),
        # 4. Random Forest
        ModelSpec(
            name="RandomForestClassifier_n100",
            family="tree",
            estimator=RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=1
            ),
            param_grid={},
        ),
        # 5. Gradient Boosting (sklearn)
        ModelSpec(
            name="GradientBoostingClassifier_n100",
            family="boosting",
            estimator=GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            ),
            param_grid={},
        ),
        # 6. XGBoost
        ModelSpec(
            name="XGBClassifier_n100",
            family="boosting",
            estimator=XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
            ),
            param_grid={},
        ),
    ]
