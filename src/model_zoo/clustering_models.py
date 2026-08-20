"""
Clustering model zoo.

Fixed zoo (Rules.md §1, §7) — scikit-learn clustering estimators.
Clustering is unsupervised so each model uses its own internal
hyperparameters (no target, no train/test split). We still time each run
and log silhouette score (if labels can be computed) as a quality metric.

Each ModelSpec expands to ONE experiment run.
"""

from __future__ import annotations

from typing import List

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from src.model_zoo import ModelSpec


def get_models() -> List[ModelSpec]:
    """
    Return the fixed clustering zoo.

    For DBSCAN we keep eps fixed at 0.5 / min_samples=5 — a deliberately
    conservative default that often surfaces as "no clusters formed" on
    tiny toy datasets, which is itself a useful data-quality signal.
    """
    return [
        # 1. KMeans (k=3 — small default, suitable for toy data)
        ModelSpec(
            name="KMeans_k3",
            family="centroid",
            estimator=KMeans(n_clusters=3, random_state=42, n_init=10),
            param_grid={},
        ),
        # 2. KMeans (k=5)
        ModelSpec(
            name="KMeans_k5",
            family="centroid",
            estimator=KMeans(n_clusters=5, random_state=42, n_init=10),
            param_grid={},
        ),
        # 3. Gaussian Mixture (k=3)
        ModelSpec(
            name="GaussianMixture_k3",
            family="distribution",
            estimator=GaussianMixture(n_components=3, random_state=42),
            param_grid={},
        ),
        # 4. Agglomerative (k=3, ward linkage)
        ModelSpec(
            name="AgglomerativeClustering_k3_ward",
            family="hierarchical",
            estimator=AgglomerativeClustering(n_clusters=3, linkage="ward"),
            param_grid={},
        ),
        # 5. DBSCAN (fixed eps/min_samples — frequently produces "no clusters"
        #    on small toy datasets, which is itself an informative result)
        ModelSpec(
            name="DBSCAN_eps0.5_min5",
            family="density",
            estimator=DBSCAN(eps=0.5, min_samples=5),
            param_grid={},
        ),
    ]
