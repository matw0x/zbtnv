from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

type Matrix = NDArray[np.float64]
type Vector = NDArray[np.float64]


class Scorer(Protocol):
    """Unsupervised novelty scorer: fit on normal data, higher score means more anomalous."""

    name: str

    def fit(self, z: Matrix) -> Self: ...

    def score(self, z: Matrix) -> Vector: ...


class IsolationForestScorer:
    name = "Isolation Forest"

    def __init__(self, n_estimators: int = 300, max_samples: int = 512, seed: int = 0) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=-1
        )

    def fit(self, z: Matrix) -> Self:
        self.model.fit(z)
        return self

    def score(self, z: Matrix) -> Vector:
        return np.asarray(-self.model.score_samples(z), dtype=np.float64)


class LocalOutlierFactorScorer:
    name = "Local Outlier Factor"

    def __init__(self, n_neighbors: int = 60) -> None:
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)

    def fit(self, z: Matrix) -> Self:
        self.model.fit(z)
        return self

    def score(self, z: Matrix) -> Vector:
        return np.asarray(-self.model.score_samples(z), dtype=np.float64)


class OneClassSvmScorer:
    name = "One-Class SVM"

    def __init__(
        self,
        nu: float = 0.01,
        gamma: str | float = "scale",
        max_samples: int = 8_000,
        seed: int = 0,
    ) -> None:
        self.model = OneClassSVM(nu=nu, gamma=gamma)
        self.max_samples = max_samples
        self.seed = seed

    def fit(self, z: Matrix) -> Self:
        if len(z) > self.max_samples:
            rows = np.random.default_rng(self.seed).choice(len(z), self.max_samples, replace=False)
            z = z[rows]
        self.model.fit(z)
        return self

    def score(self, z: Matrix) -> Vector:
        return np.asarray(-self.model.score_samples(z), dtype=np.float64)


class MaxRobustZScorer:
    """Classical baseline: the largest absolute robust z-score across features."""

    name = "Robust z-score"

    def fit(self, z: Matrix) -> Self:  # noqa: ARG002
        return self

    def score(self, z: Matrix) -> Vector:
        return np.asarray(np.abs(z).max(axis=1), dtype=np.float64)


def default_scorers(seed: int = 0) -> list[Scorer]:
    """Each method at the hyper-parameters selected on validation seeds."""
    return [
        OneClassSvmScorer(seed=seed),
        IsolationForestScorer(n_estimators=500, max_samples=4096, seed=seed),
        LocalOutlierFactorScorer(n_neighbors=60),
        MaxRobustZScorer(),
    ]
