from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import sklearn

from meshguard.detection.features import FEATURE_NAMES, FEATURES, RouteInventory, WindowFeatures
from meshguard.detection.scorers import Matrix, OneClassSvmScorer, Scorer, Vector
from meshguard.domain import FeatureDeviation

ARTIFACT_VERSION = 1
Z_CLIP = 50.0
_LOG_MASK = np.array([f.log_scale for f in FEATURES])
_MIN_SCALE = np.array([f.min_scale for f in FEATURES])


def to_model_space(values: Sequence[float] | Vector) -> Vector:
    x = np.asarray(values, dtype=np.float64)
    return np.where(_LOG_MASK, np.log1p(np.maximum(x, 0.0)), x)


def from_model_space(x: Vector) -> Vector:
    return np.where(_LOG_MASK, np.expm1(x), x)


class BaselineProfile:
    """Per-service robust location and scale of every feature in model space."""

    def __init__(self, center: dict[str, Vector], scale: dict[str, Vector]) -> None:
        self.center = center
        self.scale = scale

    @classmethod
    def fit(cls, windows: Sequence[WindowFeatures]) -> Self:
        grouped: defaultdict[str, list[Vector]] = defaultdict(list)
        for window in windows:
            grouped[window.service].append(to_model_space(window.values))
        center, scale = {}, {}
        for service, rows in grouped.items():
            x = np.vstack(rows)
            q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0)
            center[service] = q50
            scale[service] = np.maximum((q75 - q25) / 1.349, _MIN_SCALE)
        return cls(center, scale)

    @property
    def services(self) -> tuple[str, ...]:
        return tuple(sorted(self.center))

    def standardize(self, window: WindowFeatures) -> Vector:
        z = (to_model_space(window.values) - self.center[window.service]) / self.scale[
            window.service
        ]
        return np.clip(z, -Z_CLIP, Z_CLIP)

    def standardize_many(self, windows: Sequence[WindowFeatures]) -> Matrix:
        return np.vstack([self.standardize(w) for w in windows])

    def typical(self, service: str) -> dict[str, float]:
        natural = from_model_space(self.center[service])
        return dict(zip(FEATURE_NAMES, natural.tolist(), strict=True))


@dataclass(frozen=True, slots=True)
class Detection:
    window: WindowFeatures
    score: float
    threshold: float
    deviations: tuple[FeatureDeviation, ...]

    @property
    def is_anomaly(self) -> bool:
        return self.score > self.threshold


class AnomalyDetector:
    """Statistical first stage: flags service windows that deviate from learned behaviour."""

    def __init__(
        self,
        profile: BaselineProfile,
        scorer: Scorer,
        threshold: float,
        inventory: RouteInventory,
        window_size: timedelta,
    ) -> None:
        self.profile = profile
        self.scorer = scorer
        self.threshold = threshold
        self.inventory = inventory
        self.window_size = window_size

    @classmethod
    def fit(
        cls,
        windows: Sequence[WindowFeatures],
        inventory: RouteInventory,
        *,
        scorer: Scorer | None = None,
        target_fpr: float = 0.002,
        calibration_fraction: float = 0.3,
    ) -> Self:
        """Fit on chronologically earlier windows, calibrate the threshold on later ones."""
        if not windows:
            raise ValueError("cannot fit a detector on an empty training set")
        ordered = sorted(windows, key=lambda w: w.start)
        cut = ordered[int(len(ordered) * (1 - calibration_fraction))].start
        fit_part = [w for w in ordered if w.start < cut]
        calibration = [w for w in ordered if w.start >= cut]
        profile = BaselineProfile.fit(fit_part)
        scorer = (scorer or OneClassSvmScorer()).fit(profile.standardize_many(fit_part))
        scores = scorer.score(profile.standardize_many(calibration))
        threshold = float(np.quantile(scores, 1 - target_fpr))
        return cls(
            profile, scorer, threshold, inventory.freeze(), ordered[0].end - ordered[0].start
        )

    def score_many(self, windows: Sequence[WindowFeatures]) -> Vector:
        return self.scorer.score(self.profile.standardize_many(windows))

    def detect(self, window: WindowFeatures, top_k: int = 5) -> Detection:
        return self.detect_many([window], top_k)[0]

    def detect_many(self, windows: Sequence[WindowFeatures], top_k: int = 5) -> list[Detection]:
        z = self.profile.standardize_many(windows)
        scores = self.scorer.score(z)
        return [
            Detection(
                window,
                float(score),
                self.threshold,
                self.explain(window, row, top_k) if score > self.threshold else (),
            )
            for window, row, score in zip(windows, z, scores, strict=True)
        ]

    def explain(
        self, window: WindowFeatures, z: Vector, top_k: int = 5
    ) -> tuple[FeatureDeviation, ...]:
        typical = self.profile.typical(window.service)
        order = np.argsort(-np.abs(z))
        strong = [i for i in order if abs(z[i]) >= 3.0] or list(order[:3])
        return tuple(
            FeatureDeviation(
                feature=FEATURE_NAMES[i],
                value=window.values[i],
                baseline=typical[FEATURE_NAMES[i]],
                z_score=float(z[i]),
            )
            for i in strong[:top_k]
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "version": ARTIFACT_VERSION,
                "sklearn": sklearn.__version__,
                "features": FEATURE_NAMES,
                "detector": self,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load an artifact produced by :meth:`save` (a pickle: never load untrusted files)."""
        payload = joblib.load(path)
        if payload.get("version") != ARTIFACT_VERSION or payload.get("features") != FEATURE_NAMES:
            raise ValueError(f"{path} was produced by an incompatible meshguard version")
        if payload.get("sklearn") != sklearn.__version__:
            raise ValueError(
                f"{path} was trained with scikit-learn {payload.get('sklearn')}, "
                f"runtime has {sklearn.__version__}; retrain the detector"
            )
        detector = payload["detector"]
        if not isinstance(detector, cls):
            raise TypeError(f"{path} does not contain an {cls.__name__}")
        return detector
