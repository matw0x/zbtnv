import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from meshguard.detection.alerting import SignalGate
from meshguard.detection.datasets import LabeledRun
from meshguard.detection.detector import AnomalyDetector
from meshguard.detection.features import RouteInventory, WindowFeatures
from meshguard.detection.scorers import Scorer
from meshguard.domain import AnomalySignal


def by_window(windows: Sequence[WindowFeatures]) -> Iterator[list[WindowFeatures]]:
    for _, group in groupby(windows, key=lambda w: w.start):
        yield list(group)


def replay(detector: AnomalyDetector, windows: Sequence[WindowFeatures]) -> list[AnomalySignal]:
    gate = SignalGate()
    observed = (gate.observe(detector.detect_many(group)) for group in by_window(windows))
    return [signal for signal in observed if signal is not None]


@dataclass(frozen=True, slots=True)
class ScorerReport:
    name: str
    threshold: float
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    false_alarms_per_hour: float
    incident_recall: dict[str, float]
    detection_delay_s: dict[str, float]
    fit_seconds: float
    score_us_per_window: float


@dataclass(slots=True)
class Evaluation:
    reports: list[ScorerReport] = field(default_factory=list)
    curves: dict[str, tuple[NDArray[np.bool_], NDArray[np.float64]]] = field(default_factory=dict)
    detectors: dict[str, AnomalyDetector] = field(default_factory=dict)


def evaluate_scorers(
    train: Sequence[WindowFeatures],
    inventory: RouteInventory,
    runs: Sequence[LabeledRun],
    clean: Sequence[WindowFeatures],
    scorers: Sequence[Scorer],
    *,
    target_fpr: float = 0.002,
) -> Evaluation:
    result = Evaluation()
    test_windows = [w for run in runs for w in run.windows]
    y_true = np.concatenate([run.labels for run in runs])
    clean_hours = len(list(by_window(clean))) * (clean[0].end - clean[0].start).seconds / 3600

    for scorer in scorers:
        started = time.perf_counter()
        detector = AnomalyDetector.fit(train, inventory, scorer=scorer, target_fpr=target_fpr)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        y_score = detector.score_many(test_windows)
        per_window_us = (time.perf_counter() - started) / len(test_windows) * 1e6
        y_pred = y_score > detector.threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        hits: defaultdict[str, list[bool]] = defaultdict(list)
        delays: defaultdict[str, list[float]] = defaultdict(list)
        for run in runs:
            delay = first_detection_delay(run, replay(detector, run.windows))
            hits[run.scenario.name].append(delay is not None)
            if delay is not None:
                delays[run.scenario.name].append(delay)

        result.reports.append(
            ScorerReport(
                name=scorer.name,
                threshold=detector.threshold,
                roc_auc=float(roc_auc_score(y_true, y_score)),
                pr_auc=float(average_precision_score(y_true, y_score)),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                false_alarms_per_hour=len(replay(detector, clean)) / clean_hours,
                incident_recall={k: float(np.mean(v)) for k, v in hits.items()},
                detection_delay_s={k: float(np.mean(v)) for k, v in delays.items()},
                fit_seconds=fit_seconds,
                score_us_per_window=per_window_us,
            )
        )
        result.curves[scorer.name] = (y_true, y_score)
        result.detectors[scorer.name] = detector
    return result


def first_detection_delay(run: LabeledRun, signals: Sequence[AnomalySignal]) -> float | None:
    scenario = run.scenario
    for signal in signals:
        relevant = set(signal.services) & scenario.affected_services
        if relevant and scenario.start <= signal.window_end <= scenario.end:
            return (signal.window_end - scenario.start).total_seconds()
    return None
