from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from meshguard.detection.alerting import SignalGate
from meshguard.detection.datasets import baseline_windows, scenario_run
from meshguard.detection.detector import AnomalyDetector, Detection
from meshguard.detection.evaluation import first_detection_delay, replay
from meshguard.detection.features import (
    FEATURE_NAMES,
    RouteInventory,
    WindowAggregator,
    WindowFeatures,
)
from meshguard.domain import EGRESS, Level, LogEvent


def event(ts: datetime, **overrides: Any) -> LogEvent:
    base = LogEvent(
        ts=ts,
        trace_id="t",
        source="gateway",
        target="catalog",
        method="GET",
        path="/v1/products/search",
        status=200,
        latency_ms=10.0,
        bytes_out=1000,
        client_ip="77.37.1.1",
        user_id="u-1",
    )
    return replace(base, **overrides)


class TestWindowAggregator:
    def test_features_of_a_window(self, start: datetime) -> None:
        inventory = RouteInventory()
        aggregator = WindowAggregator(["catalog", "gateway"], inventory)
        events = [
            event(start + timedelta(seconds=1)),
            event(start + timedelta(seconds=2), status=503, client_ip="77.37.1.2"),
            event(start + timedelta(seconds=3), status=401, level=Level.WARN),
            event(start + timedelta(seconds=4), status=404),
        ]
        closed = [w for e in events for w in aggregator.add(e)]
        closed += aggregator.add(event(start + timedelta(seconds=13)))
        catalog = next(w for w in closed if w.service == "catalog")
        assert catalog["requests"] == 4
        assert catalog["server_error_rate"] == pytest.approx(0.25)
        assert catalog["auth_failure_rate"] == pytest.approx(0.25)
        assert catalog["client_error_rate"] == pytest.approx(0.25)
        assert catalog["unique_clients"] == 2
        assert catalog["top_client_share"] == pytest.approx(0.75)
        gateway = next(w for w in closed if w.service == "gateway")
        assert gateway["outbound_bytes"] == 4000

    def test_silent_services_still_emit_windows(self, start: datetime) -> None:
        aggregator = WindowAggregator(["catalog", "orders"], RouteInventory())
        aggregator.add(event(start))
        closed = aggregator.add(event(start + timedelta(seconds=15)))
        orders = next(w for w in closed if w.service == "orders")
        assert orders["requests"] == 0

    def test_novel_routes_are_flagged_once_inventory_is_frozen(self, start: datetime) -> None:
        learning = RouteInventory()
        learning.is_novel(event(start))
        frozen = learning.freeze()
        aggregator = WindowAggregator(["users"], frozen)
        aggregator.add(event(start, target="users", source="notifications", path="/export"))
        closed = aggregator.add(event(start + timedelta(seconds=15), target="users"))
        assert closed[0]["novel_route_ratio"] == 1.0
        assert not frozen.is_novel(event(start))

    def test_egress_destination_is_part_of_the_route(self, start: datetime) -> None:
        inventory = RouteInventory()
        known = event(start, target=EGRESS, dest_ip="198.51.100.25")
        inventory.is_novel(known)
        frozen = inventory.freeze()
        assert not frozen.is_novel(known)
        assert frozen.is_novel(replace(known, dest_ip="203.0.113.77"))

    def test_late_events_beyond_tolerance_are_dropped(self, start: datetime) -> None:
        aggregator = WindowAggregator(["catalog"], RouteInventory())
        aggregator.add(event(start))
        aggregator.add(event(start + timedelta(seconds=30)))
        assert aggregator.add(event(start + timedelta(seconds=1))) == []

    def test_close_drops_the_trailing_partial_window(self, start: datetime) -> None:
        aggregator = WindowAggregator(["catalog"], RouteInventory())
        assert aggregator.add(event(start)) == []
        assert aggregator.add(event(start + timedelta(seconds=11))) == []
        assert [w.start for w in aggregator.close()] == [start]


class TestDetector:
    def test_threshold_controls_false_positives_on_unseen_normal_traffic(
        self, detector: AnomalyDetector, start: datetime
    ) -> None:
        clean, _ = baseline_windows(
            timedelta(hours=1), seed=4242, start=start.replace(hour=12), warmup=timedelta(0)
        )
        rate = np.mean([d.is_anomaly for d in detector.detect_many(clean)])
        assert rate < 0.02
        assert replay(detector, clean) == []

    @pytest.mark.parametrize(
        "scenario", ["ddos", "brute_force", "software_fault", "data_exfiltration", "benign"]
    )
    def test_each_incident_raises_a_timely_signal(
        self, detector: AnomalyDetector, scenario: str
    ) -> None:
        run = scenario_run(
            scenario,
            inventory=detector.inventory,
            seed=11,
            lead=timedelta(minutes=3),
            duration=timedelta(minutes=3),
            tail=timedelta(minutes=1),
        )
        delay = first_detection_delay(run, replay(detector, run.windows))
        assert delay is not None
        assert delay <= 60

    def test_explanations_name_the_deviating_features(
        self, detector: AnomalyDetector, training: tuple[list[WindowFeatures], RouteInventory]
    ) -> None:
        window = training[0][-1]
        values = list(window.values)
        values[FEATURE_NAMES.index("auth_failure_rate")] = 0.9
        detection = detector.detect(replace(window, values=tuple(values)))
        assert detection.is_anomaly
        assert detection.deviations[0].feature == "auth_failure_rate"

    def test_roundtrip_and_version_guard(self, detector: AnomalyDetector, tmp_path: Path) -> None:
        path = tmp_path / "detector.joblib"
        detector.save(path)
        assert AnomalyDetector.load(path).threshold == detector.threshold
        payload = joblib.load(path)
        joblib.dump({**payload, "features": ("other",)}, path)
        with pytest.raises(ValueError, match="incompatible"):
            AnomalyDetector.load(path)

    def test_fit_rejects_empty_training_set(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            AnomalyDetector.fit([], RouteInventory())


def detection(service: str, score: float, start: datetime) -> Detection:
    window = WindowFeatures(
        service, start, start + timedelta(seconds=10), (0.0,) * len(FEATURE_NAMES)
    )
    return Detection(window, score, threshold=1.0, deviations=())


class TestSignalGate:
    def test_requires_two_of_three_votes(self, start: datetime) -> None:
        gate = SignalGate()
        assert gate.observe([detection("auth", 2.0, start)]) is None
        assert gate.observe([detection("auth", 0.5, start)]) is None
        signal = gate.observe([detection("auth", 2.0, start)])
        assert signal is not None
        assert signal.service == "auth"

    def test_cascade_joins_the_open_incident(self, start: datetime) -> None:
        gate = SignalGate(min_votes=1)
        first = gate.observe([detection("gateway", 3.0, start), detection("catalog", 2.0, start)])
        assert first is not None
        assert first.correlated_services == ("catalog",)
        assert gate.observe([detection("users", 2.0, start)]) is None
        assert gate.members == {"gateway", "catalog", "users"}

    def test_incident_closes_after_quiet_period(self, start: datetime) -> None:
        gate = SignalGate(min_votes=1, quiet_windows=2)
        assert gate.observe([detection("auth", 2.0, start)]) is not None
        for _ in range(2):
            assert gate.observe([detection("auth", 0.1, start)]) is None
        assert not gate.incident_open
        assert gate.observe([detection("auth", 2.0, start)]) is not None

    def test_rejects_inconsistent_configuration(self) -> None:
        with pytest.raises(ValueError, match="min_votes"):
            SignalGate(votes_window=2, min_votes=3)
