from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from meshguard.detection.features import RouteInventory, WindowFeatures, aggregate
from meshguard.domain import IncidentType
from meshguard.simulation.scenarios import Scenario, build_scenario
from meshguard.simulation.topology import default_topology
from meshguard.simulation.traffic import TrafficGenerator

DEFAULT_START = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
INCIDENT_DAY_START = DEFAULT_START.replace(hour=13)
WINDOW = timedelta(seconds=10)


def simulate_windows(
    duration: timedelta,
    *,
    inventory: RouteInventory,
    seed: int,
    start: datetime = DEFAULT_START,
    scenarios: Sequence[Scenario] = (),
    window: timedelta = WINDOW,
) -> list[WindowFeatures]:
    topology = default_topology()
    generator = TrafficGenerator(topology, start=start, seed=seed, scenarios=scenarios)
    return aggregate(generator.run(duration), topology.services, inventory, size=window)


def baseline_windows(
    duration: timedelta,
    *,
    seed: int = 7,
    start: datetime = DEFAULT_START,
    warmup: timedelta = timedelta(minutes=20),
) -> tuple[list[WindowFeatures], RouteInventory]:
    """Normal traffic only; the warm-up period populates the route inventory and is discarded."""
    inventory = RouteInventory(learning=True)
    windows = simulate_windows(duration + warmup, inventory=inventory, seed=seed, start=start)
    return [w for w in windows if w.start >= start + warmup], inventory.freeze()


@dataclass(frozen=True, slots=True)
class LabeledRun:
    scenario: Scenario
    windows: list[WindowFeatures]

    @property
    def labels(self) -> NDArray[np.bool_]:
        return np.array([is_affected(w, self.scenario) for w in self.windows])

    @property
    def kind(self) -> IncidentType:
        return self.scenario.kind


def is_affected(window: WindowFeatures, scenario: Scenario) -> bool:
    overlaps = window.start < scenario.end and window.end > scenario.start
    return overlaps and window.service in scenario.affected_services


def scenario_run(
    name: str,
    *,
    inventory: RouteInventory,
    seed: int,
    start: datetime = INCIDENT_DAY_START,
    lead: timedelta = timedelta(minutes=8),
    duration: timedelta = timedelta(minutes=6),
    tail: timedelta = timedelta(minutes=4),
) -> LabeledRun:
    scenario = build_scenario(name, start + lead, duration, seed=seed)
    windows = simulate_windows(
        lead + duration + tail, inventory=inventory, seed=seed, start=start, scenarios=[scenario]
    )
    return LabeledRun(scenario, windows)
