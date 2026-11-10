from datetime import datetime, timedelta

import pytest

from meshguard.detection.datasets import INCIDENT_DAY_START, baseline_windows
from meshguard.detection.detector import AnomalyDetector
from meshguard.detection.features import RouteInventory, WindowFeatures


@pytest.fixture(scope="session")
def training() -> tuple[list[WindowFeatures], RouteInventory]:
    return baseline_windows(timedelta(hours=3), seed=7, start=INCIDENT_DAY_START.replace(hour=11))


@pytest.fixture(scope="session")
def detector(training: tuple[list[WindowFeatures], RouteInventory]) -> AnomalyDetector:
    windows, inventory = training
    return AnomalyDetector.fit(windows, inventory)


@pytest.fixture
def start() -> datetime:
    return INCIDENT_DAY_START
