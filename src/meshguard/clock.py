from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class EventClock:
    """Event-time clock driven by the log stream, so replayed and simulated runs stay coherent."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, to: datetime) -> None:
        self._now = max(self._now, to)
