from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta

from meshguard.domain import LogEvent


class LogStore:
    """Time-indexed in-memory journal with bounded retention.

    Events arrive almost sorted (spans of one request may straddle a tick boundary),
    so out-of-order inserts land near the tail and stay cheap.
    """

    def __init__(self, retention: timedelta = timedelta(minutes=30)) -> None:
        self.retention = retention
        self._ts: list[float] = []
        self._events: list[LogEvent] = []

    def __len__(self) -> int:
        return len(self._events)

    def append(self, event: LogEvent) -> None:
        ts = event.ts.timestamp()
        if not self._ts or ts >= self._ts[-1]:
            self._ts.append(ts)
            self._events.append(event)
        else:
            index = bisect_right(self._ts, ts)
            self._ts.insert(index, ts)
            self._events.insert(index, event)
        horizon = ts - self.retention.total_seconds()
        if self._ts[0] < horizon - 60:
            cut = bisect_left(self._ts, horizon)
            del self._ts[:cut]
            del self._events[:cut]

    def extend(self, events: Iterable[LogEvent]) -> None:
        for event in events:
            self.append(event)

    def between(self, since: datetime, until: datetime) -> Iterator[LogEvent]:
        lo = bisect_left(self._ts, since.timestamp())
        hi = bisect_right(self._ts, until.timestamp())
        yield from self._events[lo:hi]
