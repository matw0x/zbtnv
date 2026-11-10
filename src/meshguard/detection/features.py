import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from meshguard.domain import EGRESS, LogEvent


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    name: str
    description: str
    log_scale: bool = False
    min_scale: float = 0.05


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("requests", "inbound requests per window", log_scale=True, min_scale=0.1),
    FeatureSpec("server_error_rate", "share of 5xx responses", min_scale=0.01),
    FeatureSpec("client_error_rate", "share of 4xx responses except 401/403", min_scale=0.02),
    FeatureSpec("auth_failure_rate", "share of 401/403 responses", min_scale=0.02),
    FeatureSpec("latency_p50_ms", "median latency", log_scale=True),
    FeatureSpec("latency_p95_ms", "95th percentile latency", log_scale=True, min_scale=0.1),
    FeatureSpec("bytes_per_response", "mean response size", log_scale=True, min_scale=0.1),
    FeatureSpec(
        "outbound_bytes",
        "bytes pulled from dependencies or sent to egress",
        log_scale=True,
        min_scale=0.25,
    ),
    FeatureSpec("unique_clients", "distinct client IPs", log_scale=True, min_scale=0.1),
    FeatureSpec("top_client_share", "share of requests from the busiest IP", min_scale=0.03),
    FeatureSpec("users_per_client", "distinct user ids per client IP", log_scale=True),
    FeatureSpec("novel_route_ratio", "share of traffic on never-seen routes", min_scale=0.01),
    FeatureSpec("hour_sin", "time of day, sine component", min_scale=0.1),
    FeatureSpec("hour_cos", "time of day, cosine component", min_scale=0.1),
)
FEATURE_NAMES = tuple(f.name for f in FEATURES)


def route_key(event: LogEvent) -> str:
    target = f"{EGRESS}:{event.dest_ip}" if event.target == EGRESS else event.target
    return f"{event.source}->{target} {event.method} {event.path}"


@dataclass(slots=True)
class RouteInventory:
    """Set of routes observed during the learning period; anything else counts as novel."""

    routes: set[str] = field(default_factory=set)
    learning: bool = True

    def is_novel(self, event: LogEvent) -> bool:
        key = route_key(event)
        if key in self.routes:
            return False
        if self.learning:
            self.routes.add(key)
        return True

    def freeze(self) -> RouteInventory:
        return RouteInventory(set(self.routes), learning=False)


@dataclass(frozen=True, slots=True)
class WindowFeatures:
    service: str
    start: datetime
    end: datetime
    values: tuple[float, ...]

    def __getitem__(self, name: str) -> float:
        return self.values[FEATURE_NAMES.index(name)]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))


def _time_of_day(at: datetime) -> tuple[float, float]:
    angle = 2 * math.pi * (at.hour * 3600 + at.minute * 60 + at.second) / 86_400
    return math.sin(angle), math.cos(angle)


@dataclass(slots=True)
class _Accumulator:
    latencies: list[float] = field(default_factory=list)
    statuses: Counter[int] = field(default_factory=Counter)
    bytes_served: int = 0
    outbound_bytes: int = 0
    clients: Counter[str] = field(default_factory=Counter)
    users: set[tuple[str, str]] = field(default_factory=set)
    involved: int = 0
    novel: int = 0

    def features(self, start: datetime) -> tuple[float, ...]:
        n = len(self.latencies)
        if n:
            p50, p95 = np.percentile(self.latencies, [50, 95])
            server = sum(c for s, c in self.statuses.items() if s >= 500) / n
            auth = (self.statuses[401] + self.statuses[403]) / n
            client = sum(c for s, c in self.statuses.items() if 400 <= s < 500) / n - auth
            top_share = self.clients.most_common(1)[0][1] / n
            users_per_client = len(self.users) / len(self.clients) if self.users else 0.0
        else:
            p50 = p95 = server = auth = client = top_share = users_per_client = 0.0
        return (
            float(n),
            server,
            client,
            auth,
            float(p50),
            float(p95),
            self.bytes_served / n if n else 0.0,
            float(self.outbound_bytes),
            float(len(self.clients)),
            top_share,
            users_per_client,
            self.novel / self.involved if self.involved else 0.0,
            *_time_of_day(start),
        )


class WindowAggregator:
    """Event-time tumbling windows per service with bounded tolerance to late events."""

    def __init__(
        self,
        services: Sequence[str],
        inventory: RouteInventory,
        *,
        size: timedelta = timedelta(seconds=10),
        lateness: timedelta = timedelta(seconds=2),
    ) -> None:
        self.services = tuple(services)
        self.inventory = inventory
        self.size = size
        self.lateness = lateness
        self._open: dict[datetime, defaultdict[str, _Accumulator]] = {}
        self._watermark: datetime | None = None
        self._origin: datetime | None = None

    def _window_start(self, ts: datetime) -> datetime:
        if self._origin is None:
            self._origin = ts.replace(second=0, microsecond=0)
        return ts - (ts - self._origin) % self.size

    def add(self, event: LogEvent) -> list[WindowFeatures]:
        start = self._window_start(event.ts)
        if self._watermark is not None and start + self.size <= self._watermark - self.lateness:
            return []
        buckets = self._open.setdefault(start, defaultdict(_Accumulator))
        novel = self.inventory.is_novel(event)
        if event.target in self.services:
            acc = buckets[event.target]
            acc.latencies.append(event.latency_ms)
            acc.statuses[event.status] += 1
            acc.bytes_served += event.bytes_out
            acc.clients[event.client_ip] += 1
            if event.user_id is not None:
                acc.users.add((event.client_ip, event.user_id))
            acc.involved += 1
            acc.novel += novel
        if event.source in self.services:
            acc = buckets[event.source]
            acc.outbound_bytes += event.bytes_out
            acc.involved += 1
            acc.novel += novel
        self._watermark = max(self._watermark or event.ts, event.ts)
        return self._flush(self._watermark - self.lateness)

    def _flush(self, horizon: datetime) -> list[WindowFeatures]:
        closed = sorted(s for s in self._open if s + self.size <= horizon)
        return [w for start in closed for w in self._emit(start)]

    def close(self) -> list[WindowFeatures]:
        """Flush windows completely covered by the stream; the trailing partial one is dropped."""
        if self._watermark is None:
            return []
        windows = self._flush(self._watermark)
        self._open.clear()
        return windows

    def _emit(self, start: datetime) -> Iterator[WindowFeatures]:
        buckets = self._open.pop(start)
        for service in self.services:
            acc = buckets.get(service) or _Accumulator()
            yield WindowFeatures(service, start, start + self.size, acc.features(start))


def aggregate(
    events: Iterable[LogEvent],
    services: Sequence[str],
    inventory: RouteInventory,
    *,
    size: timedelta = timedelta(seconds=10),
) -> list[WindowFeatures]:
    aggregator = WindowAggregator(services, inventory, size=size)
    windows = [w for event in events for w in aggregator.add(event)]
    windows.extend(aggregator.close())
    return windows
