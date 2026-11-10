import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import accumulate
from typing import TYPE_CHECKING, Protocol

from meshguard.domain import EGRESS, EXTERNAL, Level, LogEvent
from meshguard.simulation.topology import POD_IPS, Endpoint, Topology

if TYPE_CHECKING:
    from meshguard.simulation.scenarios import Scenario

RESIDENTIAL_PREFIXES = (
    "31.173",
    "37.112",
    "46.39",
    "77.37",
    "78.106",
    "79.139",
    "85.140",
    "89.113",
    "91.77",
    "93.80",
    "95.24",
    "109.252",
    "176.59",
    "178.66",
    "188.162",
    "212.164",
)

BASE_SERVER_ERROR_RATE = 0.002
BASE_CLIENT_ERROR_RATE = 0.012
BASE_INVALID_LOGIN_RATE = 0.08
SHED_LATENCY_MS = 4.0
MIN_CAPACITY_RPS = 12.0


class Enforcement(Protocol):
    def is_blocked(self, ip: str) -> bool: ...

    def is_isolated(self, service: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class Request:
    entry: str
    client_ip: str
    user_id: str | None = None
    credentials_valid: bool = True


@dataclass(frozen=True, slots=True)
class Fault:
    error_rate: float = 0.0
    status: int = 500
    message: str = ""
    latency_factor: float = 1.0
    timeout_rate: float = 0.0


def poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    threshold, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


def random_ip(rng: random.Random, prefix: str) -> str:
    octets = prefix.split(".")
    while len(octets) < 4:
        octets.append(str(rng.randint(1, 254)))
    return ".".join(octets)


class ClientPool:
    """Population of clients with heavy-tailed activity and sticky user identities."""

    def __init__(
        self,
        rng: random.Random,
        size: int,
        prefixes: Sequence[str] = RESIDENTIAL_PREFIXES,
        user_space: int = 20_000,
        skew: float = 0.8,
    ) -> None:
        unique: dict[str, None] = {}
        while len(unique) < size:
            unique[random_ip(rng, rng.choice(prefixes))] = None
        self.ips = list(unique)
        self.users = {ip: f"u-{rng.randrange(user_space):05d}" for ip in self.ips}
        weights = [1 / (rank**skew) for rank in range(1, size + 1)]
        self._cumulative = list(accumulate(weights))

    def pick(self, rng: random.Random) -> tuple[str, str]:
        ip = rng.choices(self.ips, cum_weights=self._cumulative)[0]
        return ip, self.users[ip]


@dataclass(slots=True)
class _Context:
    trace_id: str
    request: Request
    load: dict[str, float]
    faults: dict[str, Fault]


class TrafficGenerator:
    """Second-by-second simulator of a service mesh under a configurable set of scenarios."""

    def __init__(
        self,
        topology: Topology,
        *,
        start: datetime,
        seed: int = 0,
        base_rps: float = 12.0,
        diurnal_amplitude: float = 0.3,
        capacity_factor: float = 3.0,
        scenarios: Sequence[Scenario] = (),
        enforcement: Enforcement | None = None,
    ) -> None:
        self.topology = topology
        self.clock = start
        self.base_rps = base_rps
        self.diurnal_amplitude = diurnal_amplitude
        self.scenarios = list(scenarios)
        self.enforcement = enforcement
        self.rng = random.Random(seed)
        self.clients = ClientPool(self.rng, size=1_500)
        self._entry_keys = [r.endpoint for r in topology.entries]
        self._entry_weights = [r.weight for r in topology.entries]
        peak = base_rps * (1 + diurnal_amplitude)
        self._capacity = {
            service: max(capacity_factor * peak * share, MIN_CAPACITY_RPS)
            for service, share in topology.spans_per_request.items()
        }
        self._nominal_utilisation = 1 / capacity_factor
        self._noise = 0.0

    def rate(self, at: datetime) -> float:
        hours = at.hour + at.minute / 60
        diurnal = 1 + self.diurnal_amplitude * math.sin(2 * math.pi * (hours - 10) / 24)
        self._noise += -0.05 * self._noise + 0.03 * self.rng.gauss(0, 1)
        return self.base_rps * diurnal * math.exp(self._noise)

    def sample_base_request(self) -> Request:
        entry = self.rng.choices(self._entry_keys, weights=self._entry_weights)[0]
        ip, user = self.clients.pick(self.rng)
        valid = self.rng.random() >= BASE_INVALID_LOGIN_RATE
        return Request(entry, ip, user, credentials_valid=valid)

    def run(self, duration: timedelta) -> Iterator[LogEvent]:
        end = self.clock + duration
        while self.clock < end:
            yield from self.tick()

    def tick(self) -> list[LogEvent]:
        second = self.clock
        self.clock += timedelta(seconds=1)
        active = [s for s in self.scenarios if s.is_active(second)]

        requests = [self.sample_base_request() for _ in range(poisson(self.rng, self.rate(second)))]
        for scenario in active:
            requests.extend(scenario.requests(second, self))
        if self.enforcement is not None:
            requests = [r for r in requests if not self.enforcement.is_blocked(r.client_ip)]

        load = self._load(requests)
        faults = {key: fault for scenario in active for key, fault in scenario.faults().items()}
        events: list[LogEvent] = []
        for request in requests:
            arrival = second + timedelta(seconds=self.rng.random())
            ctx = _Context(f"{self.rng.getrandbits(64):016x}", request, load, faults)
            self._span(request.entry, EXTERNAL, ctx, arrival, events)
        for scenario in active:
            events.extend(self._admit(scenario.side_events(second, self)))
        events.sort(key=lambda e: e.ts)
        return events

    def _admit(self, events: list[LogEvent]) -> Iterator[LogEvent]:
        for event in events:
            if self.enforcement is None:
                yield event
            elif self.enforcement.is_isolated(event.source) or (
                event.dest_ip is not None and self.enforcement.is_blocked(event.dest_ip)
            ):
                continue
            else:
                yield event

    def _load(self, requests: list[Request]) -> dict[str, float]:
        counts: defaultdict[str, int] = defaultdict(int)
        for request in requests:
            for _, endpoint in self.topology.walk(request.entry):
                counts[endpoint.service] += 1
        return {s: counts.get(s, 0) / cap for s, cap in self._capacity.items()}

    def _queue_factor(self, utilisation: float) -> float:
        factor = (1 - self._nominal_utilisation) / (1 - min(utilisation, 0.97))
        return max(0.8, factor)

    def _span(
        self, key: str, caller: str, ctx: _Context, start: datetime, sink: list[LogEvent]
    ) -> tuple[int, float]:
        endpoint = self.topology[key]
        rng = self.rng
        utilisation = ctx.load.get(endpoint.service, 0.0)
        status, message = 200, ""

        if self.enforcement is not None and self.enforcement.is_isolated(endpoint.service):
            return self._emit(
                endpoint,
                caller,
                ctx,
                start,
                sink,
                status=503,
                latency=SHED_LATENCY_MS,
                message="connection refused: workload quarantined by mesh policy",
            )
        if utilisation > 1 and rng.random() < 1 - 1 / utilisation:
            return self._emit(
                endpoint,
                caller,
                ctx,
                start,
                sink,
                status=503,
                latency=SHED_LATENCY_MS,
                message=f"load shedding: {endpoint.service} concurrency limit reached",
            )

        fault = ctx.faults.get(endpoint.key, Fault())
        own = rng.lognormvariate(
            math.log(endpoint.latency_ms * self._queue_factor(utilisation) * fault.latency_factor),
            0.35,
        )
        if rng.random() < fault.timeout_rate:
            status, own, message = 504, 5_000.0, f"upstream timeout after 5000ms in {endpoint.key}"
        elif endpoint.authenticates and not ctx.request.credentials_valid:
            status, message = 401, f"authentication failed for login={ctx.request.user_id}"
        elif rng.random() < fault.error_rate:
            status, message = fault.status, fault.message
        elif rng.random() < BASE_SERVER_ERROR_RATE:
            status, message = 500, "unhandled exception: connection reset by peer"
        elif rng.random() < BASE_CLIENT_ERROR_RATE:
            status, message = rng.choice([(400, "validation failed"), (404, "resource not found")])

        elapsed = own * 0.3
        if status < 500:
            for child in endpoint.calls:
                child_status, child_latency = self._span(
                    child, endpoint.service, ctx, start + timedelta(milliseconds=elapsed), sink
                )
                elapsed += child_latency
                if child_status >= 500 and status < 400 and rng.random() < 0.85:
                    service = self.topology[child].service
                    status, message = 502, f"dependency {service} failed with {child_status}"
        return self._emit(
            endpoint,
            caller,
            ctx,
            start,
            sink,
            status=status,
            latency=elapsed + own * 0.7,
            message=message,
        )

    def _emit(
        self,
        endpoint: Endpoint,
        caller: str,
        ctx: _Context,
        start: datetime,
        sink: list[LogEvent],
        *,
        status: int,
        latency: float,
        message: str,
    ) -> tuple[int, float]:
        size = (
            round(self.rng.lognormvariate(math.log(endpoint.bytes_out), 0.5))
            if status < 400
            else self.rng.randint(90, 260)
        )
        is_egress = endpoint.service == EGRESS
        sink.append(
            LogEvent(
                ts=start,
                trace_id=ctx.trace_id,
                source=caller,
                target=endpoint.service,
                method=endpoint.method,
                path=endpoint.path,
                status=status,
                latency_ms=latency,
                bytes_out=size,
                client_ip=POD_IPS[caller] if is_egress else ctx.request.client_ip,
                level=Level.ERROR if status >= 500 else Level.WARN if status >= 400 else Level.INFO,
                user_id=ctx.request.user_id,
                dest_ip=endpoint.dest_ip,
                message=message,
            )
        )
        return status, latency
