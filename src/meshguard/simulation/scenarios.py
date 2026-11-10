import math
import random
from abc import ABC
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

from meshguard.domain import EGRESS, IncidentType, Level, LogEvent
from meshguard.simulation.topology import POD_IPS, SMTP_RELAY_IP
from meshguard.simulation.traffic import RESIDENTIAL_PREFIXES, ClientPool, Fault, Request, poisson

if TYPE_CHECKING:
    from meshguard.simulation.traffic import TrafficGenerator


class Scenario(ABC):
    """An incident injected into simulated traffic, carrying its own ground truth."""

    name: ClassVar[str]
    kind: ClassVar[IncidentType]
    affected_services: ClassVar[frozenset[str]]
    ramp: ClassVar[timedelta] = timedelta(seconds=30)

    def __init__(self, start: datetime, duration: timedelta, seed: int = 0) -> None:
        self.start = start
        self.end = start + duration
        self.rng = random.Random(seed)

    @property
    def offender_ips(self) -> frozenset[str]:
        return frozenset()

    def is_active(self, at: datetime) -> bool:
        return self.start <= at < self.end

    def intensity(self, at: datetime) -> float:
        return min(1.0, (at - self.start) / self.ramp) if self.is_active(at) else 0.0

    def requests(self, at: datetime, sim: TrafficGenerator) -> list[Request]:  # noqa: ARG002
        return []

    def faults(self) -> dict[str, Fault]:
        return {}

    def side_events(self, at: datetime, sim: TrafficGenerator) -> list[LogEvent]:  # noqa: ARG002
        return []


class HttpFlood(Scenario):
    """Volumetric L7 DDoS: a botnet from hosting ranges hammers an expensive search endpoint."""

    name = "ddos"
    kind = IncidentType.DDOS
    affected_services = frozenset({"gateway", "catalog"})
    prefixes: ClassVar[tuple[str, ...]] = ("45.141", "91.243", "185.234", "193.233", "212.192")
    bots: ClassVar[int] = 400
    rate_factor: ClassVar[float] = 12.0
    targets: ClassVar[tuple[str, ...]] = ("gateway GET /api/catalog/search",)

    def __init__(self, start: datetime, duration: timedelta, seed: int = 0) -> None:
        super().__init__(start, duration, seed)
        self.botnet = ClientPool(self.rng, size=self.bots, prefixes=self.prefixes, skew=0.1)

    @property
    def offender_ips(self) -> frozenset[str]:
        return frozenset(self.botnet.ips)

    def requests(self, at: datetime, sim: TrafficGenerator) -> list[Request]:
        volume = poisson(self.rng, sim.base_rps * self.rate_factor * self.intensity(at))
        return [
            Request(self.rng.choice(self.targets), self.botnet.pick(self.rng)[0])
            for _ in range(volume)
        ]


class ResidentialFlood(HttpFlood):
    """Flood from hijacked home routers that share subnets with real customers."""

    name = "ddos_residential"
    prefixes = RESIDENTIAL_PREFIXES
    bots = 600
    rate_factor = 8.0
    targets = ("gateway GET /api/catalog/search", "gateway GET /api/catalog/items/{id}")


class CredentialStuffing(Scenario):
    """Brute-force login attempts from a small set of proxy subnets."""

    name = "brute_force"
    kind = IncidentType.BRUTE_FORCE
    affected_services = frozenset({"gateway", "auth", "users"})
    prefixes: ClassVar[tuple[str, ...]] = ("194.26.29", "5.188.62", "141.98.11")
    proxies: ClassVar[int] = 36
    rps: ClassVar[float] = 8.0

    def __init__(self, start: datetime, duration: timedelta, seed: int = 0) -> None:
        super().__init__(start, duration, seed)
        self.pool = ClientPool(self.rng, size=self.proxies, prefixes=self.prefixes, skew=0.2)

    @property
    def offender_ips(self) -> frozenset[str]:
        return frozenset(self.pool.ips)

    def requests(self, at: datetime, sim: TrafficGenerator) -> list[Request]:  # noqa: ARG002
        return [
            Request(
                "gateway POST /api/auth/login",
                self.pool.pick(self.rng)[0],
                user_id=f"u-{self.rng.randrange(100_000):05d}",
                credentials_valid=self.rng.random() < 0.01,
            )
            for _ in range(poisson(self.rng, self.rps * self.intensity(at)))
        ]


class DistributedStuffing(CredentialStuffing):
    """Low-and-slow stuffing spread over hundreds of residential proxies."""

    name = "brute_force_distributed"
    prefixes = RESIDENTIAL_PREFIXES
    proxies = 300
    rps = 6.0


class FaultyRelease(Scenario):
    """A bad deployment of the payments service: exceptions, timeouts and cascading 502s."""

    name = "software_fault"
    kind = IncidentType.SOFTWARE_FAULT
    affected_services = frozenset({"payments", "orders", "gateway"})
    release = "payments@2.14.0"

    def faults(self) -> dict[str, Fault]:
        return {
            "payments POST /v1/charges": Fault(
                error_rate=0.45,
                status=500,
                latency_factor=4.0,
                timeout_rate=0.08,
                message=(
                    'java.lang.NullPointerException: Cannot invoke "Currency.getCode()" '
                    "at ChargeService.capture(ChargeService.java:142) "
                    f"[release={self.release}]"
                ),
            )
        }


class SilentDegradation(Scenario):
    """Inventory slows down and times out without any explicit exception in the logs."""

    name = "software_fault_silent"
    kind = IncidentType.SOFTWARE_FAULT
    affected_services = frozenset({"inventory", "catalog", "orders", "gateway"})

    def faults(self) -> dict[str, Fault]:
        degraded = Fault(error_rate=0.05, status=503, latency_factor=6.0, timeout_rate=0.2)
        return {
            "inventory GET /v1/stock/{id}": degraded,
            "inventory POST /v1/reservations": degraded,
        }


class DataExfiltration(Scenario):
    """A compromised notifications workload bulk-reads user records and ships them out."""

    name = "data_exfiltration"
    kind = IncidentType.DATA_EXFILTRATION
    affected_services = frozenset({"notifications", "users"})
    compromised = "notifications"
    drop_server: ClassVar[str | None] = "203.0.113.77"

    @property
    def offender_ips(self) -> frozenset[str]:
        return frozenset({self.drop_server}) if self.drop_server else frozenset()

    def _read(self, ts: datetime, trace: str) -> LogEvent:
        return LogEvent(
            ts=ts,
            trace_id=trace,
            source=self.compromised,
            target="users",
            method="GET",
            path="/v1/users/export",
            status=200,
            latency_ms=self.rng.lognormvariate(math.log(650), 0.25),
            bytes_out=round(self.rng.lognormvariate(math.log(2_500_000), 0.3)),
            client_ip=POD_IPS[self.compromised],
        )

    def _upload(self, ts: datetime, trace: str, size: int) -> LogEvent:
        return LogEvent(
            ts=ts,
            trace_id=trace,
            source=self.compromised,
            target=EGRESS,
            method="PUT",
            path="/upload",
            status=201,
            latency_ms=self.rng.lognormvariate(math.log(900), 0.3),
            bytes_out=size,
            client_ip=POD_IPS[self.compromised],
            dest_ip=self.drop_server,
        )

    def side_events(self, at: datetime, sim: TrafficGenerator) -> list[LogEvent]:  # noqa: ARG002
        events: list[LogEvent] = []
        for _ in range(poisson(self.rng, 1.5 * self.intensity(at))):
            ts = at + timedelta(seconds=self.rng.random())
            trace = f"{self.rng.getrandbits(64):016x}"
            read = self._read(ts, trace)
            events += [read, self._upload(ts + timedelta(milliseconds=700), trace, read.bytes_out)]
        if self.rng.random() < 0.15 * self.intensity(at):
            events.append(
                LogEvent(
                    ts=at,
                    trace_id=f"{self.rng.getrandbits(64):016x}",
                    source=self.compromised,
                    target="users",
                    method="GET",
                    path="/internal/admin/users",
                    status=403,
                    latency_ms=4.0,
                    bytes_out=120,
                    client_ip=POD_IPS[self.compromised],
                    level=Level.WARN,
                    message="forbidden: service account notifications-sa lacks role users:admin",
                )
            )
        return events


class RelayExfiltration(DataExfiltration):
    """Exfiltration hidden in known routes: contact lookups and the partner mail relay."""

    name = "data_exfiltration_relay"
    drop_server = None

    def _read(self, ts: datetime, trace: str) -> LogEvent:
        return LogEvent(
            ts=ts,
            trace_id=trace,
            source=self.compromised,
            target="users",
            method="GET",
            path="/v1/users/{id}/contacts",
            status=200,
            latency_ms=self.rng.lognormvariate(math.log(6), 0.35),
            bytes_out=round(self.rng.lognormvariate(math.log(500), 0.5)),
            client_ip=POD_IPS[self.compromised],
        )

    def side_events(self, at: datetime, sim: TrafficGenerator) -> list[LogEvent]:  # noqa: ARG002
        intensity = self.intensity(at)
        trace = f"{self.rng.getrandbits(64):016x}"
        events = [
            self._read(at + timedelta(seconds=self.rng.random()), trace)
            for _ in range(poisson(self.rng, 25 * intensity))
        ]
        events.extend(
            LogEvent(
                ts=at + timedelta(seconds=self.rng.random()),
                trace_id=trace,
                source=self.compromised,
                target=EGRESS,
                method="POST",
                path="/v3/mail/send",
                status=202,
                latency_ms=self.rng.lognormvariate(math.log(1_400), 0.3),
                bytes_out=round(self.rng.lognormvariate(math.log(3_000_000), 0.3)),
                client_ip=POD_IPS[self.compromised],
                dest_ip=SMTP_RELAY_IP,
            )
            for _ in range(poisson(self.rng, 0.5 * intensity))
        )
        return events


class FlashSale(Scenario):
    """Legitimate demand spike from new customers: anomalous volume, benign intent."""

    name = "benign"
    kind = IncidentType.BENIGN
    affected_services = frozenset(
        {"gateway", "catalog", "inventory", "auth", "users", "orders", "payments", "notifications"}
    )
    rate_factor: ClassVar[float] = 1.8

    def __init__(self, start: datetime, duration: timedelta, seed: int = 0) -> None:
        super().__init__(start, duration, seed)
        self.customers = ClientPool(self.rng, size=2_000, prefixes=RESIDENTIAL_PREFIXES)

    def requests(self, at: datetime, sim: TrafficGenerator) -> list[Request]:
        requests = []
        for _ in range(poisson(self.rng, sim.base_rps * self.rate_factor * self.intensity(at))):
            base = sim.sample_base_request()
            ip, user = self.customers.pick(self.rng)
            requests.append(Request(base.entry, ip, user, base.credentials_valid))
        return requests


class CampaignLanding(FlashSale):
    """Marketing push: organic customers all land on one product page at once."""

    name = "benign_campaign"
    affected_services = frozenset({"gateway", "catalog", "inventory"})
    rate_factor = 2.5

    def requests(self, at: datetime, sim: TrafficGenerator) -> list[Request]:
        return [
            Request("gateway GET /api/catalog/items/{id}", *self.customers.pick(self.rng))
            for _ in range(poisson(self.rng, sim.base_rps * self.rate_factor * self.intensity(at)))
        ]


CORE: tuple[type[Scenario], ...] = (
    HttpFlood,
    CredentialStuffing,
    FaultyRelease,
    DataExfiltration,
    FlashSale,
)
HARD: tuple[type[Scenario], ...] = (
    ResidentialFlood,
    DistributedStuffing,
    SilentDegradation,
    RelayExfiltration,
    CampaignLanding,
)
SCENARIOS: dict[str, type[Scenario]] = {cls.name: cls for cls in (*CORE, *HARD)}
CORE_NAMES = tuple(cls.name for cls in CORE)
HARD_NAMES = tuple(cls.name for cls in HARD)


def build_scenario(name: str, start: datetime, duration: timedelta, seed: int = 0) -> Scenario:
    try:
        return SCENARIOS[name](start, duration, seed)
    except KeyError:
        raise ValueError(f"unknown scenario {name!r}; choose from {sorted(SCENARIOS)}") from None
