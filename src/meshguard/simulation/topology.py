from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

from meshguard.domain import EGRESS


@dataclass(frozen=True, slots=True)
class Endpoint:
    service: str
    method: str
    path: str
    latency_ms: float
    bytes_out: int
    calls: tuple[str, ...] = ()
    authenticates: bool = False
    dest_ip: str | None = None

    @property
    def key(self) -> str:
        return f"{self.service} {self.method} {self.path}"


@dataclass(frozen=True, slots=True)
class EntryRoute:
    endpoint: str
    weight: float


class Topology:
    """Static description of the service mesh: endpoints, call tree and ingress mix."""

    def __init__(self, endpoints: list[Endpoint], entries: list[EntryRoute]) -> None:
        self.endpoints = {e.key: e for e in endpoints}
        self.entries = entries
        missing = {c for e in endpoints for c in e.calls} - self.endpoints.keys()
        missing |= {r.endpoint for r in entries} - self.endpoints.keys()
        if missing:
            raise ValueError(f"unknown endpoints referenced: {sorted(missing)}")

    def __getitem__(self, key: str) -> Endpoint:
        return self.endpoints[key]

    @cached_property
    def services(self) -> tuple[str, ...]:
        return tuple(sorted({e.service for e in self.endpoints.values()} - {EGRESS}))

    def walk(self, key: str) -> Iterator[tuple[str | None, Endpoint]]:
        """Yield (caller service, endpoint) pairs of the call tree rooted at ``key``."""
        stack: list[tuple[str | None, str]] = [(None, key)]
        while stack:
            caller, current = stack.pop()
            endpoint = self.endpoints[current]
            yield caller, endpoint
            stack.extend((endpoint.service, child) for child in reversed(endpoint.calls))

    @cached_property
    def spans_per_request(self) -> dict[str, float]:
        """Expected spans each service receives per ingress request under the base mix."""
        total: defaultdict[str, float] = defaultdict(float)
        for route in self.entries:
            for _, endpoint in self.walk(route.endpoint):
                total[endpoint.service] += route.weight
        return dict(total)


PSP_IP = "198.51.100.40"
SMTP_RELAY_IP = "198.51.100.25"

_ENDPOINTS = [
    Endpoint(
        "gateway", "GET", "/api/catalog/search", 9, 1_200, ("catalog GET /v1/products/search",)
    ),
    Endpoint(
        "gateway", "GET", "/api/catalog/items/{id}", 7, 2_400, ("catalog GET /v1/products/{id}",)
    ),
    Endpoint("gateway", "POST", "/api/auth/login", 6, 600, ("auth POST /v1/sessions",)),
    Endpoint(
        "gateway",
        "GET",
        "/api/profile",
        5,
        1_800,
        ("auth GET /v1/tokens/verify", "users GET /v1/users/{id}"),
    ),
    Endpoint(
        "gateway",
        "GET",
        "/api/orders",
        5,
        3_500,
        ("auth GET /v1/tokens/verify", "orders GET /v1/orders"),
    ),
    Endpoint(
        "gateway",
        "POST",
        "/api/orders",
        8,
        900,
        ("auth GET /v1/tokens/verify", "orders POST /v1/orders"),
    ),
    Endpoint("catalog", "GET", "/v1/products/search", 35, 14_000),
    Endpoint("catalog", "GET", "/v1/products/{id}", 12, 4_000, ("inventory GET /v1/stock/{id}",)),
    Endpoint("inventory", "GET", "/v1/stock/{id}", 6, 300),
    Endpoint("inventory", "POST", "/v1/reservations", 18, 400),
    Endpoint(
        "auth",
        "POST",
        "/v1/sessions",
        45,
        700,
        ("users GET /v1/users/by-login",),
        authenticates=True,
    ),
    Endpoint("auth", "GET", "/v1/tokens/verify", 3, 200),
    Endpoint("users", "GET", "/v1/users/by-login", 8, 900),
    Endpoint("users", "GET", "/v1/users/{id}", 7, 1_500),
    Endpoint("users", "GET", "/v1/users/{id}/contacts", 6, 500),
    Endpoint("orders", "GET", "/v1/orders", 25, 6_000),
    Endpoint(
        "orders",
        "POST",
        "/v1/orders",
        40,
        800,
        (
            "inventory POST /v1/reservations",
            "payments POST /v1/charges",
            "notifications POST /v1/messages",
        ),
    ),
    Endpoint("payments", "POST", "/v1/charges", 120, 600, ("egress POST /v2/payment_intents",)),
    Endpoint("egress", "POST", "/v2/payment_intents", 180, 1_100, dest_ip=PSP_IP),
    Endpoint(
        "notifications",
        "POST",
        "/v1/messages",
        15,
        300,
        ("users GET /v1/users/{id}/contacts", "egress POST /v3/mail/send"),
    ),
    Endpoint("egress", "POST", "/v3/mail/send", 90, 2_200, dest_ip=SMTP_RELAY_IP),
]

_ENTRIES = [
    EntryRoute("gateway GET /api/catalog/search", 0.38),
    EntryRoute("gateway GET /api/catalog/items/{id}", 0.25),
    EntryRoute("gateway POST /api/auth/login", 0.05),
    EntryRoute("gateway GET /api/profile", 0.10),
    EntryRoute("gateway GET /api/orders", 0.12),
    EntryRoute("gateway POST /api/orders", 0.10),
]


def default_topology() -> Topology:
    return Topology(_ENDPOINTS, _ENTRIES)


POD_IPS = {
    service: f"10.0.{index}.{10 + index}"
    for index, service in enumerate(
        ("gateway", "auth", "users", "orders", "payments", "inventory", "catalog", "notifications"),
        start=1,
    )
}
