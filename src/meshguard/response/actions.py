import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ipaddress import IPv4Address, IPv4Network
from pathlib import Path
from textwrap import dedent

from meshguard.clock import Clock
from meshguard.domain import Severity

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BlockRule:
    network: IPv4Network
    reason: str
    incident_id: str
    created_at: datetime
    expires_at: datetime

    @property
    def command(self) -> str:
        ttl = int((self.expires_at - self.created_at).total_seconds())
        return f"nft add element inet filter blocklist {{ {self.network} timeout {ttl}s }}"


class Firewall:
    """Simulated edge firewall backed by an nftables-style set with per-element timeouts."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.rules: dict[IPv4Network, BlockRule] = {}
        self._hosts: dict[str, BlockRule] = {}
        self._ranges: dict[IPv4Network, BlockRule] = {}

    def block(
        self, networks: list[IPv4Network], *, reason: str, ttl: timedelta, incident_id: str
    ) -> list[BlockRule]:
        now = self.clock.now()
        applied = []
        for network in networks:
            rule = BlockRule(network, reason, incident_id, now, now + ttl)
            self.rules[network] = rule
            if network.prefixlen == 32:
                self._hosts[str(network.network_address)] = rule
            else:
                self._ranges[network] = rule
            applied.append(rule)
        log.info("firewall: blocked %d networks for incident %s", len(applied), incident_id)
        return applied

    def is_blocked(self, ip: str) -> bool:
        now = self.clock.now()
        if (host := self._hosts.get(ip)) is not None and host.expires_at > now:
            return True
        if not self._ranges:
            return False
        address = IPv4Address(ip)
        return any(address in net and r.expires_at > now for net, r in self._ranges.items())


@dataclass(frozen=True, slots=True)
class Quarantine:
    service: str
    reason: str
    incident_id: str
    created_at: datetime

    @property
    def manifest(self) -> str:
        return dedent(
            f"""\
            apiVersion: networking.k8s.io/v1
            kind: NetworkPolicy
            metadata:
              name: quarantine-{self.service}
              namespace: shop
              labels: {{meshguard/incident: {self.incident_id}}}
            spec:
              podSelector: {{matchLabels: {{app: {self.service}}}}}
              policyTypes: [Ingress, Egress]
              ingress:
                - from: [{{namespaceSelector: {{matchLabels: {{name: observability}}}}}}]
              egress: []
            """
        )


class ServiceMesh:
    """Simulated mesh control plane able to cut a workload off the network."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.quarantined: dict[str, Quarantine] = {}

    def isolate(self, service: str, *, reason: str, incident_id: str) -> Quarantine:
        quarantine = Quarantine(service, reason, incident_id, self.clock.now())
        self.quarantined[service] = quarantine
        log.info("mesh: applied NetworkPolicy quarantine-%s", service)
        return quarantine

    def is_isolated(self, service: str) -> bool:
        return service in self.quarantined


@dataclass(frozen=True, slots=True)
class AdminReport:
    incident_id: str
    severity: Severity
    title: str
    summary: str
    recommendations: tuple[str, ...]
    created_at: datetime

    def render(self) -> str:
        steps = "\n".join(f"- {r}" for r in self.recommendations) or "- none"
        return (
            f"# [{self.severity.value.upper()}] {self.title}\n\n"
            f"Incident: `{self.incident_id}`  \nTime: {self.created_at.isoformat()}\n\n"
            f"{self.summary}\n\n## Recommended actions\n\n{steps}\n"
        )


@dataclass(slots=True)
class AdminNotifier:
    """Delivers reports to the on-call channel; here a markdown outbox stands in for it."""

    clock: Clock
    outbox: Path | None = None
    sent: list[AdminReport] = field(default_factory=list)

    def send(
        self,
        *,
        incident_id: str,
        severity: Severity,
        title: str,
        summary: str,
        recommendations: tuple[str, ...],
    ) -> AdminReport:
        report = AdminReport(
            incident_id, severity, title, summary, recommendations, self.clock.now()
        )
        self.sent.append(report)
        if self.outbox is not None:
            self.outbox.mkdir(parents=True, exist_ok=True)
            path = self.outbox / f"{incident_id}-{len(self.sent):02d}.md"
            path.write_text(report.render(), encoding="utf-8")
        log.info("notifier: [%s] %s", severity.value, title)
        return report


class ResponseControls:
    """Facade over enforcement points; also what the simulator consults to honour actions."""

    def __init__(self, clock: Clock, outbox: Path | None = None) -> None:
        self.firewall = Firewall(clock)
        self.mesh = ServiceMesh(clock)
        self.notifier = AdminNotifier(clock, outbox)

    def is_blocked(self, ip: str) -> bool:
        return self.firewall.is_blocked(ip)

    def is_isolated(self, service: str) -> bool:
        return self.mesh.is_isolated(service)
