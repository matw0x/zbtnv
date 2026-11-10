from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from ipaddress import IPv4Network

from pydantic import BaseModel, Field

from meshguard.domain import IncidentType
from meshguard.simulation.topology import PSP_IP, SMTP_RELAY_IP


class Action(StrEnum):
    BLOCK_IPS = "block_ips"
    ISOLATE_SERVICE = "isolate_service"


CONTAINMENT_MATRIX: dict[IncidentType, frozenset[Action]] = {
    IncidentType.DDOS: frozenset({Action.BLOCK_IPS}),
    IncidentType.BRUTE_FORCE: frozenset({Action.BLOCK_IPS}),
    IncidentType.DATA_EXFILTRATION: frozenset({Action.BLOCK_IPS, Action.ISOLATE_SERVICE}),
    IncidentType.SOFTWARE_FAULT: frozenset(),
    IncidentType.BENIGN: frozenset(),
}


class PolicyConfig(BaseModel):
    protected_networks: list[IPv4Network] = Field(
        default_factory=lambda: [
            IPv4Network(n)
            for n in (
                "10.0.0.0/8",
                "172.16.0.0/12",
                "192.168.0.0/16",
                "127.0.0.0/8",
                "100.64.0.0/10",
                f"{PSP_IP}/32",
                f"{SMTP_RELAY_IP}/32",
            )
        ]
    )
    min_prefix_length: int = Field(16, ge=8, le=32)
    max_networks_per_action: int = Field(1024, ge=1)
    critical_services: frozenset[str] = frozenset({"gateway", "auth"})
    known_services: frozenset[str] = frozenset(
        {"gateway", "auth", "users", "orders", "payments", "inventory", "catalog", "notifications"}
    )
    min_confidence: float = Field(0.7, ge=0, le=1)


@dataclass(frozen=True, slots=True)
class Classification:
    incident_type: IncidentType
    confidence: float


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    reason: str = ""

    @classmethod
    def deny(cls, reason: str) -> Decision:
        return cls(False, reason)


ALLOW = Decision(True)


class ResponsePolicy:
    """Deterministic guardrails between the agent's intent and enforcement points."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def _containment(self, action: Action, classification: Classification | None) -> Decision:
        if classification is None:
            return Decision.deny("record a classification before taking containment actions")
        if action not in CONTAINMENT_MATRIX[classification.incident_type]:
            return Decision.deny(
                f"{action.value} is not an approved response to "
                f"{classification.incident_type.value} incidents"
            )
        if classification.confidence < self.config.min_confidence:
            return Decision.deny(
                f"confidence {classification.confidence:.2f} is below the "
                f"{self.config.min_confidence:.2f} required for autonomous containment; escalate"
            )
        return ALLOW

    def check_block(
        self, networks: Sequence[IPv4Network], classification: Classification | None
    ) -> Decision:
        if not (decision := self._containment(Action.BLOCK_IPS, classification)).allowed:
            return decision
        if not networks:
            return Decision.deny("no addresses supplied")
        if len(networks) > self.config.max_networks_per_action:
            return Decision.deny(
                f"{len(networks)} targets exceed the per-action limit of "
                f"{self.config.max_networks_per_action}; aggregate into subnets"
            )
        for network in networks:
            if network.prefixlen < self.config.min_prefix_length:
                return Decision.deny(f"{network} is broader than /{self.config.min_prefix_length}")
            if protected := next(
                (p for p in self.config.protected_networks if network.overlaps(p)), None
            ):
                return Decision.deny(f"{network} overlaps protected range {protected}")
        return ALLOW

    def check_isolation(self, service: str, classification: Classification | None) -> Decision:
        if not (decision := self._containment(Action.ISOLATE_SERVICE, classification)).allowed:
            return decision
        if service not in self.config.known_services:
            return Decision.deny(f"unknown service {service!r}")
        if service in self.config.critical_services:
            return Decision.deny(
                f"{service} is business-critical; isolation requires human approval"
            )
        return ALLOW
