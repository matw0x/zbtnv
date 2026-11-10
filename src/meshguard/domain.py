from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4


class Level(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class IncidentType(StrEnum):
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    SOFTWARE_FAULT = "software_fault"
    BENIGN = "benign"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


EGRESS = "egress"
EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class LogEvent:
    """One span of inter-service communication as logged by the receiving side."""

    ts: datetime
    trace_id: str
    source: str
    target: str
    method: str
    path: str
    status: int
    latency_ms: float
    bytes_out: int
    client_ip: str
    level: Level = Level.INFO
    user_id: str | None = None
    dest_ip: str | None = None
    message: str = ""

    @property
    def route(self) -> str:
        return f"{self.source}->{self.target} {self.method} {self.path}"

    @property
    def is_server_error(self) -> bool:
        return self.status >= 500

    @property
    def is_client_error(self) -> bool:
        return 400 <= self.status < 500

    @property
    def is_auth_failure(self) -> bool:
        return self.status in {401, 403}

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "trace_id": self.trace_id,
            "source": self.source,
            "target": self.target,
            "method": self.method,
            "path": self.path,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "bytes_out": self.bytes_out,
            "client_ip": self.client_ip,
            "level": self.level.value,
            "user_id": self.user_id,
            "dest_ip": self.dest_ip,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Self:
        return cls(
            ts=datetime.fromisoformat(raw["ts"]),
            trace_id=raw["trace_id"],
            source=raw["source"],
            target=raw["target"],
            method=raw["method"],
            path=raw["path"],
            status=int(raw["status"]),
            latency_ms=float(raw["latency_ms"]),
            bytes_out=int(raw["bytes_out"]),
            client_ip=raw["client_ip"],
            level=Level(raw.get("level", Level.INFO)),
            user_id=raw.get("user_id"),
            dest_ip=raw.get("dest_ip"),
            message=raw.get("message", ""),
        )


@dataclass(frozen=True, slots=True)
class FeatureDeviation:
    feature: str
    value: float
    baseline: float
    z_score: float


@dataclass(frozen=True, slots=True)
class AnomalySignal:
    """Hand-off contract between the statistical detector and the investigating agent."""

    service: str
    window_start: datetime
    window_end: datetime
    score: float
    threshold: float
    deviations: tuple[FeatureDeviation, ...]
    correlated_services: tuple[str, ...] = ()
    signal_id: str = field(default_factory=lambda: f"sig-{uuid4().hex[:10]}")

    @property
    def services(self) -> tuple[str, ...]:
        return (self.service, *self.correlated_services)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "service": self.service,
            "window": [self.window_start.isoformat(), self.window_end.isoformat()],
            "anomaly_score": round(self.score, 4) + 0.0,
            "threshold": round(self.threshold, 4),
            "correlated_services": list(self.correlated_services),
            "top_deviations": [
                {
                    "feature": d.feature,
                    "value": round(d.value, 4),
                    "baseline_median": round(d.baseline, 4),
                    "robust_z": round(d.z_score, 2),
                }
                for d in self.deviations
            ],
        }
