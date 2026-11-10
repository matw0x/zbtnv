from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from meshguard.domain import AnomalySignal, IncidentType, Severity
from meshguard.response.policy import Classification


@dataclass(frozen=True, slots=True)
class ClassificationRecord:
    incident_type: IncidentType
    confidence: float
    affected_services: tuple[str, ...]
    summary: str
    evidence: tuple[str, ...]

    @property
    def classification(self) -> Classification:
        return Classification(self.incident_type, self.confidence)


@dataclass(frozen=True, slots=True)
class AuditEntry:
    at: datetime
    tool: str
    arguments: dict[str, Any]
    ok: bool
    result: str


@dataclass(slots=True)
class Investigation:
    """Mutable state and audit trail of one incident investigation."""

    signal: AnomalySignal
    started_at: datetime
    incident_id: str = field(default_factory=lambda: f"inc-{uuid4().hex[:8]}")
    classification: ClassificationRecord | None = None
    revisions: list[ClassificationRecord] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    isolated: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)
    notifications: list[Severity] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)

    def classify(self, record: ClassificationRecord) -> None:
        if self.classification is not None:
            self.revisions.append(self.classification)
        self.classification = record

    @property
    def is_complete(self) -> bool:
        return self.classification is not None and bool(self.notifications)
