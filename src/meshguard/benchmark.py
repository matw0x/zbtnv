"""End-to-end evaluation of the detector + agent on injected incidents with known ground truth."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from meshguard.agent.investigator import InvestigationResult
from meshguard.agent.llm.base import ChatModel, Usage
from meshguard.app import build_runtime
from meshguard.config import Settings
from meshguard.detection.datasets import INCIDENT_DAY_START
from meshguard.detection.detector import AnomalyDetector
from meshguard.domain import IncidentType
from meshguard.pipeline import simulated_stream
from meshguard.response.actions import ResponseControls
from meshguard.simulation.scenarios import DataExfiltration, Scenario, build_scenario
from meshguard.simulation.topology import default_topology
from meshguard.simulation.traffic import TrafficGenerator

LABELS = [t.value for t in IncidentType]
RESPOND_TOOLS = {"block_ips", "isolate_service"}


@dataclass(frozen=True, slots=True)
class CaseResult:
    scenario: str
    expected: str
    seed: int
    predicted: str | None
    confidence: float | None
    outcome: str
    containment_correct: bool
    offenders_blocked: float | None
    collateral_clients: int
    isolated: list[str]
    blocked_networks: int
    denied_actions: int
    detection_delay_s: float | None
    containment_delay_s: float | None
    investigations: int
    turns: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    llm_seconds: float
    incident_id: str | None

    @property
    def correct(self) -> bool:
        return self.predicted == self.expected


def _relevant(
    results: Sequence[InvestigationResult], scenario: Scenario
) -> InvestigationResult | None:
    for result in results:
        signal = result.investigation.signal
        if (
            set(signal.services) & scenario.affected_services
            and signal.window_end >= scenario.start
        ):
            return result
    return None


def _containment_correct(
    scenario: Scenario, controls: ResponseControls, coverage: float | None, collateral: int
) -> bool:
    isolated = set(controls.mesh.quarantined)
    match scenario:
        case DataExfiltration(compromised=compromised):
            blocked_destination = coverage is None or coverage > 0
            return isolated == {compromised} and blocked_destination and collateral == 0
        case _ if scenario.kind in {IncidentType.DDOS, IncidentType.BRUTE_FORCE}:
            return coverage is not None and coverage >= 0.9 and collateral == 0 and not isolated
        case _:
            return not isolated and not controls.firewall.rules


async def run_case(
    settings: Settings,
    detector: AnomalyDetector,
    model: ChatModel,
    scenario_name: str,
    seed: int,
    *,
    start: datetime = INCIDENT_DAY_START,
    lead: timedelta = timedelta(minutes=4),
    duration: timedelta = timedelta(minutes=5),
    tail: timedelta = timedelta(minutes=2),
) -> tuple[CaseResult, list[InvestigationResult]]:
    runtime = build_runtime(settings, detector, start=start, model=model, blocking=True)
    scenario = build_scenario(scenario_name, start + lead, duration, seed=seed)
    generator = TrafficGenerator(
        default_topology(),
        start=start,
        seed=seed,
        scenarios=[scenario],
        enforcement=runtime.controls,
    )
    report = await runtime.pipeline.run(simulated_stream(generator, lead + duration + tail))
    controls = runtime.controls

    offenders = scenario.offender_ips
    coverage = (
        sum(controls.is_blocked(ip) for ip in offenders) / len(offenders) if offenders else None
    )
    collateral = sum(controls.is_blocked(ip) for ip in generator.clients.ips)

    relevant = _relevant(report.investigations, scenario)
    investigation = relevant.investigation if relevant else None
    classification = investigation.classification if investigation else None
    containment_at = next(
        (
            a.at
            for result in report.investigations
            for a in result.investigation.audit
            if a.tool in RESPOND_TOOLS and a.ok
        ),
        None,
    )
    usage = sum((r.usage for r in report.investigations), Usage())

    case = CaseResult(
        scenario=scenario_name,
        expected=scenario.kind.value,
        seed=seed,
        predicted=classification.incident_type.value if classification else None,
        confidence=classification.confidence if classification else None,
        outcome=relevant.outcome.value if relevant else "missed",
        containment_correct=_containment_correct(scenario, controls, coverage, collateral),
        offenders_blocked=coverage,
        collateral_clients=collateral,
        isolated=sorted(controls.mesh.quarantined),
        blocked_networks=len(controls.firewall.rules),
        denied_actions=sum(len(r.investigation.denied) for r in report.investigations),
        detection_delay_s=(
            (investigation.signal.window_end - scenario.start).total_seconds()
            if investigation
            else None
        ),
        containment_delay_s=(
            (containment_at - scenario.start).total_seconds() if containment_at else None
        ),
        investigations=len(report.investigations),
        turns=sum(r.turns for r in report.investigations),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cached_input_tokens,
        llm_seconds=sum(r.wall_seconds for r in report.investigations),
        incident_id=investigation.incident_id if investigation else None,
    )
    return case, report.investigations


def summarize(cases: Sequence[CaseResult]) -> dict[str, Any]:
    y_true = [c.expected for c in cases]
    y_pred = [c.predicted or "missed" for c in cases]
    labels = [*LABELS, "missed"]
    per_type: dict[str, dict[str, float]] = {}
    for name in dict.fromkeys(c.scenario for c in cases):
        subset = [c for c in cases if c.scenario == name]
        per_type[name] = {
            "accuracy": float(np.mean([c.correct for c in subset])),
            "containment_correct": float(np.mean([c.containment_correct for c in subset])),
        }
    return {
        "cases": len(cases),
        "accuracy": float(np.mean([c.correct for c in cases])),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        ),
        "containment_correct": float(np.mean([c.containment_correct for c in cases])),
        "collateral_clients_total": int(sum(c.collateral_clients for c in cases)),
        "denied_actions_total": int(sum(c.denied_actions for c in cases)),
        "outcomes": dict(Counter(c.outcome for c in cases)),
        "mean_turns": float(np.mean([c.turns for c in cases])),
        "mean_input_tokens": float(np.mean([c.input_tokens for c in cases])),
        "mean_output_tokens": float(np.mean([c.output_tokens for c in cases])),
        "mean_llm_seconds": float(np.mean([c.llm_seconds for c in cases])),
        "per_scenario": per_type,
        "labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def cases_to_rows(cases: Sequence[CaseResult]) -> list[dict[str, Any]]:
    return [asdict(c) for c in cases]
