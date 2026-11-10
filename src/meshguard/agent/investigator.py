import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from meshguard.agent.context import Investigation
from meshguard.agent.llm.base import (
    AssistantTurn,
    ChatModel,
    LLMError,
    ToolCall,
    ToolOutcome,
    Usage,
)
from meshguard.agent.prompts import NUDGE, signal_message, system_prompt
from meshguard.agent.tools import Toolbox
from meshguard.clock import Clock
from meshguard.domain import AnomalySignal, Severity

log = logging.getLogger(__name__)


class Outcome(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class InvestigationResult:
    investigation: Investigation
    model: str
    outcome: Outcome
    turns: int
    usage: Usage
    wall_seconds: float
    final_text: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        inv = self.investigation
        return {
            "incident_id": inv.incident_id,
            "model": self.model,
            "outcome": self.outcome.value,
            "turns": self.turns,
            "usage": asdict(self.usage),
            "wall_seconds": round(self.wall_seconds, 2),
            "signal": inv.signal.to_dict(),
            "classification": asdict(inv.classification) if inv.classification else None,
            "revisions": [asdict(r) for r in inv.revisions],
            "blocked": inv.blocked,
            "isolated": inv.isolated,
            "denied": inv.denied,
            "notifications": [s.value for s in inv.notifications],
            "final_text": self.final_text,
            "error": self.error,
            "audit": [
                {**asdict(a), "at": a.at.isoformat(), "result": a.result[:4000]} for a in inv.audit
            ],
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.investigation.incident_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path


class InvestigationObserver(Protocol):
    def on_start(self, investigation: Investigation) -> None: ...

    def on_thought(self, investigation: Investigation, text: str) -> None: ...

    def on_tool(
        self, investigation: Investigation, call: ToolCall, outcome: ToolOutcome
    ) -> None: ...

    def on_finish(self, result: InvestigationResult) -> None: ...


class SilentObserver:
    def on_start(self, investigation: Investigation) -> None:
        pass

    def on_thought(self, investigation: Investigation, text: str) -> None:
        pass

    def on_tool(self, investigation: Investigation, call: ToolCall, outcome: ToolOutcome) -> None:
        pass

    def on_finish(self, result: InvestigationResult) -> None:
        pass


class Investigator:
    """LLM agent loop: investigate a signal with tools, classify, contain, report."""

    def __init__(
        self,
        model: ChatModel,
        toolbox: Toolbox,
        clock: Clock,
        *,
        max_turns: int = 14,
        timeout: timedelta = timedelta(minutes=5),
        language: str = "Russian",
        observer: InvestigationObserver | None = None,
    ) -> None:
        self.model = model
        self.toolbox = toolbox
        self.clock = clock
        self.max_turns = max_turns
        self.timeout = timeout
        self.language = language
        self.observer = observer or SilentObserver()

    async def investigate(self, signal: AnomalySignal) -> InvestigationResult:
        investigation = Investigation(signal, self.clock.now())
        self.observer.on_start(investigation)
        started = time.perf_counter()
        state = _LoopState()
        try:
            async with asyncio.timeout(self.timeout.total_seconds()):
                outcome = await self._loop(investigation, state)
            error = None
        except (LLMError, TimeoutError) as exc:
            outcome, error = Outcome.FAILED, str(exc) or type(exc).__name__
            log.warning("investigation %s failed: %s", investigation.incident_id, error)

        if outcome is not Outcome.COMPLETED:
            self._escalate(investigation, outcome, error)
        result = InvestigationResult(
            investigation=investigation,
            model=self.model.name,
            outcome=outcome,
            turns=state.turns,
            usage=state.usage,
            wall_seconds=time.perf_counter() - started,
            final_text=state.final_text,
            error=error,
        )
        self.observer.on_finish(result)
        return result

    async def _loop(self, investigation: Investigation, state: _LoopState) -> Outcome:
        session = self.model.session(system_prompt(self.language), self.toolbox.specs)
        turn = await state.track(
            session.send(text=signal_message(investigation.signal, self.clock.now()))
        )
        nudged = False
        while state.turns < self.max_turns:
            if turn.text:
                state.final_text = turn.text
                self.observer.on_thought(investigation, turn.text)
            if turn.tool_calls:
                outcomes = []
                for call in turn.tool_calls:
                    outcome = self.toolbox.execute(call, investigation)
                    self.observer.on_tool(investigation, call, outcome)
                    outcomes.append(outcome)
                turn = await state.track(session.send(results=outcomes))
                continue
            if investigation.is_complete:
                return Outcome.COMPLETED
            if nudged:
                break
            nudged = True
            turn = await state.track(
                session.send(text=NUDGE.format(missing=_missing(investigation)))
            )
        return Outcome.COMPLETED if investigation.is_complete else Outcome.INCOMPLETE

    def _escalate(self, investigation: Investigation, outcome: Outcome, error: str | None) -> None:
        signal = investigation.signal
        classification = investigation.classification
        summary = (
            f"Automated investigation ended as **{outcome.value}**"
            + (f" ({error})" if error else "")
            + f". Detector flagged `{signal.service}` with score {signal.score:.3f} "
            f"(threshold {signal.threshold:.3f}); correlated: "
            f"{', '.join(signal.correlated_services) or 'none'}."
        )
        if classification is not None:
            summary += (
                f" Last hypothesis: {classification.incident_type.value} "
                f"({classification.confidence:.0%}): {classification.summary}"
            )
        self.toolbox.controls.notifier.send(
            incident_id=investigation.incident_id,
            severity=Severity.HIGH,
            title=f"Manual triage required for anomaly on {signal.service}",
            summary=summary,
            recommendations=("Review the audit trail and continue the investigation manually.",),
        )
        investigation.notifications.append(Severity.HIGH)


@dataclass(slots=True)
class _LoopState:
    turns: int = 0
    usage: Usage = field(default_factory=Usage)
    final_text: str = ""

    async def track(self, pending: Any) -> AssistantTurn:
        turn: AssistantTurn = await pending
        self.turns += 1
        self.usage = self.usage + turn.usage
        return turn


def _missing(investigation: Investigation) -> str:
    gaps = []
    if investigation.classification is None:
        gaps.append("no classification has been recorded")
    if not investigation.notifications:
        gaps.append("the administrator has not been notified")
    return " and ".join(gaps)
