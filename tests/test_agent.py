from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic

from meshguard.agent.investigator import Investigator, Outcome
from meshguard.agent.llm.base import (
    AssistantTurn,
    LLMError,
    LLMRefusalError,
    StopReason,
    ToolCall,
    ToolOutcome,
    ToolSpec,
)
from meshguard.agent.llm.claude import ClaudeChatModel
from meshguard.agent.llm.heuristic import HeuristicChatModel
from meshguard.agent.tools import Toolbox
from meshguard.benchmark import run_case
from meshguard.clock import EventClock
from meshguard.config import Settings
from meshguard.detection.detector import AnomalyDetector
from meshguard.domain import AnomalySignal, IncidentType, Severity
from meshguard.log_store import LogStore
from meshguard.response.actions import ResponseControls
from meshguard.response.policy import ResponsePolicy


@dataclass
class ScriptedModel:
    """Replays canned assistant turns and records what the agent sent back."""

    script: Sequence[AssistantTurn | Exception]
    received: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.turns = list(self.script)

    @property
    def name(self) -> str:
        return "scripted"

    def session(self, system: str, tools: Sequence[ToolSpec]) -> ScriptedModel:
        self.received.append({"system": system, "tools": [t.name for t in tools]})
        return self

    async def send(
        self, *, text: str | None = None, results: Sequence[ToolOutcome] = ()
    ) -> AssistantTurn:
        self.received.append({"text": text, "results": list(results)})
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def tool_turn(*calls: ToolCall) -> AssistantTurn:
    return AssistantTurn("", calls, StopReason.TOOL_USE)


def done(text: str = "done") -> AssistantTurn:
    return AssistantTurn(text, (), StopReason.END_TURN)


CLASSIFY = ToolCall(
    "1",
    "classify_incident",
    {
        "incident_type": "software_fault",
        "confidence": 0.8,
        "affected_services": ["payments"],
        "summary": "bad release",
        "evidence": ["NPE in ChargeService"],
    },
)
NOTIFY = ToolCall(
    "2",
    "notify_admin",
    {"severity": "high", "title": "rollback", "summary": "s", "recommendations": ["rollback"]},
)


type InvestigatorFactory = Callable[..., Investigator]


@pytest.fixture
def make_investigator(detector: AnomalyDetector, start: datetime) -> InvestigatorFactory:
    def factory(model: ScriptedModel, **kwargs: Any) -> Investigator:
        clock = EventClock(start)
        toolbox = Toolbox(
            store=LogStore(),
            profile=detector.profile,
            inventory=detector.inventory,
            controls=ResponseControls(clock),
            policy=ResponsePolicy(),
            clock=clock,
        )
        return Investigator(model, toolbox, clock, **kwargs)

    return factory


@pytest.fixture
def signal(start: datetime) -> AnomalySignal:
    return AnomalySignal("payments", start, start + timedelta(seconds=10), 2.0, 1.0, ())


async def test_completes_when_classified_and_reported(
    make_investigator: InvestigatorFactory, signal: AnomalySignal
) -> None:
    model = ScriptedModel([tool_turn(CLASSIFY, NOTIFY), done("rolled back")])
    result = await make_investigator(model).investigate(signal)
    assert result.outcome is Outcome.COMPLETED
    assert result.final_text == "rolled back"
    classification = result.investigation.classification
    assert classification is not None
    assert classification.incident_type is IncidentType.SOFTWARE_FAULT
    assert "<signal>" in model.received[1]["text"]
    assert len(model.received[2]["results"]) == 2


async def test_nudges_once_then_escalates(
    make_investigator: InvestigatorFactory, signal: AnomalySignal
) -> None:
    model = ScriptedModel([done("looks fine"), done("still fine")])
    investigator = make_investigator(model)
    result = await investigator.investigate(signal)
    assert result.outcome is Outcome.INCOMPLETE
    assert "no classification" in model.received[-1]["text"]
    report = investigator.toolbox.controls.notifier.sent[-1]
    assert report.severity is Severity.HIGH
    assert "Manual triage" in report.title


@pytest.mark.parametrize("error", [LLMError("api down"), LLMRefusalError("declined")])
async def test_llm_failure_fails_safe(
    make_investigator: InvestigatorFactory, signal: AnomalySignal, error: LLMError
) -> None:
    investigator = make_investigator(ScriptedModel([tool_turn(CLASSIFY), error]))
    result = await investigator.investigate(signal)
    assert result.outcome is Outcome.FAILED
    assert result.error
    assert investigator.toolbox.controls.notifier.sent


async def test_turn_budget_is_enforced(
    make_investigator: InvestigatorFactory, signal: AnomalySignal
) -> None:
    looping = [tool_turn(ToolCall(str(i), "call_graph", {})) for i in range(10)]
    result = await make_investigator(ScriptedModel(looping), max_turns=3).investigate(signal)
    assert result.outcome is Outcome.INCOMPLETE
    assert result.turns == 3


@pytest.mark.parametrize(
    ("scenario", "expected_isolated"),
    [("data_exfiltration", ["notifications"]), ("ddos", []), ("software_fault", [])],
)
async def test_heuristic_agent_end_to_end(
    detector: AnomalyDetector,
    tmp_path: Path,
    scenario: str,
    expected_isolated: list[str],
) -> None:
    settings = Settings(var_dir=tmp_path)
    case, results = await run_case(
        settings,
        detector,
        HeuristicChatModel(),
        scenario,
        seed=5,
        lead=timedelta(minutes=3),
        duration=timedelta(minutes=3),
        tail=timedelta(minutes=1),
    )
    assert case.predicted == scenario
    assert case.containment_correct
    assert case.isolated == expected_isolated
    assert case.collateral_clients == 0
    assert all(r.outcome is Outcome.COMPLETED for r in results)


class FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.responses.pop(0)


def claude_response(stop_reason: str, *blocks: Any, category: str | None = None) -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category),
        content=list(blocks),
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=60),
    )


async def test_claude_adapter_request_shape_and_history() -> None:
    tool_use = SimpleNamespace(type="tool_use", id="tu_1", name="call_graph", input={})
    messages = FakeMessages(
        [
            claude_response("tool_use", SimpleNamespace(type="text", text="checking"), tool_use),
            claude_response("end_turn", SimpleNamespace(type="text", text="ok")),
        ]
    )
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    model = ClaudeChatModel(effort="high", client=cast("AsyncAnthropic", client))
    session = model.session("system", [ToolSpec("call_graph", "d", {"type": "object"})])

    first = await session.send(text="hello")
    assert first.tool_calls == (ToolCall("tu_1", "call_graph", {}),)
    assert first.usage.cached_input_tokens == 60
    await session.send(results=[ToolOutcome("tu_1", "{}")])

    request = messages.requests[-1]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "high"}
    assert request["fallbacks"] == "default"
    assert request["tools"][0]["strict"] is True
    assert request["tools"][0]["input_schema"]["additionalProperties"] is False
    roles = [m["role"] for m in request["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert request["messages"][2]["content"][0]["tool_use_id"] == "tu_1"
    assert "tool_choice" not in request


async def test_claude_adapter_surfaces_refusals() -> None:
    messages = FakeMessages([claude_response("refusal", category="cyber")])
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    session = ClaudeChatModel(client=cast("AsyncAnthropic", client)).session("s", [])
    with pytest.raises(LLMRefusalError, match="cyber"):
        await session.send(text="hi")


async def test_openai_compatible_adapter_maps_tool_calls() -> None:
    from meshguard.agent.llm.openai_compat import OpenAICompatibleChatModel  # noqa: PLC0415

    message = SimpleNamespace(
        content="checking",
        tool_calls=[
            SimpleNamespace(
                type="function",
                id="call_1",
                function=SimpleNamespace(name="top_clients", arguments='{"service": "auth"}'),
            )
        ],
        model_dump=lambda exclude_none: {"role": "assistant", "content": "checking"},
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    requests: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        requests.append(request)
        return response

    model = OpenAICompatibleChatModel(model="qwen3:14b")
    model.client = cast(
        "Any", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    )
    turn = await model.session("system", [ToolSpec("top_clients", "d", {"type": "object"})]).send(
        text="go"
    )
    assert turn.tool_calls == (ToolCall("call_1", "top_clients", {"service": "auth"}),)
    assert turn.stop_reason is StopReason.TOOL_USE
    assert requests[0]["messages"][0] == {"role": "system", "content": "system"}
    assert requests[0]["tools"][0]["function"]["name"] == "top_clients"
