"""Composition root: wires settings into a ready-to-run pipeline."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from meshguard.agent.investigator import InvestigationObserver, Investigator
from meshguard.agent.llm.base import ChatModel
from meshguard.agent.tools import Toolbox
from meshguard.clock import EventClock
from meshguard.config import LLMSettings, Settings
from meshguard.detection.detector import AnomalyDetector
from meshguard.log_store import LogStore
from meshguard.pipeline import Pipeline, PipelineObserver
from meshguard.response.actions import ResponseControls
from meshguard.response.policy import ResponsePolicy
from meshguard.simulation.topology import default_topology


def build_model(settings: LLMSettings) -> ChatModel:
    match settings.provider:
        case "anthropic":
            from meshguard.agent.llm.claude import ClaudeChatModel  # noqa: PLC0415

            return ClaudeChatModel(model=settings.model, effort=settings.effort)
        case "openai":
            from meshguard.agent.llm.openai_compat import (  # noqa: PLC0415
                OpenAICompatibleChatModel,
            )

            return OpenAICompatibleChatModel(model=settings.model, base_url=settings.base_url)
        case "heuristic":
            from meshguard.agent.llm.heuristic import HeuristicChatModel  # noqa: PLC0415

            return HeuristicChatModel()


@dataclass(frozen=True, slots=True)
class Runtime:
    clock: EventClock
    store: LogStore
    controls: ResponseControls
    investigator: Investigator
    pipeline: Pipeline


def build_runtime(
    settings: Settings,
    detector: AnomalyDetector,
    *,
    start: datetime,
    model: ChatModel | None = None,
    blocking: bool = False,
    pipeline_observer: PipelineObserver | None = None,
    investigation_observer: InvestigationObserver | None = None,
) -> Runtime:
    clock = EventClock(start)
    store = LogStore(retention=timedelta(hours=1))
    controls = ResponseControls(clock, outbox=settings.outbox_dir)
    toolbox = Toolbox(
        store=store,
        profile=detector.profile,
        inventory=detector.inventory,
        controls=controls,
        policy=ResponsePolicy(settings.policy),
        clock=clock,
    )
    investigator = Investigator(
        model or build_model(settings.llm),
        toolbox,
        clock,
        max_turns=settings.llm.max_turns,
        timeout=timedelta(seconds=settings.llm.timeout_seconds),
        language=settings.llm.language,
        observer=investigation_observer,
    )
    pipeline = Pipeline(
        detector=detector,
        investigator=investigator,
        store=store,
        clock=clock,
        services=default_topology().services,
        blocking=blocking,
        observer=pipeline_observer,
    )
    return Runtime(clock, store, controls, investigator, pipeline)
