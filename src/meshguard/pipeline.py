import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol

from meshguard.agent.investigator import InvestigationResult, Investigator
from meshguard.clock import EventClock
from meshguard.detection.alerting import SignalGate
from meshguard.detection.detector import AnomalyDetector, Detection
from meshguard.detection.evaluation import by_window
from meshguard.detection.features import WindowAggregator, WindowFeatures
from meshguard.domain import AnomalySignal, LogEvent
from meshguard.log_store import LogStore
from meshguard.simulation.traffic import TrafficGenerator

log = logging.getLogger(__name__)


class PipelineObserver(Protocol):
    def on_window(self, detections: Sequence[Detection]) -> None: ...

    def on_signal(self, signal: AnomalySignal) -> None: ...


@dataclass(slots=True)
class PipelineReport:
    events: int = 0
    windows: int = 0
    signals: list[AnomalySignal] = field(default_factory=list)
    investigations: list[InvestigationResult] = field(default_factory=list)


async def simulated_stream(
    generator: TrafficGenerator, duration: timedelta, *, speed: float | None = None
) -> AsyncIterator[list[LogEvent]]:
    """Yield one simulated second at a time, paced at ``speed`` x real time (None = flat out)."""
    end = generator.clock + duration
    while generator.clock < end:
        yield generator.tick()
        await asyncio.sleep(1 / speed if speed else 0)


class Pipeline:
    """Continuous detection over a log stream with asynchronous hand-off to the agent."""

    def __init__(
        self,
        *,
        detector: AnomalyDetector,
        investigator: Investigator,
        store: LogStore,
        clock: EventClock,
        services: Sequence[str],
        gate: SignalGate | None = None,
        workers: int = 2,
        blocking: bool = False,
        observer: PipelineObserver | None = None,
    ) -> None:
        self.detector = detector
        self.investigator = investigator
        self.store = store
        self.clock = clock
        self.gate = gate or SignalGate()
        self.aggregator = WindowAggregator(services, detector.inventory, size=detector.window_size)
        self.workers = workers
        self.blocking = blocking
        self.observer = observer
        self.report = PipelineReport()

    async def run(self, stream: AsyncIterator[list[LogEvent]]) -> PipelineReport:
        queue: asyncio.Queue[AnomalySignal | None] = asyncio.Queue()
        async with asyncio.TaskGroup() as group:
            for _ in range(self.workers):
                group.create_task(self._worker(queue))
            try:
                async for batch in stream:
                    for signal in self._ingest(batch):
                        if self.blocking:
                            await self._investigate(signal)
                        else:
                            queue.put_nowait(signal)
                for signal in self._process(self.aggregator.close()):
                    queue.put_nowait(signal)
            finally:
                for _ in range(self.workers):
                    queue.put_nowait(None)
        return self.report

    def _ingest(self, batch: list[LogEvent]) -> list[AnomalySignal]:
        closed: list[WindowFeatures] = []
        for event in batch:
            self.clock.advance(event.ts)
            self.store.append(event)
            closed.extend(self.aggregator.add(event))
        self.report.events += len(batch)
        return self._process(closed)

    def _process(self, windows: list[WindowFeatures]) -> list[AnomalySignal]:
        signals = []
        for group in by_window(windows):
            detections = self.detector.detect_many(group)
            self.report.windows += len(group)
            if self.observer is not None:
                self.observer.on_window(detections)
            if (signal := self.gate.observe(detections)) is not None:
                self.report.signals.append(signal)
                if self.observer is not None:
                    self.observer.on_signal(signal)
                signals.append(signal)
        return signals

    async def _worker(self, queue: asyncio.Queue[AnomalySignal | None]) -> None:
        while (signal := await queue.get()) is not None:
            await self._investigate(signal)

    async def _investigate(self, signal: AnomalySignal) -> None:
        result = await self.investigator.investigate(signal)
        self.report.investigations.append(result)
