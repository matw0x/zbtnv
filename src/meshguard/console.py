import json
from collections.abc import Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from meshguard.agent.context import Investigation
from meshguard.agent.investigator import InvestigationResult, Outcome
from meshguard.agent.llm.base import ToolCall, ToolOutcome
from meshguard.detection.detector import Detection
from meshguard.domain import AnomalySignal

_TOOL_STYLE = {"classify_incident": "bold magenta", "notify_admin": "bold cyan"}
_ACTION_TOOLS = {"block_ips", "isolate_service"}


def _shorten(value: object, limit: int = 140) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


class ConsoleObserver:
    """Human-readable live trace of detection and investigation."""

    def __init__(self, console: Console | None = None, heartbeat_windows: int = 6) -> None:
        self.console = console or Console()
        self.heartbeat_windows = heartbeat_windows
        self._windows = 0

    def on_window(self, detections: Sequence[Detection]) -> None:
        self._windows += 1
        worst = max(detections, key=lambda d: d.score)
        flagged = [d.window.service for d in detections if d.is_anomaly]
        if not flagged and self._windows % self.heartbeat_windows:
            return
        line = Text(f"{worst.window.end:%H:%M:%S} ", style="dim")
        line.append(f"max score {worst.score:.3f} / thr {worst.threshold:.3f}  ")
        if flagged:
            line.append("anomalous: " + ", ".join(flagged), style="bold red")
        else:
            line.append("normal", style="green")
        self.console.print(line)

    def on_signal(self, signal: AnomalySignal) -> None:
        table = Table(box=None, show_header=True, header_style="bold")
        for column in ("feature", "value", "baseline", "robust z"):
            table.add_column(column, justify="right" if column != "feature" else "left")
        for d in signal.deviations:
            table.add_row(d.feature, f"{d.value:,.3f}", f"{d.baseline:,.3f}", f"{d.z_score:+.1f}")
        correlated = ", ".join(signal.correlated_services) or "-"
        title = (
            f"Signal {signal.signal_id}: {signal.service} "
            f"(score {signal.score:.3f}, correlated: {correlated})"
        )
        self.console.print(Panel(table, title=title, border_style="red"))

    def on_start(self, investigation: Investigation) -> None:
        self.console.rule(f"[bold]Investigation {investigation.incident_id}")

    def on_thought(self, investigation: Investigation, text: str) -> None:  # noqa: ARG002
        self.console.print(Text(text, style="italic dim"))

    def on_tool(
        self,
        investigation: Investigation,  # noqa: ARG002
        call: ToolCall,
        outcome: ToolOutcome,
    ) -> None:
        style = _TOOL_STYLE.get(call.name, "bold yellow" if call.name in _ACTION_TOOLS else "blue")
        self.console.print(
            Text("→ ", style=style)
            + Text(call.name, style=style)
            + Text(f" {_shorten(call.arguments)}", style="dim")
        )
        mark = Text("  ✗ ", style="red") if outcome.is_error else Text("  ✓ ", style="green")
        self.console.print(mark + Text(_shorten(outcome.content), style="dim"))

    def on_finish(self, result: InvestigationResult) -> None:
        inv = result.investigation
        c = inv.classification
        lines = [
            f"outcome: {result.outcome.value}  model: {result.model}",
            f"classification: {c.incident_type.value} ({c.confidence:.0%})" if c else "no verdict",
            f"blocked: {len(inv.blocked)} networks  isolated: {', '.join(inv.isolated) or '-'}",
            f"denied by policy: {len(inv.denied)}  reports: {len(inv.notifications)}",
            f"turns: {result.turns}  tokens in/out: {result.usage.input_tokens}/"
            f"{result.usage.output_tokens}  wall: {result.wall_seconds:.1f}s",
        ]
        if c:
            lines.append(f"\n{c.summary}")
        style = "green" if result.outcome is Outcome.COMPLETED else "red"
        self.console.print(Panel("\n".join(lines), title=inv.incident_id, border_style=style))
