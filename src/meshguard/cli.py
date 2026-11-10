import asyncio
import gzip
import json
import logging
from dataclasses import asdict
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from meshguard.config import Provider, Settings
from meshguard.simulation.scenarios import CORE_NAMES, HARD_NAMES

if TYPE_CHECKING:
    from meshguard.detection.detector import AnomalyDetector

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Anomaly detection and autonomous incident response for a microservice mesh.",
)
console = Console()

SCENARIO_NAMES = (*CORE_NAMES, *HARD_NAMES)


class Suite(StrEnum):
    CORE = "core"
    HARD = "hard"
    ALL = "all"


SUITES = {Suite.CORE: CORE_NAMES, Suite.HARD: HARD_NAMES, Suite.ALL: SCENARIO_NAMES}
ProviderOption = Annotated[Provider | None, typer.Option(help="LLM backend override")]
ModelOption = Annotated[str | None, typer.Option(help="Model id override")]


def _settings(provider: Provider | None = None, model: str | None = None) -> Settings:
    settings = Settings()
    updates = {k: v for k, v in {"provider": provider, "model": model}.items() if v is not None}
    if updates:
        settings = settings.model_copy(update={"llm": settings.llm.model_copy(update=updates)})
    return settings


def _load_detector(settings: Settings) -> AnomalyDetector:
    from meshguard.detection.detector import AnomalyDetector  # noqa: PLC0415

    if not settings.detector_path.exists():
        console.print(f"[red]{settings.detector_path} not found; run `meshguard train` first")
        raise typer.Exit(1)
    return AnomalyDetector.load(settings.detector_path)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


@app.command()
def train(
    hours: Annotated[float | None, typer.Option(help="Hours of normal traffic")] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    start_hour: Annotated[int, typer.Option(min=0, max=23, help="Hour of day to start at")] = 0,
) -> None:
    """Simulate normal traffic and fit the anomaly detector."""
    from meshguard.detection.datasets import DEFAULT_START, baseline_windows  # noqa: PLC0415
    from meshguard.detection.detector import AnomalyDetector  # noqa: PLC0415

    settings = _settings()
    cfg = settings.detector
    with console.status("simulating baseline traffic"):
        windows, inventory = baseline_windows(
            timedelta(hours=hours or cfg.training_hours),
            seed=seed or cfg.seed,
            start=DEFAULT_START.replace(hour=start_hour),
        )
    with console.status(f"fitting on {len(windows)} windows"):
        detector = AnomalyDetector.fit(windows, inventory, target_fpr=cfg.target_fpr)
    detector.save(settings.detector_path)
    console.print(
        f"[green]saved[/] {settings.detector_path}: {len(windows)} windows, "
        f"{len(inventory.routes)} known routes, threshold {detector.threshold:.4f}"
    )


@app.command()
def evaluate(
    seeds: Annotated[int, typer.Option(help="Runs per scenario")] = 3,
    output: Annotated[Path, typer.Option()] = Path("reports/detection"),
) -> None:
    """Compare detectors on labelled scenario runs and render report figures."""
    from meshguard.detection.datasets import (  # noqa: PLC0415
        DEFAULT_START,
        baseline_windows,
        scenario_run,
    )
    from meshguard.detection.evaluation import evaluate_scorers  # noqa: PLC0415
    from meshguard.detection.scorers import default_scorers  # noqa: PLC0415
    from meshguard.reporting import plots  # noqa: PLC0415

    settings = _settings()
    cfg = settings.detector
    with console.status("building datasets"):
        train_windows, inventory = baseline_windows(
            timedelta(hours=cfg.training_hours), seed=cfg.seed
        )
        clean, _ = baseline_windows(
            timedelta(hours=6), seed=cfg.seed + 1000, start=DEFAULT_START.replace(hour=8)
        )
        runs = [
            scenario_run(name, inventory=inventory, seed=seed)
            for name in CORE_NAMES
            for seed in range(1, seeds + 1)
        ]
    with console.status("fitting and scoring detectors"):
        evaluation = evaluate_scorers(
            train_windows, inventory, runs, clean, default_scorers(), target_fpr=cfg.target_fpr
        )

    table = Table(title="Detector comparison")
    for column in ("model", "ROC-AUC", "PR-AUC", "P", "R", "F1", "FA/h", "fit s", "µs/win"):
        table.add_column(column, justify="right")
    for r in evaluation.reports:
        table.add_row(
            r.name,
            f"{r.roc_auc:.3f}",
            f"{r.pr_auc:.3f}",
            f"{r.precision:.3f}",
            f"{r.recall:.3f}",
            f"{r.f1:.3f}",
            f"{r.false_alarms_per_hour:.2f}",
            f"{r.fit_seconds:.1f}",
            f"{r.score_us_per_window:.0f}",
        )
    console.print(table)

    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "training_windows": len(train_windows),
        "clean_hours": 6,
        "runs": len(runs),
        "reports": [asdict(r) for r in evaluation.reports],
    }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    best = evaluation.detectors[evaluation.reports[0].name]
    one_per_kind = [next(r for r in runs if r.scenario.name == name) for name in CORE_NAMES]
    for path in (
        plots.roc_pr_curves(evaluation, output / "roc_pr.png"),
        plots.score_timelines(best, one_per_kind, output / "score_timeline.png"),
        plots.deviation_heatmap(best, runs, output / "deviation_heatmap.png"),
    ):
        console.print(f"[green]wrote[/] {path}")


@app.command()
def simulate(
    output: Annotated[Path, typer.Argument(help="Destination .jsonl.gz")],
    minutes: Annotated[int, typer.Option()] = 10,
    scenario: Annotated[
        str | None, typer.Option(help=f"One of {', '.join(SCENARIO_NAMES)}")
    ] = None,
    seed: Annotated[int, typer.Option()] = 1,
) -> None:
    """Dump simulated logs (optionally with an incident in the middle) for inspection."""
    from meshguard.detection.datasets import DEFAULT_START  # noqa: PLC0415
    from meshguard.simulation.scenarios import build_scenario  # noqa: PLC0415
    from meshguard.simulation.topology import default_topology  # noqa: PLC0415
    from meshguard.simulation.traffic import TrafficGenerator  # noqa: PLC0415

    start = DEFAULT_START.replace(hour=13)
    duration = timedelta(minutes=minutes)
    scenarios = (
        [build_scenario(scenario, start + duration / 3, duration / 3, seed=seed)]
        if scenario
        else []
    )
    generator = TrafficGenerator(default_topology(), start=start, seed=seed, scenarios=scenarios)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(output, "wt", encoding="utf-8") as sink:
        for event in generator.run(duration):
            sink.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    console.print(f"[green]wrote[/] {count} events to {output}")


@app.command()
def run(
    scenario: Annotated[str, typer.Argument(help=f"One of {', '.join(SCENARIO_NAMES)}")],
    provider: ProviderOption = None,
    model: ModelOption = None,
    speed: Annotated[float, typer.Option(help="Simulated seconds per real second; 0 = max")] = 0,
    lead: Annotated[int, typer.Option(help="Minutes of normal traffic first")] = 4,
    duration: Annotated[int, typer.Option(help="Incident minutes")] = 5,
    tail: Annotated[int, typer.Option(help="Minutes after the incident")] = 3,
    seed: Annotated[int, typer.Option()] = 1,
) -> None:
    """Live demo: stream traffic with an injected incident through detector and agent."""
    from meshguard.app import build_runtime  # noqa: PLC0415
    from meshguard.console import ConsoleObserver  # noqa: PLC0415
    from meshguard.detection.datasets import DEFAULT_START  # noqa: PLC0415
    from meshguard.pipeline import simulated_stream  # noqa: PLC0415
    from meshguard.simulation.scenarios import build_scenario  # noqa: PLC0415
    from meshguard.simulation.topology import default_topology  # noqa: PLC0415
    from meshguard.simulation.traffic import TrafficGenerator  # noqa: PLC0415

    settings = _settings(provider, model)
    detector = _load_detector(settings)
    observer = ConsoleObserver(console)
    start = DEFAULT_START.replace(hour=13)
    runtime = build_runtime(
        settings,
        detector,
        start=start,
        blocking=speed == 0,
        pipeline_observer=observer,
        investigation_observer=observer,
    )
    incident = build_scenario(
        scenario, start + timedelta(minutes=lead), timedelta(minutes=duration), seed=seed
    )
    generator = TrafficGenerator(
        default_topology(),
        start=start,
        seed=seed,
        scenarios=[incident],
        enforcement=runtime.controls,
    )
    console.print(
        f"[bold]{incident.name}[/] from {incident.start:%H:%M:%S} to {incident.end:%H:%M:%S}, "
        f"agent: {runtime.investigator.model.name}"
    )
    stream = simulated_stream(
        generator, timedelta(minutes=lead + duration + tail), speed=speed or None
    )
    report = asyncio.run(runtime.pipeline.run(stream))
    for result in report.investigations:
        path = result.save(settings.incidents_dir)
        console.print(f"audit trail: {path}")
    console.print(
        f"{report.events} events, {report.windows} windows, {len(report.signals)} signals; "
        f"reports in {settings.outbox_dir}"
    )


@app.command()
def benchmark(
    provider: ProviderOption = None,
    model: ModelOption = None,
    seeds: Annotated[int, typer.Option(help="Runs per scenario")] = 3,
    suite: Annotated[Suite, typer.Option()] = Suite.CORE,
    scenarios: Annotated[list[str] | None, typer.Option("--scenario")] = None,
    output: Annotated[Path, typer.Option()] = Path("reports/agent"),
) -> None:
    """Measure classification accuracy and response correctness of the agent end-to-end."""
    from meshguard.app import build_model  # noqa: PLC0415
    from meshguard.benchmark import cases_to_rows, run_case, summarize  # noqa: PLC0415
    from meshguard.reporting import plots  # noqa: PLC0415

    settings = _settings(provider, model)
    detector = _load_detector(settings)
    llm = build_model(settings.llm)
    label = llm.name.replace("/", "_")
    destination = output / label / suite.value
    cases = []
    for name in scenarios or SUITES[suite]:
        for seed in range(1, seeds + 1):
            with console.status(f"{name} seed {seed} with {llm.name}"):
                case, results = asyncio.run(run_case(settings, detector, llm, name, seed))
            for result in results:
                result.save(destination / "incidents")
            cases.append(case)
            verdict = "[green]ok" if case.correct else "[red]wrong"
            console.print(
                f"{name:24s} seed={seed} → {case.predicted or '-':18s} {verdict}[/]  "
                f"containment={'ok' if case.containment_correct else 'WRONG'}  "
                f"tokens={case.input_tokens}/{case.output_tokens}  llm={case.llm_seconds:.1f}s"
            )
    summary = summarize(cases)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(
            {"model": llm.name, **summary, "runs": cases_to_rows(cases)},
            indent=2,
            ensure_ascii=False,
        )
    )
    plots.confusion_heatmap(summary, destination / "confusion.png", llm.name)
    console.print(
        f"accuracy {summary['accuracy']:.2%}, macro-F1 {summary['macro_f1']:.3f}, "
        f"containment {summary['containment_correct']:.2%} → {destination}"
    )
