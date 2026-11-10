import gzip
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from meshguard.cli import app


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MESHGUARD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MESHGUARD_VAR_DIR", str(tmp_path / "var"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_train_then_run_offline(workdir: Path) -> None:
    runner = CliRunner()
    trained = runner.invoke(app, ["train", "--hours", "2", "--start-hour", "12"])
    assert trained.exit_code == 0, trained.output
    assert (workdir / "artifacts" / "detector.joblib").exists()

    result = runner.invoke(
        app,
        [
            "run",
            "data_exfiltration",
            "--provider",
            "heuristic",
            "--lead",
            "2",
            "--duration",
            "2",
            "--tail",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "isolate_service" in result.output
    trails = list((workdir / "var" / "incidents").glob("*.json"))
    assert trails
    assert json.loads(trails[0].read_text())["classification"]["incident_type"] == (
        "data_exfiltration"
    )
    assert list((workdir / "var" / "outbox").glob("*.md"))


def test_run_without_detector_fails_cleanly(workdir: Path) -> None:
    result = CliRunner().invoke(app, ["run", "ddos", "--provider", "heuristic"])
    assert result.exit_code == 1
    assert "meshguard train" in result.output


def test_simulate_dumps_jsonl(workdir: Path) -> None:
    target = workdir / "logs.jsonl.gz"
    result = CliRunner().invoke(
        app, ["simulate", str(target), "--minutes", "1", "--scenario", "ddos"]
    )
    assert result.exit_code == 0, result.output
    with gzip.open(target, "rt") as source:
        first = json.loads(source.readline())
    assert {"ts", "source", "target", "status", "client_ip"} <= first.keys()
