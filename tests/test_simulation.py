from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from ipaddress import IPv4Network

import pytest

from meshguard.clock import EventClock
from meshguard.domain import EGRESS, LogEvent
from meshguard.response.actions import ResponseControls
from meshguard.simulation.scenarios import CORE_NAMES, HARD_NAMES, SCENARIOS, build_scenario
from meshguard.simulation.topology import Endpoint, EntryRoute, Topology, default_topology
from meshguard.simulation.traffic import TrafficGenerator

SPAN = timedelta(minutes=2)


def generate(start: datetime, *, seed: int = 1, scenario: str | None = None) -> list[LogEvent]:
    scenarios = [build_scenario(scenario, start, SPAN, seed=seed)] if scenario else []
    generator = TrafficGenerator(default_topology(), start=start, seed=seed, scenarios=scenarios)
    return list(generator.run(SPAN))


def fingerprint(events: list[LogEvent]) -> Counter[str]:
    counts = Counter(str(e.status) for e in events)
    counts["gateway"] = sum(e.target == "gateway" for e in events)
    counts["export"] = sum(e.path == "/v1/users/export" for e in events)
    counts["5xx"] = sum(e.status >= 500 for e in events)
    return counts


def test_generation_is_deterministic(start: datetime) -> None:
    assert generate(start, seed=3) == generate(start, seed=3)
    assert generate(start, seed=3) != generate(start, seed=4)


def test_events_are_well_formed(start: datetime) -> None:
    events = generate(start)
    assert {e.target for e in events} >= {"gateway", "catalog", "auth", EGRESS}
    assert all(e.source != e.target for e in events)
    assert all(e.dest_ip is not None for e in events if e.target == EGRESS)


def test_topology_rejects_dangling_calls() -> None:
    with pytest.raises(ValueError, match="unknown endpoints"):
        Topology([Endpoint("a", "GET", "/", 1, 1, ("b GET /",))], [EntryRoute("a GET /", 1.0)])


@pytest.mark.parametrize(
    ("scenario", "check"),
    [
        ("ddos", lambda c, base: c["gateway"] > 5 * base["gateway"]),
        ("brute_force", lambda c, base: c["401"] > 20 * max(base["401"], 1)),
        ("software_fault", lambda c, base: c["5xx"] > 4 * max(base["5xx"], 1)),
        ("data_exfiltration", lambda c, base: c["export"] > 50 and base["export"] == 0),
        ("benign", lambda c, base: c["gateway"] > 1.8 * base["gateway"]),
    ],
)
def test_scenarios_leave_their_fingerprint(
    start: datetime, scenario: str, check: Callable[[Counter[str], Counter[str]], bool]
) -> None:
    assert check(fingerprint(generate(start, scenario=scenario)), fingerprint(generate(start)))


def test_every_scenario_is_registered() -> None:
    assert set(CORE_NAMES) | set(HARD_NAMES) == set(SCENARIOS)
    assert {SCENARIOS[n].kind for n in CORE_NAMES} == {SCENARIOS[n].kind for n in HARD_NAMES}
    with pytest.raises(ValueError, match="unknown scenario"):
        build_scenario("meteor", datetime.now(), SPAN)  # noqa: DTZ005


def test_enforcement_drops_blocked_sources_and_isolated_workloads(start: datetime) -> None:
    controls = ResponseControls(EventClock(start))
    flood = build_scenario("ddos", start, SPAN)
    exfil = build_scenario("data_exfiltration", start, SPAN)
    controls.firewall.block(
        [IPv4Network(ip) for ip in flood.offender_ips],
        reason="test",
        ttl=timedelta(hours=1),
        incident_id="inc-test",
    )
    controls.mesh.isolate("notifications", reason="test", incident_id="inc-test")
    generator = TrafficGenerator(
        default_topology(), start=start, seed=1, scenarios=[flood, exfil], enforcement=controls
    )
    events = list(generator.run(SPAN))
    assert not any(e.client_ip in flood.offender_ips for e in events)
    assert not any(e.path == "/v1/users/export" for e in events)
    assert all(e.status == 503 for e in events if e.target == "notifications")
