import json
from datetime import datetime, timedelta
from ipaddress import IPv4Network

import pytest

from meshguard.agent.context import Investigation
from meshguard.agent.llm.base import ToolCall, strict_schema
from meshguard.agent.tools import NotifyArgs, Toolbox
from meshguard.clock import EventClock
from meshguard.detection.detector import AnomalyDetector
from meshguard.domain import AnomalySignal, IncidentType, LogEvent
from meshguard.log_store import LogStore
from meshguard.response.actions import ResponseControls
from meshguard.response.policy import Classification, ResponsePolicy

DDOS = Classification(IncidentType.DDOS, 0.9)


class TestPolicy:
    policy = ResponsePolicy()

    def test_requires_classification_and_matching_type(self) -> None:
        nets = [IPv4Network("45.141.0.0/16")]
        assert not self.policy.check_block(nets, None).allowed
        assert self.policy.check_block(nets, DDOS).allowed
        fault = Classification(IncidentType.SOFTWARE_FAULT, 0.99)
        assert "not an approved response" in self.policy.check_block(nets, fault).reason

    def test_low_confidence_blocks_containment(self) -> None:
        decision = self.policy.check_block(
            [IPv4Network("45.141.1.1/32")], Classification(IncidentType.DDOS, 0.4)
        )
        assert not decision.allowed
        assert "escalate" in decision.reason

    @pytest.mark.parametrize(
        ("network", "reason"),
        [
            ("10.0.3.0/24", "protected"),
            ("198.51.100.25/32", "protected"),
            ("45.0.0.0/8", "broader than"),
        ],
    )
    def test_guards_networks(self, network: str, reason: str) -> None:
        decision = self.policy.check_block([IPv4Network(network)], DDOS)
        assert not decision.allowed
        assert reason in decision.reason

    def test_limits_blast_radius(self) -> None:
        many = [IPv4Network(f"45.141.{i // 250}.{i % 250 + 1}/32") for i in range(1100)]
        assert "exceed" in self.policy.check_block(many, DDOS).reason

    def test_isolation_rules(self) -> None:
        exfil = Classification(IncidentType.DATA_EXFILTRATION, 0.9)
        assert self.policy.check_isolation("notifications", exfil).allowed
        assert "human approval" in self.policy.check_isolation("gateway", exfil).reason
        assert "unknown" in self.policy.check_isolation("mainframe", exfil).reason
        assert not self.policy.check_isolation("notifications", DDOS).allowed


@pytest.fixture
def toolbox(detector: AnomalyDetector, start: datetime) -> Toolbox:
    clock = EventClock(start + timedelta(minutes=5))
    store = LogStore()
    store.extend(
        LogEvent(
            ts=start + timedelta(seconds=i),
            trace_id=f"t{i}",
            source="gateway",
            target="auth",
            method="POST",
            path="/v1/sessions",
            status=401,
            latency_ms=40,
            bytes_out=200,
            client_ip=f"194.26.29.{i % 20 + 1}",
            user_id=f"u-{i}",
            message="authentication failed for login=u-1",
        )
        for i in range(240)
    )
    return Toolbox(
        store=store,
        profile=detector.profile,
        inventory=detector.inventory,
        controls=ResponseControls(clock),
        policy=ResponsePolicy(),
        clock=clock,
    )


@pytest.fixture
def investigation(start: datetime) -> Investigation:
    signal = AnomalySignal("auth", start, start + timedelta(seconds=10), 2.0, 1.0, ())
    return Investigation(signal, start)


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(f"c-{name}", name, dict(arguments))


class TestToolbox:
    def test_investigation_tools_summarise_logs(
        self, toolbox: Toolbox, investigation: Investigation
    ) -> None:
        outcome = toolbox.execute(
            call("top_clients", service="auth", rank_by="errors"), investigation
        )
        assert not outcome.is_error
        payload = json.loads(outcome.content)
        assert payload["distinct_clients"] == 20
        assert payload["top_subnets_24"][0]["subnet"] == "194.26.29.0/24"
        assert payload["clients"][0]["auth_failures"] == 12

        health = json.loads(
            toolbox.execute(call("service_health", service="auth"), investigation).content
        )
        assert health["top_error_messages"][0]["count"] == 240

    def test_invalid_arguments_are_reported_back(
        self, toolbox: Toolbox, investigation: Investigation
    ) -> None:
        outcome = toolbox.execute(call("search_logs", service="mainframe"), investigation)
        assert outcome.is_error
        assert "unknown service" in outcome.content
        assert toolbox.execute(call("rm_rf"), investigation).is_error

    def test_containment_is_gated_by_classification(
        self, toolbox: Toolbox, investigation: Investigation
    ) -> None:
        block = call("block_ips", targets=["194.26.29.0/24"], reason="stuffing")
        rejected = toolbox.execute(block, investigation)
        assert rejected.is_error
        assert "classification" in rejected.content
        assert investigation.denied

        toolbox.execute(
            call(
                "classify_incident",
                incident_type="brute_force",
                confidence=0.9,
                affected_services=["auth"],
                summary="stuffing",
                evidence=["240 failed logins"],
            ),
            investigation,
        )
        assert not toolbox.execute(block, investigation).is_error
        assert toolbox.controls.is_blocked("194.26.29.7")
        assert investigation.blocked == ["194.26.29.0/24"]
        assert len(investigation.audit) == 3

    def test_report_completes_investigation(
        self, toolbox: Toolbox, investigation: Investigation
    ) -> None:
        toolbox.execute(
            call(
                "classify_incident",
                incident_type="benign",
                confidence=0.7,
                affected_services=["auth"],
                summary="noise",
                evidence=["normal"],
            ),
            investigation,
        )
        toolbox.execute(
            call("notify_admin", severity="low", title="t", summary="s", recommendations=[]),
            investigation,
        )
        assert investigation.is_complete
        assert toolbox.controls.notifier.sent[0].incident_id == investigation.incident_id


def test_strict_schema_keeps_field_names_and_drops_constraints() -> None:
    schema = strict_schema(NotifyArgs.model_json_schema())
    assert {"title", "summary", "severity"} <= set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert "title" not in schema
    serialized = json.dumps(schema)
    assert "minimum" not in serialized
    assert "default" not in serialized
