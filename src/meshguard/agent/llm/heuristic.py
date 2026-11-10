"""Rule-based stand-in for the LLM: same tools and protocol, hand-written decision logic.

It serves as an offline baseline for benchmarking and lets the full agent loop run in
tests and air-gapped demos.
"""

import json
import re
from collections.abc import Sequence
from itertools import count
from typing import Any

from meshguard.agent.llm.base import AssistantTurn, StopReason, ToolCall, ToolOutcome, ToolSpec
from meshguard.domain import IncidentType, Severity

type Json = dict[str, Any]

_SIGNAL = re.compile(r"<signal>\s*(\{.*?\})\s*</signal>", re.DOTALL)


class HeuristicChatModel:
    @property
    def name(self) -> str:
        return "heuristic-baseline"

    def session(self, system: str, tools: Sequence[ToolSpec]) -> HeuristicSession:  # noqa: ARG002
        return HeuristicSession()


class HeuristicSession:
    def __init__(self) -> None:
        self._ids = count(1)
        self._pending: dict[str, str] = {}
        self._observations: dict[str, Json] = {}
        self._signal: Json = {}
        self._decided = False

    def _call(self, name: str, **arguments: Any) -> ToolCall:
        call = ToolCall(f"h{next(self._ids)}", name, arguments)
        self._pending[call.id] = name
        return call

    async def send(
        self, *, text: str | None = None, results: Sequence[ToolOutcome] = ()
    ) -> AssistantTurn:
        for result in results:
            name = self._pending.pop(result.call_id, "")
            if not result.is_error:
                self._observations[name] = json.loads(result.content)
        if text and (match := _SIGNAL.search(text)):
            self._signal = json.loads(match.group(1))
            service = self._signal["service"]
            calls = (
                self._call(
                    "service_health", service=service, lookback_seconds=600, bucket_seconds=30
                ),
                self._call(
                    "top_clients", service=service, lookback_seconds=120, rank_by="errors", limit=50
                ),
                self._call("call_graph", lookback_seconds=300),
            )
            return AssistantTurn(
                "Collecting service health, client mix and call graph.", calls, StopReason.TOOL_USE
            )
        if self._decided:
            return AssistantTurn("Investigation finished.", (), StopReason.END_TURN)
        self._decided = True
        return AssistantTurn("Applying decision rules.", self._decide(), StopReason.TOOL_USE)

    def _decide(self) -> tuple[ToolCall, ...]:
        service = self._signal.get("service", "gateway")
        health = self._observations.get("service_health", {})
        clients = self._observations.get("top_clients", {})
        graph = self._observations.get("call_graph", {})

        recent = health.get("series", [])[-2:] or [{}]
        baseline = health.get("baseline", {})
        rate_ratio = max(b.get("requests_per_s", 0) for b in recent) / max(
            baseline.get("requests_per_s", 0), 0.1
        )
        server_errors = max(b.get("server_error_rate", 0) for b in recent)
        auth_failures = max(b.get("auth_failure_rate", 0) for b in recent)
        novel = [e for e in graph.get("edges", []) if e.get("novel_routes")]
        new_subnets = [
            s
            for s in clients.get("top_subnets_16", [])
            if s["requests_in_prior_30m"] == 0 and s["share"] >= 0.03
        ]

        actions: list[ToolCall] = []
        if novel:
            kind, confidence = IncidentType.DATA_EXFILTRATION, 0.8
            compromised = novel[0]["edge"].split(" -> ")[0]
            destinations = sorted(
                e["edge"].split("egress:")[1] for e in novel if "egress:" in e["edge"]
            )
            evidence = [f"novel edge {e['edge']} carrying {e['mb']} MB" for e in novel[:3]]
            if destinations:
                actions.append(
                    self._call(
                        "block_ips",
                        targets=destinations,
                        reason="exfiltration destination",
                        ttl_minutes=1440,
                    )
                )
            actions.append(
                self._call(
                    "isolate_service", service=compromised, reason="workload exfiltrating user data"
                )
            )
        elif auth_failures > 0.15:
            kind, confidence = IncidentType.BRUTE_FORCE, 0.85
            offenders = [
                c["ip"]
                for c in clients.get("clients", [])
                if c["auth_failures"] >= 3 and c["error_rate"] > 0.5
            ]
            evidence = [
                f"auth failure rate {auth_failures:.0%}",
                f"{len(offenders)} sources with repeated failed logins",
            ]
            if offenders:
                actions.append(
                    self._call(
                        "block_ips",
                        targets=offenders,
                        reason="credential stuffing",
                        ttl_minutes=120,
                    )
                )
        elif rate_ratio > 3 and new_subnets:
            kind, confidence = IncidentType.DDOS, 0.8
            evidence = [
                f"request rate x{rate_ratio:.1f} of baseline",
                f"{len(new_subnets)} previously unseen /16 subnets dominate traffic",
            ]
            actions.append(
                self._call(
                    "block_ips",
                    targets=[s["subnet"] for s in new_subnets],
                    reason="HTTP flood sources",
                    ttl_minutes=60,
                )
            )
        elif server_errors > 0.05 and rate_ratio < 2:
            kind, confidence = IncidentType.SOFTWARE_FAULT, 0.75
            messages = [m["message"] for m in health.get("top_error_messages", [])[:2]]
            evidence = [f"server error rate {server_errors:.0%} at normal volume", *messages]
        else:
            kind, confidence = IncidentType.BENIGN, 0.6
            evidence = [f"request rate x{rate_ratio:.1f}, server errors {server_errors:.1%}"]

        severity = {
            IncidentType.BENIGN: Severity.LOW,
            IncidentType.SOFTWARE_FAULT: Severity.HIGH,
        }.get(kind, Severity.CRITICAL)
        summary = f"Heuristic triage of {service}: {kind.value}. " + "; ".join(evidence)
        return (
            self._call(
                "classify_incident",
                incident_type=kind.value,
                confidence=confidence,
                affected_services=[service],
                summary=summary,
                evidence=evidence,
            ),
            *actions,
            self._call(
                "notify_admin",
                severity=severity.value,
                title=f"{kind.value} on {service}",
                summary=summary,
                recommendations=["Validate the automated triage."],
            ),
        )
