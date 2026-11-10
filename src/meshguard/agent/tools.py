import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from ipaddress import IPv4Network
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from meshguard.agent.context import AuditEntry, ClassificationRecord, Investigation
from meshguard.agent.llm.base import ToolCall, ToolOutcome, ToolSpec
from meshguard.clock import Clock
from meshguard.detection.detector import BaselineProfile
from meshguard.detection.features import RouteInventory, route_key
from meshguard.domain import EGRESS, IncidentType, LogEvent, Severity
from meshguard.log_store import LogStore
from meshguard.response.actions import ResponseControls
from meshguard.response.policy import CONTAINMENT_MATRIX, ResponsePolicy

type Json = dict[str, Any]

HISTORY = timedelta(minutes=30)


class ToolKind(StrEnum):
    INVESTIGATE = "investigate"
    CLASSIFY = "classify"
    RESPOND = "respond"


class ToolRejectedError(Exception):
    """Raised by a handler when the request is valid JSON but cannot be honoured."""


SERVICES: tuple[str, ...] = (
    "gateway",
    "auth",
    "users",
    "orders",
    "payments",
    "inventory",
    "catalog",
    "notifications",
)


def _known_service(name: str) -> str:
    if name not in SERVICES:
        raise ValueError(f"unknown service {name!r}; expected one of {', '.join(SERVICES)}")
    return name


ServiceName = Annotated[str, AfterValidator(_known_service)]
Lookback = Annotated[int, Field(ge=10, le=1800, description="Seconds to look back from now")]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchLogsArgs(_Args):
    service: ServiceName = Field(description="Service whose inbound or outbound spans to search")
    lookback_seconds: Lookback = 300
    min_status: int | None = Field(None, description="Only spans with status >= this value")
    client_ip: str | None = Field(None, description="Exact client IP filter")
    path_contains: str | None = Field(None, description="Substring of the request path")
    limit: int = Field(20, ge=1, le=60, description="Maximum spans to return (newest first)")


class TopClientsArgs(_Args):
    service: ServiceName
    lookback_seconds: Lookback = 300
    rank_by: Literal["requests", "bytes", "errors"] = "requests"
    limit: int = Field(15, ge=1, le=50)


class ServiceHealthArgs(_Args):
    service: ServiceName
    lookback_seconds: Lookback = 600
    bucket_seconds: int = Field(60, ge=10, le=300)


class CallGraphArgs(_Args):
    lookback_seconds: Lookback = 300
    only_service: ServiceName | None = Field(None, description="Restrict to edges touching it")


class TraceArgs(_Args):
    trace_id: str


class ClassifyArgs(_Args):
    incident_type: IncidentType
    confidence: float = Field(ge=0, le=1, description="Calibrated probability the type is right")
    affected_services: list[ServiceName]
    summary: str = Field(description="Root-cause hypothesis in two or three sentences")
    evidence: list[str] = Field(min_length=1, description="Concrete observations backing it")


class BlockArgs(_Args):
    targets: list[str] = Field(
        min_length=1, description="IPv4 addresses or CIDR subnets (no wider than /16)"
    )
    reason: str
    ttl_minutes: int = Field(60, ge=5, le=1440)


class IsolateArgs(_Args):
    service: ServiceName
    reason: str


class NotifyArgs(_Args):
    severity: Severity
    title: str
    summary: str = Field(description="Markdown incident report: what, impact, evidence, actions")
    recommendations: list[str] = Field(description="Follow-up steps for the on-call engineer")


@dataclass(frozen=True, slots=True)
class Tool[A: _Args]:
    name: str
    kind: ToolKind
    description: str
    args: type[A]
    handler: Callable[[A, Investigation], Json]

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.args.model_json_schema())

    def invoke(self, raw: dict[str, Any], investigation: Investigation) -> Json:
        return self.handler(self.args.model_validate(raw), investigation)


def _compact(event: LogEvent) -> Json:
    record: Json = {
        "ts": event.ts.strftime("%H:%M:%S.%f")[:-3],
        "trace": event.trace_id,
        "src": event.source,
        "dst": event.target,
        "req": f"{event.method} {event.path}",
        "status": event.status,
        "ms": round(event.latency_ms, 1),
        "bytes": event.bytes_out,
        "ip": event.client_ip,
    }
    if event.user_id:
        record["user"] = event.user_id
    if event.dest_ip:
        record["dest_ip"] = event.dest_ip
    if event.message:
        record["msg"] = event.message
    return record


def _subnet(ip: str, prefix: int) -> str:
    return str(IPv4Network(f"{ip}/{prefix}", strict=False))


def _p95(values: list[float]) -> float:
    return round(float(np.percentile(values, 95)), 1) if values else 0.0


class Toolbox:
    """Read-only investigation tools plus policy-gated response actions."""

    def __init__(
        self,
        *,
        store: LogStore,
        profile: BaselineProfile,
        inventory: RouteInventory,
        controls: ResponseControls,
        policy: ResponsePolicy,
        clock: Clock,
    ) -> None:
        self.store = store
        self.profile = profile
        self.inventory = inventory
        self.controls = controls
        self.policy = policy
        self.clock = clock
        self.tools: dict[str, Tool[Any]] = {
            tool.name: tool
            for tool in (
                Tool(
                    "search_logs",
                    ToolKind.INVESTIGATE,
                    "Fetch raw spans received or sent by a service, newest first, with filters. "
                    "Use it to read error messages, payload sizes and suspicious requests.",
                    SearchLogsArgs,
                    self.search_logs,
                ),
                Tool(
                    "top_clients",
                    ToolKind.INVESTIGATE,
                    "Rank client IPs calling a service and summarise their /24 and /16 subnets. "
                    "Use it to tell a botnet or a few abusive sources from organic traffic.",
                    TopClientsArgs,
                    self.top_clients,
                ),
                Tool(
                    "service_health",
                    ToolKind.INVESTIGATE,
                    "Time series of rate, error ratios and p95 latency for a service compared "
                    "with its learned baseline, plus the most frequent error messages.",
                    ServiceHealthArgs,
                    self.service_health,
                ),
                Tool(
                    "call_graph",
                    ToolKind.INVESTIGATE,
                    "Inter-service and egress edges with volume, errors, latency and bytes; "
                    "edges never seen during training are flagged as novel.",
                    CallGraphArgs,
                    self.call_graph,
                ),
                Tool(
                    "get_trace",
                    ToolKind.INVESTIGATE,
                    "All spans of one distributed trace, to follow a request across services.",
                    TraceArgs,
                    self.get_trace,
                ),
                Tool(
                    "classify_incident",
                    ToolKind.CLASSIFY,
                    "Record the incident classification. Required before any containment action; "
                    "may be called again to revise it. Returns the actions policy permits.",
                    ClassifyArgs,
                    self.classify_incident,
                ),
                Tool(
                    "block_ips",
                    ToolKind.RESPOND,
                    "Block client addresses or subnets at the edge firewall for a limited time.",
                    BlockArgs,
                    self.block_ips,
                ),
                Tool(
                    "isolate_service",
                    ToolKind.RESPOND,
                    "Quarantine a compromised workload with a deny-all network policy.",
                    IsolateArgs,
                    self.isolate_service,
                ),
                Tool(
                    "notify_admin",
                    ToolKind.RESPOND,
                    "Send the incident report to the on-call administrator. Every investigation "
                    "must end with exactly one report.",
                    NotifyArgs,
                    self.notify_admin,
                ),
            )
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self.tools.values()]

    def execute(self, call: ToolCall, investigation: Investigation) -> ToolOutcome:
        tool = self.tools.get(call.name)
        ok, payload = False, ""
        if tool is None:
            payload = f"unknown tool {call.name!r}"
        else:
            try:
                payload = json.dumps(
                    tool.invoke(call.arguments, investigation),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )
                ok = True
            except ValidationError as exc:
                payload = f"invalid arguments: {exc.errors(include_url=False, include_input=False)}"
            except ToolRejectedError as exc:
                payload = f"rejected by policy: {exc}"
                investigation.denied.append(f"{call.name}: {exc}")
        investigation.audit.append(
            AuditEntry(self.clock.now(), call.name, call.arguments, ok, payload)
        )
        return ToolOutcome(call.id, payload, is_error=not ok)

    def _since(self, lookback_seconds: int) -> tuple[datetime, datetime]:
        now = self.clock.now()
        return now - timedelta(seconds=lookback_seconds), now

    def _events(self, lookback_seconds: int, service: str | None = None) -> Iterable[LogEvent]:
        since, until = self._since(lookback_seconds)
        events = self.store.between(since, until)
        if service is None:
            return events
        return (e for e in events if service in (e.target, e.source))

    def search_logs(self, args: SearchLogsArgs, _: Investigation) -> Json:
        matched = [
            e
            for e in self._events(args.lookback_seconds, args.service)
            if (args.min_status is None or e.status >= args.min_status)
            and (args.client_ip is None or e.client_ip == args.client_ip)
            and (args.path_contains is None or args.path_contains in e.path)
        ]
        return {
            "matched": len(matched),
            "status_breakdown": dict(Counter(e.status for e in matched).most_common()),
            "spans": [_compact(e) for e in reversed(matched[-args.limit :])],
        }

    def top_clients(self, args: TopClientsArgs, _: Investigation) -> Json:
        since, _now = self._since(args.lookback_seconds)
        events = [e for e in self._events(args.lookback_seconds) if e.target == args.service]
        history = Counter(
            e.client_ip
            for e in self.store.between(since - HISTORY, since)
            if e.target == args.service
        )
        stats: defaultdict[str, Counter[str]] = defaultdict(Counter)
        users: defaultdict[str, set[str]] = defaultdict(set)
        paths: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for e in events:
            s = stats[e.client_ip]
            s["requests"] += 1
            s["bytes"] += e.bytes_out
            s["errors"] += e.status >= 400
            s["auth_failures"] += e.is_auth_failure
            if e.user_id:
                users[e.client_ip].add(e.user_id)
            paths[e.client_ip][f"{e.method} {e.path}"] += 1
        total = max(len(events), 1)
        ranked = sorted(stats, key=lambda ip: stats[ip][args.rank_by], reverse=True)

        def subnets(prefix: int) -> list[Json]:
            current: Counter[str] = Counter()
            clients: defaultdict[str, set[str]] = defaultdict(set)
            for ip, s in stats.items():
                current[_subnet(ip, prefix)] += s["requests"]
                clients[_subnet(ip, prefix)].add(ip)
            before: Counter[str] = Counter()
            for ip, count in history.items():
                before[_subnet(ip, prefix)] += count
            return [
                {
                    "subnet": net,
                    "clients": len(clients[net]),
                    "requests": count,
                    "share": round(count / total, 3),
                    "requests_in_prior_30m": before[net],
                }
                for net, count in current.most_common(8)
            ]

        return {
            "service": args.service,
            "total_requests": len(events),
            "distinct_clients": len(stats),
            "clients_new_in_window": sum(ip not in history for ip in stats),
            "clients": [
                {
                    "ip": ip,
                    "requests": stats[ip]["requests"],
                    "share": round(stats[ip]["requests"] / total, 4),
                    "error_rate": round(stats[ip]["errors"] / stats[ip]["requests"], 3),
                    "auth_failures": stats[ip]["auth_failures"],
                    "bytes": stats[ip]["bytes"],
                    "distinct_users": len(users[ip]),
                    "seen_before": ip in history,
                    "top_request": paths[ip].most_common(1)[0][0],
                }
                for ip in ranked[: args.limit]
            ],
            "top_subnets_24": subnets(24),
            "top_subnets_16": subnets(16),
        }

    def service_health(self, args: ServiceHealthArgs, _: Investigation) -> Json:
        since, _now = self._since(args.lookback_seconds)
        buckets: defaultdict[int, list[LogEvent]] = defaultdict(list)
        messages: Counter[str] = Counter()
        for e in self._events(args.lookback_seconds):
            if e.target != args.service:
                continue
            buckets[int((e.ts - since).total_seconds() // args.bucket_seconds)].append(e)
            if e.status >= 400 and e.message:
                messages[e.message] += 1
        typical = self.profile.typical(args.service)
        window = 10
        return {
            "service": args.service,
            "baseline": {
                "requests_per_s": round(typical["requests"] / window, 2),
                "server_error_rate": round(typical["server_error_rate"], 4),
                "auth_failure_rate": round(typical["auth_failure_rate"], 4),
                "p95_ms": round(typical["latency_p95_ms"], 1),
                "bytes_per_response": round(typical["bytes_per_response"]),
            },
            "series": [
                {
                    "t": (since + timedelta(seconds=i * args.bucket_seconds)).strftime("%H:%M:%S"),
                    "requests_per_s": round(len(es) / args.bucket_seconds, 2),
                    "server_error_rate": round(sum(e.is_server_error for e in es) / len(es), 3),
                    "auth_failure_rate": round(sum(e.is_auth_failure for e in es) / len(es), 3),
                    "p95_ms": _p95([e.latency_ms for e in es]),
                    "mb_out": round(sum(e.bytes_out for e in es) / 1e6, 2),
                }
                for i, es in sorted(buckets.items())
            ],
            "top_error_messages": [{"message": m, "count": c} for m, c in messages.most_common(5)],
        }

    def call_graph(self, args: CallGraphArgs, _: Investigation) -> Json:
        edges: defaultdict[tuple[str, str], list[LogEvent]] = defaultdict(list)
        for e in self._events(args.lookback_seconds, args.only_service):
            target = f"{EGRESS}:{e.dest_ip}" if e.target == EGRESS else e.target
            edges[(e.source, target)].append(e)
        rows: list[Json] = []
        for (source, target), es in edges.items():
            novel = sorted(
                {f"{e.method} {e.path}" for e in es if route_key(e) not in self.inventory.routes}
            )
            rows.append(
                {
                    "edge": f"{source} -> {target}",
                    "calls": len(es),
                    "error_rate": round(sum(e.status >= 400 for e in es) / len(es), 3),
                    "p95_ms": _p95([e.latency_ms for e in es]),
                    "mb": round(sum(e.bytes_out for e in es) / 1e6, 2),
                    "novel_routes": novel,
                }
            )
        rows.sort(key=lambda r: (not r["novel_routes"], -r["mb"]))
        return {"edges": rows[:30], "novel_edges": sum(bool(r["novel_routes"]) for r in rows)}

    def get_trace(self, args: TraceArgs, _: Investigation) -> Json:
        spans = [e for e in self._events(1800) if e.trace_id == args.trace_id]
        if not spans:
            raise ToolRejectedError(f"trace {args.trace_id} not found in the retention window")
        return {"trace_id": args.trace_id, "spans": [_compact(e) for e in spans]}

    def classify_incident(self, args: ClassifyArgs, investigation: Investigation) -> Json:
        investigation.classify(
            ClassificationRecord(
                incident_type=args.incident_type,
                confidence=args.confidence,
                affected_services=tuple(args.affected_services),
                summary=args.summary,
                evidence=tuple(args.evidence),
            )
        )
        permitted = sorted(a.value for a in CONTAINMENT_MATRIX[args.incident_type])
        return {
            "recorded": True,
            "incident_id": investigation.incident_id,
            "containment_permitted": permitted
            if args.confidence >= self.policy.config.min_confidence
            else [],
            "always_permitted": ["notify_admin"],
        }

    def block_ips(self, args: BlockArgs, investigation: Investigation) -> Json:
        try:
            networks = sorted({IPv4Network(t, strict=False) for t in args.targets})
        except ValueError as exc:
            raise ToolRejectedError(str(exc)) from exc
        current = investigation.classification
        decision = self.policy.check_block(networks, current.classification if current else None)
        if not decision.allowed:
            raise ToolRejectedError(decision.reason)
        rules = self.controls.firewall.block(
            networks,
            reason=args.reason,
            ttl=timedelta(minutes=args.ttl_minutes),
            incident_id=investigation.incident_id,
        )
        investigation.blocked.extend(str(r.network) for r in rules)
        return {
            "blocked": len(rules),
            "expires_at": rules[0].expires_at.isoformat(),
            "sample_commands": [r.command for r in rules[:3]],
        }

    def isolate_service(self, args: IsolateArgs, investigation: Investigation) -> Json:
        current = investigation.classification
        decision = self.policy.check_isolation(
            args.service, current.classification if current else None
        )
        if not decision.allowed:
            raise ToolRejectedError(decision.reason)
        quarantine = self.controls.mesh.isolate(
            args.service, reason=args.reason, incident_id=investigation.incident_id
        )
        investigation.isolated.append(args.service)
        return {"isolated": args.service, "applied_manifest": quarantine.manifest}

    def notify_admin(self, args: NotifyArgs, investigation: Investigation) -> Json:
        report = self.controls.notifier.send(
            incident_id=investigation.incident_id,
            severity=args.severity,
            title=args.title,
            summary=args.summary,
            recommendations=tuple(args.recommendations),
        )
        investigation.notifications.append(args.severity)
        return {"delivered": True, "at": report.created_at.isoformat()}
