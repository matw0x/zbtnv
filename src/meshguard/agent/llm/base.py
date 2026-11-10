from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

type JsonSchema = dict[str, Any]
type Effort = Literal["low", "medium", "high", "xhigh", "max"]


class LLMError(RuntimeError):
    pass


class LLMRefusalError(LLMError):
    pass


class StopReason(StrEnum):
    TOOL_USE = "tool_use"
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: JsonSchema


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True, slots=True)
class AssistantTurn:
    text: str
    tool_calls: tuple[ToolCall, ...]
    stop_reason: StopReason
    usage: Usage = Usage()


class ChatSession(Protocol):
    """One conversation; the implementation owns the provider-native message history."""

    async def send(
        self, *, text: str | None = None, results: Sequence[ToolOutcome] = ()
    ) -> AssistantTurn: ...


class ChatModel(Protocol):
    @property
    def name(self) -> str: ...

    def session(self, system: str, tools: Sequence[ToolSpec]) -> ChatSession: ...


_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "uniqueItems",
    }
)


def strict_schema(schema: JsonSchema) -> JsonSchema:
    """Adapt a pydantic JSON schema to the subset accepted by strict tool calling.

    Dropped constraints are still enforced client-side because every call is
    re-validated by the pydantic model that produced the schema.
    """

    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in {"properties", "$defs"}:
                cleaned[key] = {name: visit(sub) for name, sub in value.items()}
            elif key not in _UNSUPPORTED_KEYWORDS:
                cleaned[key] = visit(value)
        if cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
        return cleaned

    result: JsonSchema = visit(schema)
    return result
