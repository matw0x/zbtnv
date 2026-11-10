from collections.abc import Sequence
from typing import Any

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaToolParam

from meshguard.agent.llm.base import (
    AssistantTurn,
    Effort,
    LLMError,
    LLMRefusalError,
    StopReason,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    Usage,
    strict_schema,
)

_STOP_REASONS = {"tool_use": StopReason.TOOL_USE, "max_tokens": StopReason.MAX_TOKENS}


class ClaudeChatModel:
    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        effort: Effort = "medium",
        max_tokens: int = 16_000,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.client = client or anthropic.AsyncAnthropic()

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def session(self, system: str, tools: Sequence[ToolSpec]) -> ClaudeSession:
        return ClaudeSession(self, system, tools)


class ClaudeSession:
    def __init__(self, model: ClaudeChatModel, system: str, tools: Sequence[ToolSpec]) -> None:
        self._model = model
        self._system = system
        self._tools: list[BetaToolParam] = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": strict_schema(t.parameters),
                "strict": True,
            }
            for t in tools
        ]
        self._messages: list[BetaMessageParam] = []

    async def send(
        self, *, text: str | None = None, results: Sequence[ToolOutcome] = ()
    ) -> AssistantTurn:
        content: list[Any] = [
            {
                "type": "tool_result",
                "tool_use_id": r.call_id,
                "content": r.content,
                "is_error": r.is_error,
            }
            for r in results
        ]
        if text:
            content.append({"type": "text", "text": text})
        self._messages.append({"role": "user", "content": content})

        try:
            response = await self._model.client.beta.messages.create(
                model=self._model.model,
                max_tokens=self._model.max_tokens,
                system=self._system,
                tools=self._tools,
                messages=self._messages,
                output_config={"effort": self._model.effort},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            raise LLMError(f"Claude API call failed: {exc}") from exc

        if response.stop_reason == "refusal":
            details = response.stop_details
            category = details.category if details is not None else None
            raise LLMRefusalError(f"model declined the request (category={category})")

        self._messages.append({"role": "assistant", "content": response.content})
        usage = response.usage
        return AssistantTurn(
            text="\n".join(b.text for b in response.content if b.type == "text").strip(),
            tool_calls=tuple(
                ToolCall(b.id, b.name, dict(b.input))
                for b in response.content
                if b.type == "tool_use"
            ),
            stop_reason=_STOP_REASONS.get(response.stop_reason or "", StopReason.END_TURN),
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cache_read_input_tokens or 0,
            ),
        )
