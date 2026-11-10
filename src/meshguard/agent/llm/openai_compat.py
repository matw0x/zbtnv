"""Adapter for OpenAI-compatible endpoints (Ollama, vLLM, LM Studio) to keep logs on-premises."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from meshguard.agent.llm.base import (
    AssistantTurn,
    LLMError,
    StopReason,
    ToolCall,
    ToolOutcome,
    ToolSpec,
    Usage,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAICompatibleChatModel:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "not-needed",
        max_tokens: int = 4_096,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise LLMError("install the optional dependency: uv sync --extra openai") from exc
        self.model = model
        self.max_tokens = max_tokens
        self.client: AsyncOpenAI = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @property
    def name(self) -> str:
        return f"openai-compatible/{self.model}"

    def session(self, system: str, tools: Sequence[ToolSpec]) -> OpenAICompatibleSession:
        return OpenAICompatibleSession(self, system, tools)


class OpenAICompatibleSession:
    def __init__(
        self, model: OpenAICompatibleChatModel, system: str, tools: Sequence[ToolSpec]
    ) -> None:
        self._model = model
        self._tools: list[Any] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        self._messages: list[Any] = [{"role": "system", "content": system}]

    async def send(
        self, *, text: str | None = None, results: Sequence[ToolOutcome] = ()
    ) -> AssistantTurn:
        self._messages.extend(
            {"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in results
        )
        if text:
            self._messages.append({"role": "user", "content": text})
        try:
            response = await self._model.client.chat.completions.create(
                model=self._model.model,
                messages=self._messages,
                tools=self._tools,
                max_tokens=self._model.max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"OpenAI-compatible call failed: {exc}") from exc

        choice = response.choices[0]
        message = choice.message
        self._messages.append(message.model_dump(exclude_none=True))
        calls = []
        for call in message.tool_calls or []:
            if call.type != "function":
                continue
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {"__invalid_json__": call.function.arguments}
            calls.append(ToolCall(call.id, call.function.name, arguments))
        usage = response.usage
        return AssistantTurn(
            text=(message.content or "").strip(),
            tool_calls=tuple(calls),
            stop_reason=StopReason.TOOL_USE
            if calls
            else StopReason.MAX_TOKENS
            if choice.finish_reason == "length"
            else StopReason.END_TURN,
            usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
        )
