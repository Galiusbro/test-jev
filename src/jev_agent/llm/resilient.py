"""LangChain chat model over the NVIDIA API Catalog with a fallback chain.

Free-tier models stall at random — sometimes before the HTTP headers, sometimes
after `200 OK` with no tokens. Every call therefore streams, and a model that
sends no token within `first_token_timeout_s` (or errors, or answers empty) is
abandoned for the next model in the chain. Each attempt is recorded in a
`CallLog` for metrics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
from langchain_core.messages.tool import invalid_tool_call, tool_call
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field, SecretStr

from jev_agent.config import Reasoning
from jev_agent.llm.reasoning import reasoning_params


class LLMError(RuntimeError):
    """A call that no fallback can fix (bad key, or every model failed)."""


class _ModelUnavailable(Exception):
    """This model failed; the next one in the chain may work."""


@dataclass
class CallRecord:
    model: str
    ok: bool
    elapsed_s: float
    first_token_s: float | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class CallLog:
    records: list[CallRecord] = field(default_factory=list)

    def add(self, record: CallRecord) -> None:
        self.records.append(record)


@dataclass
class _Stream:
    """Accumulates OpenAI-style SSE chunks into one message."""

    first_token_s: float | None = None
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, str]] = field(default_factory=dict)
    finish_reason: str | None = None
    model_name: str | None = None
    usage: dict[str, int] | None = None

    def add(self, chunk: dict[str, Any], at_s: float) -> None:
        self.model_name = chunk.get("model") or self.model_name
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            thought = delta.get("reasoning_content") or delta.get("reasoning")
            calls = delta.get("tool_calls") or []
            if (text or thought or calls) and self.first_token_s is None:
                self.first_token_s = at_s
            if text:
                self.content.append(text)
            if thought:
                self.reasoning.append(thought)
            for call in calls:
                slot = self.tool_calls.setdefault(
                    call.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                fn = call.get("function") or {}
                slot["id"] = call.get("id") or slot["id"]
                slot["name"] = slot["name"] or fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
            self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def to_message(self) -> AIMessage:
        calls, invalid = [], []
        for index, slot in sorted(self.tool_calls.items()):
            call_id = slot["id"] or f"call_{index}"
            try:
                args = json.loads(slot["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments are not an object")
                calls.append(tool_call(name=slot["name"], args=args, id=call_id))
            except ValueError as exc:
                invalid.append(
                    invalid_tool_call(
                        name=slot["name"], args=slot["arguments"], id=call_id, error=str(exc)
                    )
                )
        content = "".join(self.content)
        if calls and not content.strip():
            content = ""
        usage = None
        if self.usage:
            usage = {
                "input_tokens": self.usage.get("prompt_tokens", 0),
                "output_tokens": self.usage.get("completion_tokens", 0),
                "total_tokens": self.usage.get("total_tokens", 0),
            }
        return AIMessage(
            content=content,
            tool_calls=calls,
            invalid_tool_calls=invalid,
            additional_kwargs={"reasoning_content": "".join(self.reasoning)}
            if self.reasoning
            else {},
            response_metadata={"model_name": self.model_name, "finish_reason": self.finish_reason},
            usage_metadata=usage,
        )

    @property
    def empty(self) -> bool:
        return not "".join(self.content).strip() and not self.tool_calls


class ResilientChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[str] = Field(min_length=1)
    api_key: SecretStr
    base_url: str = "https://integrate.api.nvidia.com/v1"
    reasoning: Reasoning = "default"
    temperature: float = 0.0
    max_tokens: int = 4096
    first_token_timeout_s: float = 15.0
    total_timeout_s: float = 180.0
    call_log: CallLog = Field(default_factory=CallLog, exclude=True)
    clock: Callable[[], float] = Field(default=time.perf_counter, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "resilient-nvidia"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"models": self.models, "reasoning": self.reasoning}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice in ("any", "required"):
            kwargs["tool_choice"] = "required"
        elif tool_choice and tool_choice not in ("auto", "none"):
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        elif tool_choice:
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "messages": convert_to_openai_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if stop:
            payload["stop"] = stop
        for key in ("tools", "tool_choice"):
            if kwargs.get(key):
                payload[key] = kwargs[key]

        timeout = httpx.Timeout(10.0, read=self.first_token_timeout_s)
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
        failures: list[str] = []
        with httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers) as http:
            for model in self.models:
                start = self.clock()
                stream = _Stream()
                try:
                    self._stream_one(http, model, payload, stream, start)
                except _ModelUnavailable as exc:
                    failures.append(f"{model}: {exc}")
                    self._record(model, start, stream, error=str(exc))
                    continue
                self._record(model, start, stream)
                return ChatResult(generations=[ChatGeneration(message=stream.to_message())])
        raise LLMError("all models failed — " + "; ".join(failures))

    def _stream_one(
        self,
        http: httpx.Client,
        model: str,
        payload: dict[str, Any],
        stream: _Stream,
        start: float,
    ) -> None:
        body = {**payload, "model": model, **reasoning_params(model, self.reasoning)}
        try:
            with http.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code in (401, 403):
                    response.read()
                    raise LLMError(f"NVIDIA rejected the key: HTTP {response.status_code}")
                if response.status_code >= 400:
                    response.read()
                    raise _ModelUnavailable(
                        f"HTTP {response.status_code} {response.text[:200]}".strip()
                    )
                for line in response.iter_lines():
                    elapsed = self.clock() - start
                    if stream.first_token_s is None and elapsed > self.first_token_timeout_s:
                        raise _ModelUnavailable(f"no tokens in {self.first_token_timeout_s:.0f}s")
                    if elapsed > self.total_timeout_s:
                        raise _ModelUnavailable(f"total timeout {self.total_timeout_s:.0f}s")
                    if not line.startswith("data:"):
                        continue  # SSE comments / keep-alives
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    stream.add(json.loads(data), elapsed)
        except httpx.TimeoutException as exc:
            raise _ModelUnavailable(f"timeout ({type(exc).__name__})") from exc
        except httpx.HTTPError as exc:
            raise _ModelUnavailable(type(exc).__name__) from exc
        except json.JSONDecodeError as exc:
            raise _ModelUnavailable("malformed stream") from exc
        if stream.empty:
            out = (stream.usage or {}).get("completion_tokens")
            raise _ModelUnavailable(
                f"empty response (finish={stream.finish_reason}, output_tokens={out}, "
                f"reasoning_chars={len(''.join(stream.reasoning))})"
            )

    def _record(self, model: str, start: float, stream: _Stream, error: str | None = None) -> None:
        usage = stream.usage or {}
        self.call_log.add(
            CallRecord(
                model=model,
                ok=error is None,
                elapsed_s=self.clock() - start,
                first_token_s=stream.first_token_s,
                error=error,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            )
        )
