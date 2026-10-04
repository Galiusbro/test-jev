"""LangChain chat model over the NVIDIA API Catalog with a fallback chain.

Free-tier models stall at random — sometimes before the HTTP headers, sometimes
after `200 OK` with no tokens. Every call therefore streams, and a model that
sends no token within `first_token_timeout_s` (or errors, or answers empty) is
abandoned for the next model in the chain. Each attempt is recorded in a
`CallLog` for metrics.
"""

from __future__ import annotations

import json
import re
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
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

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

    def normalize_tool_calls(self, tool_names: set[str]) -> None:
        """Repair tool calls that open models get slightly wrong.

        - Names in the wrong case (gpt-oss sends `plan` for `Plan`).
        - Calls written as text instead of via the tools API, e.g.
          `[{"name": "Plan", "parameters": {...}}]` (Nemotron). Only replies
          that are entirely such JSON, naming a provided tool, are converted.
        """
        if not tool_names:
            return
        by_lower = {name.lower(): name for name in tool_names}
        for slot in self.tool_calls.values():
            # gpt-oss can leak its channel tokens: "replace_in_file<|channel|>commentary"
            name = slot["name"].split("<|")[0].strip()
            slot["name"] = name if name in tool_names else by_lower.get(name.lower(), name)
        if self.tool_calls:
            return
        calls = _parse_text_tool_calls("".join(self.content), tool_names)
        if calls:
            self.content = []
            for index, (name, args) in enumerate(calls):
                self.tool_calls[index] = {
                    "id": f"text_call_{index}",
                    "name": name,
                    "arguments": json.dumps(args),
                }


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_TAG = re.compile(r"^<tool_call>\s*(.*?)\s*</tool_call>$", re.DOTALL)


def _parse_text_tool_calls(text: str, tool_names: set[str]) -> list[tuple[str, dict[str, Any]]]:
    body = text.strip()
    for pattern in (_FENCE, _TAG):
        if match := pattern.match(body):
            body = match.group(1).strip()
    try:
        data = json.loads(body)
    except ValueError:
        try:
            data = json.loads(_close_brackets(body))
        except ValueError:
            return []
    by_lower = {name.lower(): name for name in tool_names}
    while isinstance(data, list) and len(data) == 1 and isinstance(data[0], list):
        data = data[0]  # [[{...}]] seen from Nemotron
    items = data if isinstance(data, list) else [data]
    calls: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        name = by_lower.get(str(item.get("name", "")).lower()) if isinstance(item, dict) else None
        if name is None:
            return []
        args = item.get("parameters", item.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return []
        if not isinstance(args, dict):
            return []
        calls.append((name, args))
    return calls


def _close_brackets(text: str) -> str:
    """Append the closers a truncated or unbalanced JSON value is missing.

    Nemotron sometimes opens with `[[` and closes with a single `]`.
    """
    closers = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in closers:
            stack.append(closers[char])
        elif char in "}]" and stack and stack[-1] == char:
            stack.pop()
    return text + "".join(reversed(stack))


class ResilientChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    models: list[str] = Field(min_length=1)
    api_key: SecretStr
    base_url: str = "https://integrate.api.nvidia.com/v1"
    reasoning: Reasoning = "default"
    temperature: float = 0.0
    max_tokens: int = 4096
    # Base wait for the first token; long prompts get +1s per ~2k tokens because
    # prefill alone takes longer.
    first_token_timeout_s: float = 15.0
    total_timeout_s: float = 180.0
    # Free-tier failures are usually brief: after the whole chain fails, pause
    # and walk it again.
    chain_passes: int = Field(default=2, ge=1)
    retry_pause_s: float = 2.0
    # A model that just failed goes to the back of the chain for this many calls,
    # so a stuck model isn't retried first on every step of a long loop.
    cooldown_calls: int = 3
    call_log: CallLog = Field(default_factory=CallLog, exclude=True)
    clock: Callable[[], float] = Field(default=time.perf_counter, exclude=True)
    sleep: Callable[[float], None] = Field(default=time.sleep, exclude=True)

    _cooldown: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "resilient-nvidia"

    def _chain(self) -> list[str]:
        ready = [m for m in self.models if self._cooldown.get(m, 0) <= 0]
        cooling = [m for m in self.models if self._cooldown.get(m, 0) > 0]
        for m in cooling:
            self._cooldown[m] -= 1
        return ready + cooling

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
        if tool_choice in ("any", "required") and len(formatted) == 1:
            # Structured output: name the function. Open models answer a generic
            # "required" by calling tools they remember from earlier steps.
            name = formatted[0]["function"]["name"]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": name}}
        elif tool_choice in ("any", "required"):
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

        first_token = self.first_token_timeout_s + len(json.dumps(payload["messages"])) / 8000
        timeout = httpx.Timeout(10.0, read=first_token)
        headers = {"Authorization": f"Bearer {self.api_key.get_secret_value()}"}
        tool_names = {t["function"]["name"] for t in payload.get("tools", [])}
        failures: list[str] = []
        with httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers) as http:
            for attempt in range(self.chain_passes):
                if attempt:
                    self.sleep(self.retry_pause_s)
                for model in self._chain() if attempt == 0 else self.models:
                    start = self.clock()
                    stream = _Stream()
                    try:
                        self._stream_one(http, model, payload, stream, start, first_token)
                    except _ModelUnavailable as exc:
                        failures.append(f"{model}: {exc}")
                        self._record(model, start, stream, error=str(exc))
                        self._cooldown[model] = self.cooldown_calls
                        continue
                    self._record(model, start, stream)
                    stream.normalize_tool_calls(tool_names)
                    message = stream.to_message()
                    return ChatResult(generations=[ChatGeneration(message=message)])
        raise LLMError("all models failed — " + "; ".join(failures))

    def _stream_one(
        self,
        http: httpx.Client,
        model: str,
        payload: dict[str, Any],
        stream: _Stream,
        start: float,
        first_token_timeout_s: float,
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
                    if stream.first_token_s is None and elapsed > first_token_timeout_s:
                        raise _ModelUnavailable(f"no tokens in {first_token_timeout_s:.0f}s")
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
