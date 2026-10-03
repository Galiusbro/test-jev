"""Latency and tool-calling check for NVIDIA catalog models.

Free-tier latency depends mostly on how busy a model is, not on its size, so
model choice is measured, not guessed. Each trial sends one small tool-calling
request and records wall time and whether the model produced a valid tool call.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx

from jev_agent.config import Settings

# Shortlist per tier: models that answered with valid tool calls in the last
# measured run (2026-10-03). Many catalog entries return 404 (retired) and the
# most requested models time out on the free tier, so re-measure before demos.
CANDIDATES: dict[str, list[str]] = {
    "fast": [
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "openai/gpt-oss-20b",
    ],
    "strong": [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "openai/gpt-oss-20b",
        "moonshotai/kimi-k3",
    ],
    "coder": [
        "poolside/laguna-xs-2.1",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ],
}

_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
_PROMPT = "Open the login route handler at src/demo_api/auth.py. Use the tool."


@dataclass
class BenchResult:
    model: str
    latencies_s: list[float] = field(default_factory=list)
    tool_calls_ok: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def trials(self) -> int:
        return len(self.latencies_s) + len(self.errors)

    @property
    def median_s(self) -> float | None:
        return statistics.median(self.latencies_s) if self.latencies_s else None


def _valid_tool_call(body: dict[str, object]) -> bool:
    try:
        call = body["choices"][0]["message"]["tool_calls"][0]["function"]  # type: ignore[index]
        return call["name"] == "read_file" and "path" in json.loads(call["arguments"])
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def bench_model(http: httpx.Client, model: str, trials: int) -> BenchResult:
    result = BenchResult(model)
    for _ in range(trials):
        start = time.perf_counter()
        try:
            response = http.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": _PROMPT}],
                    "tools": [_TOOL],
                    "max_tokens": 256,
                    "temperature": 0,
                },
            )
        except httpx.HTTPError as exc:
            result.errors.append(type(exc).__name__)
            continue
        elapsed = time.perf_counter() - start
        if response.status_code >= 400:
            result.errors.append(f"HTTP {response.status_code}")
            continue
        result.latencies_s.append(elapsed)
        result.tool_calls_ok += _valid_tool_call(response.json())
    return result


def run_bench(
    settings: Settings, models: Sequence[str], trials: int, timeout_s: float = 90.0
) -> list[BenchResult]:
    if settings.nvidia_api_key is None:
        raise ValueError("NVIDIA_API_KEY is not set (get one at build.nvidia.com)")
    with httpx.Client(
        base_url=settings.nvidia_base_url,
        timeout=timeout_s,
        headers={"Authorization": f"Bearer {settings.nvidia_api_key.get_secret_value()}"},
    ) as http:
        return [bench_model(http, model, trials) for model in models]
