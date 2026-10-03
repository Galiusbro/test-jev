from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import SecretStr
from typer.testing import CliRunner

from jev_agent import cli
from jev_agent.bench import run_bench
from jev_agent.config import Settings

BASE = "https://nvidia.test/v1"


def _settings() -> Settings:
    return Settings(_env_file=None, nvidia_api_key=SecretStr("nvapi-x"), nvidia_base_url=BASE)


def _tool_reply(name: str = "read_file", args: str = '{"path": "a.py"}') -> dict[str, object]:
    call = {"type": "function", "function": {"name": name, "arguments": args}}
    return {"choices": [{"message": {"role": "assistant", "tool_calls": [call]}}]}


@respx.mock
def test_counts_valid_tool_calls_and_errors() -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        if model == "good":
            return httpx.Response(200, json=_tool_reply())
        if model == "chatty":
            return httpx.Response(200, json={"choices": [{"message": {"content": "Sure!"}}]})
        return httpx.Response(429, text="busy")

    route = respx.post(f"{BASE}/chat/completions").mock(side_effect=reply)
    good, chatty, busy = run_bench(_settings(), ["good", "chatty", "busy"], trials=2)

    assert route.calls[0].request.headers["Authorization"] == "Bearer nvapi-x"
    assert (good.tool_calls_ok, good.trials, good.median_s is not None) == (2, 2, True)
    assert (chatty.tool_calls_ok, chatty.trials) == (0, 2)
    assert busy.errors == ["HTTP 429", "HTTP 429"] and busy.median_s is None


@respx.mock
def test_bad_tool_arguments_not_counted() -> None:
    respx.post(f"{BASE}/chat/completions").respond(json=_tool_reply(args="not json"))
    (result,) = run_bench(_settings(), ["m"], trials=1)
    assert result.tool_calls_ok == 0


@respx.mock
def test_network_error_recorded() -> None:
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    (result,) = run_bench(_settings(), ["m"], trials=1)
    assert result.errors == ["ReadTimeout"]


def test_requires_key() -> None:
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        run_bench(Settings(_env_file=None), ["m"], trials=1)


@respx.mock
def test_cli_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", _settings)
    respx.post(f"{BASE}/chat/completions").respond(json=_tool_reply())
    result = CliRunner().invoke(cli.app, ["bench", "vendor/model-a", "-n", "1"])
    assert result.exit_code == 0, result.output
    assert "vendor/model-a" in result.output and "1/1" in result.output


def test_cli_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    result = CliRunner().invoke(cli.app, ["bench", "m"])
    assert result.exit_code == 1
