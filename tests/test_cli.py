from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from jev_agent import cli
from jev_agent.config import ModelTier, Settings
from jev_agent.decisions import FakeJevClient, HttpJevClient, JevError, Question
from jev_agent.decisions.client import State
from jev_agent.llm import CallLog, CallRecord, LLMConfigError, LLMError

runner = CliRunner()


class _FakeChat:
    """Answers on the last model of the chain after the others failed."""

    def __init__(self, log: CallLog, chain: list[str]) -> None:
        self._log, self._chain = log, chain

    def invoke(self, _prompt: str) -> AIMessage:
        for model in self._chain[:-1]:
            self._log.add(CallRecord(model=model, ok=False, elapsed_s=15.0, error="timeout"))
        self._log.add(CallRecord(model=self._chain[-1], ok=True, elapsed_s=0.4))
        return AIMessage(content="pong")


def _fake_chat_model(chain: list[str]) -> Any:
    def factory(_tier: ModelTier, _settings: Settings, *, call_log: CallLog) -> _FakeChat:
        return _FakeChat(call_log, chain)

    return factory


def _done(_state: State, _questions: Mapping[str, Question]) -> Mapping[str, Any]:
    return {
        "done": {"type": "boolean", "probability": 0.9, "confidence": 0.8},
        "q": {"type": "boolean", "probability": 0.25, "confidence": 0.6},
        "kind": {"type": "choice", "value": "bug", "probability": 0.9, "confidence": 0.85},
        "risk": {"type": "score", "value": "low", "score": 0.3, "probability": 0.7},
    }


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))


def _patch_jev(monkeypatch: pytest.MonkeyPatch, fake: FakeJevClient | Exception) -> None:
    def from_settings(_settings: Settings) -> FakeJevClient:
        if isinstance(fake, Exception):
            raise fake
        return fake

    monkeypatch.setattr(HttpJevClient, "from_settings", staticmethod(from_settings))


def test_doctor_all_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "chat_model", _fake_chat_model(["vendor/primary"]))
    _patch_jev(monkeypatch, FakeJevClient(_done))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert result.output.count("vendor/primary") == 3  # fast, strong, coder
    assert "'pong'" in result.output
    assert "fallback" not in result.output
    assert "done p=0.90" in result.output
    assert "kind=bug" in result.output and "risk=low" in result.output


def test_doctor_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(_t: ModelTier, _s: Settings, *, call_log: CallLog) -> _FakeChat:
        raise LLMConfigError("NVIDIA_API_KEY is not set")

    monkeypatch.setattr(cli, "chat_model", broken)
    _patch_jev(monkeypatch, JevError("TYPESAFE_API_KEY is not set"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "NVIDIA_API_KEY" in result.output
    assert "TYPESAFE_API_KEY" in result.output


def test_doctor_shows_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "chat_model", _fake_chat_model(["slow/model", "backup/model"]))
    _patch_jev(monkeypatch, FakeJevClient(_done))
    result = runner.invoke(cli.app, ["doctor"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    assert "backup/model" in result.output
    assert "after fallback from slow/model (timeout)" in result.output


def test_doctor_all_models_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(_t: ModelTier, _s: Settings, *, call_log: CallLog) -> _FakeChat:
        raise LLMError("all models failed — a: HTTP 503")

    monkeypatch.setattr(cli, "chat_model", failing)
    _patch_jev(monkeypatch, FakeJevClient(_done))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "all models failed" in result.output


def test_models_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "list_models", lambda _s: ["meta/llama", "qwen/coder"])
    result = runner.invoke(cli.app, ["models", "QWEN"])
    assert result.exit_code == 0
    assert "qwen/coder" in result.output
    assert "meta/llama" not in result.output


def test_models_without_key() -> None:
    result = runner.invoke(cli.app, ["models"])
    assert result.exit_code == 1
    assert "NVIDIA_API_KEY" in result.output


def test_ask_parses_json_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeJevClient(_done)
    _patch_jev(monkeypatch, fake)
    result = runner.invoke(cli.app, ["ask", "Is it risky?", "-s", '{"cmd": "rm -rf /"}'])
    assert result.exit_code == 0, result.output
    assert "0.250" in result.output
    assert fake.calls[0][0] == {"cmd": "rm -rf /"}


def test_ask_reports_jev_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_jev(monkeypatch, JevError("Jev returned HTTP 401"))
    result = runner.invoke(cli.app, ["ask", "Done?", "-s", "x"])
    assert result.exit_code == 1
    assert "401" in result.output


def test_ask_plain_text_state(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeJevClient(_done)
    _patch_jev(monkeypatch, fake)
    runner.invoke(cli.app, ["ask", "Done?", "-s", "tests pass"])
    assert fake.calls[0][0] == "tests pass"
