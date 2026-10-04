from __future__ import annotations

import os

import pytest
from pydantic import SecretStr

from jev_agent.config import Settings
from jev_agent.observability import configure_tracing, run_url


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_tracing_off_without_key() -> None:
    assert configure_tracing(Settings(_env_file=None)) is False
    assert "LANGSMITH_TRACING" not in os.environ


def test_tracing_off_when_disabled() -> None:
    settings = Settings(_env_file=None, langsmith_api_key=SecretStr("k"), langsmith_tracing=False)
    assert configure_tracing(settings) is False


def test_tracing_exports_env() -> None:
    settings = Settings(
        _env_file=None,
        langsmith_api_key=SecretStr("lsv2-key"),
        langsmith_project="demo",
        langsmith_endpoint="https://eu.api.smith.langchain.com",
    )
    assert configure_tracing(settings) is True
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2-key"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == "demo"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://eu.api.smith.langchain.com"


def test_run_url_never_raises() -> None:
    assert run_url(object(), "demo") is None
