from __future__ import annotations

import pytest

from jev_agent.config import ModelTier, Settings


def test_default_chains_have_fallbacks() -> None:
    settings = Settings(_env_file=None)
    for tier in ModelTier:
        assert len(settings.models_for(tier)) >= 2


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_CODER", "qwen/some-coder,openai/gpt-oss-20b")
    monkeypatch.setenv("REASONING_STRONG", "on")
    monkeypatch.setenv("JEV_MIN_CONFIDENCE", "0.8")
    settings = Settings(_env_file=None)
    assert settings.models_for(ModelTier.CODER) == ["qwen/some-coder", "openai/gpt-oss-20b"]
    assert settings.reasoning_for(ModelTier.STRONG) == "on"
    assert settings.jev_min_confidence == 0.8


def test_invalid_reasoning_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REASONING_FAST", "maybe")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
