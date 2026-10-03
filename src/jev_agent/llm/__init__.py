"""Chat models from the NVIDIA API Catalog, selected by tier."""

from __future__ import annotations

import httpx

from jev_agent.config import ModelTier, Settings
from jev_agent.llm.resilient import CallLog, CallRecord, LLMError, ResilientChatModel

__all__ = [
    "CallLog",
    "CallRecord",
    "LLMConfigError",
    "LLMError",
    "ResilientChatModel",
    "chat_model",
    "list_models",
]


class LLMConfigError(RuntimeError):
    pass


def _key(settings: Settings) -> str:
    if settings.nvidia_api_key is None:
        raise LLMConfigError("NVIDIA_API_KEY is not set (get one at build.nvidia.com)")
    return settings.nvidia_api_key.get_secret_value()


def chat_model(
    tier: ModelTier,
    settings: Settings,
    *,
    temperature: float = 0.0,
    call_log: CallLog | None = None,
) -> ResilientChatModel:
    return ResilientChatModel(
        models=settings.models_for(tier),
        api_key=_key(settings),
        base_url=settings.nvidia_base_url,
        reasoning=settings.reasoning_for(tier),
        temperature=temperature,
        max_tokens=settings.llm_max_tokens,
        first_token_timeout_s=settings.llm_first_token_timeout_s,
        total_timeout_s=settings.llm_total_timeout_s,
        call_log=call_log or CallLog(),
    )


def list_models(settings: Settings) -> list[str]:
    """Model ids available to this key — the free catalog changes often."""
    response = httpx.get(
        f"{settings.nvidia_base_url}/models",
        headers={"Authorization": f"Bearer {_key(settings)}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return sorted(m["id"] for m in response.json()["data"])
