"""Runtime configuration, loaded from environment variables and `.env`."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# "default" sends nothing and lets the model decide; see `llm.reasoning`.
Reasoning = Literal["default", "on", "off", "low"]


class ModelTier(StrEnum):
    """Model tiers the router can pick from. Each maps to a chain of model ids."""

    FAST = "fast"  # classification, summaries, PR text
    STRONG = "strong"  # planning, review
    CODER = "coder"  # implementation, repair


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # NVIDIA API Catalog (build.nvidia.com) — free tier, OpenAI-compatible.
    nvidia_api_key: SecretStr | None = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    # Comma-separated fallback chains, first = preferred. Free-tier queues stall
    # at random, so every tier needs alternatives. Measure with `jev-agent bench`.
    model_fast: str = "nvidia/nemotron-3.5-lightning-30b-a3b,openai/gpt-oss-20b"
    model_strong: str = "nvidia/nemotron-3-ultra-550b-a55b,openai/gpt-oss-20b,moonshotai/kimi-k3"
    model_coder: str = "poolside/laguna-xs-2.1,nvidia/nemotron-3-ultra-550b-a55b,openai/gpt-oss-20b"
    reasoning_fast: Reasoning = "off"
    reasoning_strong: Reasoning = "default"
    reasoning_coder: Reasoning = "default"
    # No token within this window = stuck in the queue; move to the next model.
    llm_first_token_timeout_s: float = Field(default=15.0, gt=0)
    llm_total_timeout_s: float = Field(default=120.0, gt=0)
    # Room for full-file writes and for reasoning models' thinking.
    llm_max_tokens: int = Field(default=8192, gt=0)
    llm_chain_passes: int = Field(default=2, ge=1)

    # Jev via the TypeSafe API (console.typesafe.ai).
    typesafe_api_key: SecretStr | None = None
    typesafe_base_url: str = "https://api.typesafe.ai"  # SDK appends /v1/systemone
    # Alias of the latest release. Pin a versioned id (e.g. "jev-1.13.0") once
    # confidence thresholds are tuned against it.
    jev_model: str = "jev-latest"
    jev_timeout_s: float = 30.0
    # Decisions below this confidence take the conservative branch.
    jev_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    # LangSmith tracing (smith.langchain.com); on whenever a key is set.
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "jev-agent"
    langsmith_tracing: bool = True
    langsmith_endpoint: str | None = None  # EU / self-hosted

    # Workflow limits.
    max_repair_attempts: int = Field(default=3, ge=0)

    def models_for(self, tier: ModelTier) -> list[str]:
        chain = {
            ModelTier.FAST: self.model_fast,
            ModelTier.STRONG: self.model_strong,
            ModelTier.CODER: self.model_coder,
        }[tier]
        return [m.strip() for m in chain.split(",") if m.strip()]

    def reasoning_for(self, tier: ModelTier) -> Reasoning:
        return {
            ModelTier.FAST: self.reasoning_fast,
            ModelTier.STRONG: self.reasoning_strong,
            ModelTier.CODER: self.reasoning_coder,
        }[tier]


@lru_cache
def get_settings() -> Settings:
    return Settings()
