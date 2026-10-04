"""LangSmith tracing.

LangChain/LangGraph trace automatically once the LANGSMITH_* environment
variables are set. Our keys live in `.env` (read by pydantic-settings, not
exported), so the CLI calls `configure_tracing` before a run. Every LLM call,
Jev decision (TypeSafeClassifier is a Runnable), tool call and graph node then
lands in one tree under a named root run.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

from jev_agent.config import Settings


def configure_tracing(settings: Settings) -> bool:
    """Export LangSmith settings to the environment. Returns True if tracing is on."""
    if settings.langsmith_api_key is None or not settings.langsmith_tracing:
        return False
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    return True


def run_url(run: Any, project: str) -> str | None:
    """Link to a root run in the LangSmith UI; None if it can't be resolved."""
    try:
        from langsmith import Client

        with warnings.catch_warnings():
            # Deprecated (removal after 2027-01-31) in favour of client.runs.get_url,
            # which also needs project and trace ids; migrate when bumping langsmith.
            warnings.simplefilter("ignore", DeprecationWarning)
            return str(Client().get_run_url(run=run, project_name=project))
    except Exception:  # noqa: BLE001 — a missing link must never fail a run
        return None


def flush() -> None:
    """Wait for pending trace uploads before the process exits."""
    from langchain_core.tracers.langchain import wait_for_all_tracers

    wait_for_all_tracers()
