"""Request parameters that switch a model's reasoning on, off or to low effort.

Model families use different knobs (measured on build.nvidia.com, 2026-10-03):

- Nemotron, Laguna and most chat-template models honour
  `chat_template_kwargs.enable_thinking`; `reasoning_effort: low` also works.
- gpt-oss ignores `enable_thinking`; it only takes `reasoning_effort` and
  cannot switch reasoning off entirely, so "off" maps to "low".
"""

from __future__ import annotations

from typing import Any

from jev_agent.config import Reasoning


def reasoning_params(model: str, mode: Reasoning) -> dict[str, Any]:
    if mode == "default":
        return {}
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "high" if mode == "on" else "low"}
    if mode == "low":
        return {"reasoning_effort": "low"}
    return {"chat_template_kwargs": {"enable_thinking": mode == "on"}}
