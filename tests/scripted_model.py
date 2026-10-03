"""A chat model that replays scripted replies — for offline agent/graph tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field


def call(name: str, call_id: str = "", **args: Any) -> AIMessage:
    """An AIMessage that calls one tool."""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id or name}])


def say(text: str) -> AIMessage:
    return AIMessage(text)


class ScriptedChatModel(BaseChatModel):
    """Replays `replies` in order; an exception in the list is raised instead."""

    replies: list[Any]  # AIMessage | Exception; Any keeps pydantic from coercing
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self, tools: Sequence[Any], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools.append([getattr(t, "name", getattr(t, "__name__", "?")) for t in tools])
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(list(messages))
        if not self.replies:
            raise AssertionError("script exhausted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])
