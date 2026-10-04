"""Agent harness: LangChain `create_agent` plus middleware.

Every tool-using agent (planner exploration, implementer, repairer) is a
standard `create_agent` graph. What used to be a hand-written loop is now
middleware at LangChain's extension points:

- `ModelCallLimitMiddleware` (built in)     step budget
- `CompactionMiddleware`  wrap_model_call   keep the latest read of each file,
                                            elide stale/old tool output
- `InvalidToolCallMiddleware` after_model   send unparseable tool calls back to
                                            the model instead of ending the run
- `StepTrackerMiddleware` after_model       steps / tool calls / finished
- `JevModelRouter`        ModelRouterMiddleware subclass: Jev picks the model
                                            for the run, with a fallback
- `JevWriteGateMiddleware` wrap_tool_call   Jev screens each edit against the
                                            rules code cannot check

Deterministic policy stays inside the tools (see `tools.py`, `policy.py`):
middleware adds judgment, it never replaces enforcement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    hook_config,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_typesafe import ChoiceAnswer
from langchain_typesafe.experimental.middleware import ModelRouterMiddleware
from langchain_typesafe.experimental.middleware.model_router import (
    ModelChoice,
    _ModelRouterConfig,
)
from langgraph.types import Command

from jev_agent.decisions.fabric import Decisions
from jev_agent.llm import LLMError
from jev_agent.policy import Policy
from jev_agent.workspace import Workspace, WorkspaceError

KEEP_RECENT_TOOL_OUTPUTS = 6
ELIDE_ABOVE_CHARS = 400


@dataclass
class LoopResult:
    messages: list[BaseMessage]
    steps: int
    finished: bool  # False = step budget exhausted or the model failed
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None  # every model in the chain failed mid-run
    route: str | None = None  # model route picked by the router, if any


# --- context compaction -------------------------------------------------------


def compact(history: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Shrink old tool output while keeping what the agent is working with.

    - The latest `read_file` of every path is kept: elided file contents made
      agents re-read the file they were editing in a loop.
    - Earlier reads of the same path are stale and are elided.
    - Other tool outputs older than the last few are elided when long.

    Tool-call / tool-result pairing stays intact, so providers accept it.
    """
    read_paths = _read_paths(history)
    latest_read = {path: i for i, path in read_paths.items()}
    keep = set(latest_read.values())
    other = [i for i, m in enumerate(history) if isinstance(m, ToolMessage) and i not in read_paths]
    keep |= set(other[-KEEP_RECENT_TOOL_OUTPUTS:])
    out: list[BaseMessage] = []
    for i, message in enumerate(history):
        if isinstance(message, ToolMessage) and i not in keep:
            text = str(message.content)
            if len(text) > ELIDE_ABOVE_CHARS or i in read_paths:
                note = "superseded by a later read" if i in read_paths else "elided"
                message = ToolMessage(
                    f"[{message.name or 'tool'} output {note} ({len(text)} chars)]",
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
        out.append(message)
    return out


def _read_paths(history: Sequence[BaseMessage]) -> dict[int, str]:
    """Index of each read_file result -> the path it read."""
    paths: dict[str, str] = {}  # tool_call_id -> path
    for message in history:
        if isinstance(message, AIMessage):
            for tc in message.tool_calls:
                if tc["name"] == "read_file":
                    paths[tc["id"] or ""] = str(tc["args"].get("path", ""))
    return {
        i: paths[m.tool_call_id]
        for i, m in enumerate(history)
        if isinstance(m, ToolMessage) and m.tool_call_id in paths
    }


class CompactionMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        compacted: Any = compact(request.messages)
        return handler(request.override(messages=compacted))


# --- loop hygiene ---------------------------------------------------------------


class InvalidToolCallMiddleware(AgentMiddleware):
    """`create_agent` ends the run on unparseable tool calls; ask for valid JSON."""

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or last.tool_calls or not last.invalid_tool_calls:
            return None
        errors = [
            ToolMessage(
                f"ERROR: could not parse arguments: {bad.get('error')}. Send valid JSON.",
                tool_call_id=bad.get("id") or "",
                name=bad.get("name") or None,
            )
            for bad in last.invalid_tool_calls
        ]
        return {"messages": errors, "jump_to": "model"}


class StepTrackerMiddleware(AgentMiddleware):
    """Counts real model calls; survives a model failure mid-run."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = 0
        self.tool_calls: list[str] = []
        self.finished = False

    def after_model(self, state: Any, runtime: Any) -> None:
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            self.steps += 1
            self.tool_calls += [tc["name"] for tc in last.tool_calls]
            self.finished = not last.tool_calls and not last.invalid_tool_calls


# --- Jev model routing ----------------------------------------------------------


class JevModelRouter(ModelRouterMiddleware):
    """`ModelRouterMiddleware` with our fallback semantics.

    The stock middleware lets classifier failures terminate the run and always
    trusts the top label. Here the decision goes through the decision fabric:
    low confidence or a Jev error falls back to `default`, and the decision is
    logged with the others.
    """

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: str,
        decisions: Decisions,
        default: str,
    ) -> None:
        # Deliberately not calling super().__init__: it builds a classifier from
        # the TYPESAFE_API_KEY env var; we route through `decisions` instead.
        AgentMiddleware.__init__(self)
        self.config = _ModelRouterConfig.model_validate(
            {"choices": choices, "instructions": instructions}
        )
        self.models = {}
        for route, choice in self.config.choices.items():
            if not isinstance(choice.model, BaseChatModel):
                raise TypeError(f"route {route!r} needs a model instance, not {choice.model!r}")
            self.models[route] = choice.model
        self.decisions = decisions
        self.default = default

    def before_agent(self, state: Any, runtime: Any) -> dict[str, ChoiceAnswer]:
        task = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        route = self.decisions.route_model(
            str(task.content) if task else "",
            {r: str(c.criteria) for r, c in self.config.choices.items()},
            str(self.config.instructions),
            self.default,
        )
        answer = ChoiceAnswer(
            type="choice", choice=route, probabilities={route: 1.0}, confidence=1.0
        )
        return {"model_route": answer}


# --- Jev write gate ---------------------------------------------------------------

WRITE_TOOLS = ("write_file", "replace_in_file")


class JevWriteGateMiddleware(AgentMiddleware):
    """Before an edit runs, ask Jev whether it breaks a rule code can't check.

    A flagged edit goes to the human approver through the policy, so it is
    audited like any other approval. Deterministic policy still runs inside
    the tool afterwards.
    """

    def __init__(
        self,
        *,
        decisions: Decisions,
        policy: Policy,
        workspace: Workspace,
        ticket_title: str,
        plan_summary: str,
    ) -> None:
        super().__init__()
        self.decisions = decisions
        self.policy = policy
        self.ws = workspace
        self.ticket_title = ticket_title
        self.plan_summary = plan_summary

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        if call["name"] in WRITE_TOOLS and (proposal := self._proposal(call["args"])):
            path, before, after = proposal
            reason = self.decisions.write_gate(
                self.ticket_title, self.plan_summary, self.policy.advisory, path, before, after
            )
            if reason and (denied := self.policy.request_write_approval(path, reason)):
                return ToolMessage(
                    denied, tool_call_id=call["id"] or "", name=call["name"], status="error"
                )
        return handler(request)

    def _proposal(self, args: Mapping[str, Any]) -> tuple[str, str | None, str] | None:
        """(path, current content, content after the edit), or None if unparseable."""
        try:
            target = self.ws.resolve(str(args.get("path", "")))
            path = target.relative_to(self.ws.root).as_posix()
            before = target.read_text() if target.exists() else None
        except (WorkspaceError, OSError, UnicodeDecodeError):
            return None  # the tool will report the problem itself
        if "content" in args:
            return path, before, str(args["content"])
        old, new = str(args.get("old", "")), str(args.get("new", ""))
        if before is None or before.count(old) != 1:
            return None
        return path, before, before.replace(old, new, 1)


# --- running an agent -------------------------------------------------------------


def run_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str,
    task: str,
    max_steps: int,
    middleware: Sequence[AgentMiddleware] = (),
) -> LoopResult:
    tracker = StepTrackerMiddleware()
    stack: list[Any] = [
        InvalidToolCallMiddleware(),
        ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="end"),
        CompactionMiddleware(),
        *middleware,
        # after_model hooks run last-to-first and a jump skips the rest, so the
        # tracker goes last to see every model reply.
        tracker,
    ]
    agent = create_agent(model, tools=list(tools), system_prompt=system_prompt, middleware=stack)
    # Each step is a model node plus a tools node, with middleware nodes around.
    config: RunnableConfig = {"recursion_limit": max_steps * 4 + 20}
    request: Any = {"messages": [HumanMessage(task)]}
    try:
        state = agent.invoke(request, config)
    except LLMError as exc:
        return LoopResult([], tracker.steps, False, tracker.tool_calls, error=str(exc))
    route = state.get("model_route")
    return LoopResult(
        list(state["messages"]),
        tracker.steps,
        tracker.finished,
        tracker.tool_calls,
        route=route.choice if route is not None else None,
    )
