"""Planner, implementer, repairer and reviewer agents.

Tool-using agents share one bounded loop: the model calls tools until it
answers without tool calls or the step budget runs out. No unbounded autonomy.
Old tool output is elided from the prompt so long loops stay affordable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from jev_agent.llm import LLMError
from jev_agent.project import ProjectInstructions
from jev_agent.tickets import Ticket


class Plan(BaseModel):
    """Implementation plan for a ticket."""

    summary: str = Field(description="One or two sentences: what will change and why.")
    files_to_change: list[str] = Field(description="Repo-relative paths that will be edited.")
    files_to_create: list[str] = Field(default_factory=list, description="New files, if any.")
    steps: list[str] = Field(description="Ordered, concrete implementation steps.")
    tests: list[str] = Field(description="Tests to add or update and what each one checks.")
    risk: Literal["low", "medium", "high"]
    public_api_change: bool = Field(description="New/removed/renamed endpoints, fields, codes.")
    db_migration: bool = Field(description="Any database schema change.")


@dataclass
class LoopResult:
    messages: list[BaseMessage]
    steps: int
    finished: bool  # False = step budget exhausted or the model failed
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None  # every model in the chain failed mid-loop


KEEP_RECENT_TOOL_OUTPUTS = 6
ELIDE_ABOVE_CHARS = 400


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


def run_tool_loop(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    messages: list[BaseMessage],
    max_steps: int,
) -> LoopResult:
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(list(tools))
    history = list(messages)
    called: list[str] = []
    for step in range(1, max_steps + 1):
        try:
            reply = bound.invoke(compact(history))
        except LLMError as exc:
            return LoopResult(history, step - 1, False, called, error=str(exc))
        assert isinstance(reply, AIMessage)
        history.append(reply)
        if not reply.tool_calls and not reply.invalid_tool_calls:
            return LoopResult(history, step, True, called)
        for call in reply.tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                output = f"ERROR: unknown tool {call['name']!r}; available: {sorted(by_name)}"
            else:
                output = str(tool.invoke(call["args"]))
            called.append(call["name"])
            history.append(ToolMessage(output, tool_call_id=call["id"] or "", name=call["name"]))
        for bad in reply.invalid_tool_calls:
            history.append(
                ToolMessage(
                    f"ERROR: could not parse arguments: {bad.get('error')}. Send valid JSON.",
                    tool_call_id=bad.get("id") or "",
                )
            )
    return LoopResult(history, max_steps, False, called)


_PLANNER_SYSTEM = """\
You are the planning stage of a controlled software-engineering agent.
Investigate the repository with the read-only tools, then stop calling tools
and reply with a short plain-text analysis. Do not write code.

Find: the files involved, existing patterns to follow, existing tests, and
anything the project rules say about this kind of change.

Project rules (AGENTS.md):
{rules}
"""

_PLAN_REQUEST = (
    "Now produce the implementation plan. Only list files you actually inspected "
    "or that must be created. Keep the change minimal and within the ticket's scope."
)


def plan_ticket(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    ticket: Ticket,
    instructions: ProjectInstructions,
    max_steps: int = 12,
) -> tuple[Plan, LoopResult]:
    explore = run_tool_loop(
        model,
        tools,
        [
            SystemMessage(_PLANNER_SYSTEM.format(rules=instructions.text)),
            HumanMessage(f"Ticket:\n\n{ticket.as_text()}"),
        ],
        max_steps,
    )
    if explore.error:
        raise LLMError(explore.error)
    # Keep the findings, drop raw tool traffic: the structured call only needs
    # the ticket and the planner's conclusions.
    findings = explore.messages[-1].text if explore.finished else ""
    request = [
        SystemMessage(_PLANNER_SYSTEM.format(rules=instructions.text)),
        HumanMessage(f"Ticket:\n\n{ticket.as_text()}"),
        HumanMessage(f"Your investigation notes:\n\n{findings}\n\n{_PLAN_REQUEST}"),
    ]
    return structured(model, Plan, request), explore


class StructuredOutputError(RuntimeError):
    pass


def structured[T: BaseModel](
    model: BaseChatModel, schema: type[T], messages: list[BaseMessage], retries: int = 1
) -> T:
    """Structured output with one corrective retry on missing/invalid output."""
    runnable = model.with_structured_output(schema, include_raw=True)
    history = list(messages)
    problem = ""
    for _ in range(retries + 1):
        result = runnable.invoke(history)
        assert isinstance(result, dict)
        parsed = result.get("parsed")
        if isinstance(parsed, schema):
            return parsed
        raw = result.get("raw")
        error = result.get("parsing_error")
        problem = f"invalid {schema.__name__}: {error}" if error else "no tool call"
        if isinstance(raw, AIMessage):
            history.append(raw)
            for tc in raw.tool_calls:
                history.append(ToolMessage(f"ERROR: {problem}", tool_call_id=tc["id"] or ""))
        history.append(
            HumanMessage(
                f"Your previous reply was not a valid `{schema.__name__}` call ({problem}). "
                f"Call the `{schema.__name__}` tool now with arguments matching its schema."
            )
        )
    raise StructuredOutputError(f"model did not return a valid {schema.__name__}: {problem}")


_IMPLEMENTER_SYSTEM = """\
You are the implementation stage of a controlled software-engineering agent.
Make the change described in the plan using the tools. Rules:

- Edit only the files the plan lists (plus new test files it names).
- Prefer `replace_in_file` for small edits; read a file before editing it.
- Follow the project's conventions and add the tests the plan asks for.
- Use `run_check` to run the tests (and other checks) yourself; run the tests
  before you finish and fix what fails. Formatting is applied automatically.
- When the change is complete and the tests pass, stop calling tools and reply
  with a short summary.

Project rules (AGENTS.md):
{rules}
"""


def implement_plan(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    ticket: Ticket,
    plan: Plan,
    instructions: ProjectInstructions,
    max_steps: int = 30,
) -> LoopResult:
    return run_tool_loop(
        model,
        tools,
        [
            SystemMessage(_IMPLEMENTER_SYSTEM.format(rules=instructions.text)),
            HumanMessage(
                f"Ticket:\n\n{ticket.as_text()}\n\n"
                f"Approved plan:\n\n{plan.model_dump_json(indent=2)}"
            ),
        ],
        max_steps,
    )


_REPAIR_SYSTEM = """\
You are the repair stage of a controlled software-engineering agent. The
change for the ticket below is already in the working tree, but automated
checks or a code review found problems. Fix them with the tools.

- Read the failing output carefully; fix the cause, not the symptom.
- Keep the change within the ticket's scope. Never weaken, skip or delete
  pre-existing tests; tests added for this ticket may be corrected if they
  contradict the ticket.
- Use `run_check` to reproduce the failure first, form a hypothesis, then
  verify your fix by running the check again. Formatting is applied
  automatically.
- When the checks pass, stop calling tools and reply with a short summary.

Project rules (AGENTS.md):
{rules}
"""


def repair_change(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    ticket: Ticket,
    plan: Plan,
    instructions: ProjectInstructions,
    problems: str,
    diff: str,
    max_steps: int = 15,
) -> LoopResult:
    return run_tool_loop(
        model,
        tools,
        [
            SystemMessage(_REPAIR_SYSTEM.format(rules=instructions.text)),
            HumanMessage(
                f"Ticket:\n\n{ticket.as_text()}\n\nPlan summary: {plan.summary}\n\n"
                f"Current diff:\n\n```diff\n{_clip(diff, 12_000)}\n```\n\n"
                f"Problems to fix:\n\n{problems}"
            ),
        ],
        max_steps,
    )


class Finding(BaseModel):
    severity: Literal["blocker", "major", "minor"] = Field(
        description="blocker: wrong/unsafe; major: violates ticket or project rules; "
        "minor: style or nice-to-have."
    )
    file: str
    issue: str
    suggestion: str


class Review(BaseModel):
    """Code review of a change."""

    summary: str = Field(description="One or two sentences on the overall change.")
    findings: list[Finding] = Field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity in ("blocker", "major")]

    @property
    def approved(self) -> bool:
        # Decided by code from the findings, not by a model-reported flag.
        return not self.blocking


_REVIEW_SYSTEM = """\
You are an independent code reviewer. You did not write this change. Review
the diff against the ticket and the project rules. Check for: requirement
mismatches (including off-by-one and boundary behaviour), missing tests,
security problems, violations of the project conventions, and changes outside
the ticket's scope. Automated tests, lint and type checks already pass.

Report only real problems. Use `blocker`/`major` only for issues that must be
fixed before merge; style preferences are `minor`. An empty findings list
means approve.

Project rules (AGENTS.md):
{rules}
"""


def review_change(
    model: BaseChatModel,
    ticket: Ticket,
    instructions: ProjectInstructions,
    diff: str,
) -> Review:
    return structured(
        model,
        Review,
        [
            SystemMessage(_REVIEW_SYSTEM.format(rules=instructions.text)),
            HumanMessage(
                f"Ticket:\n\n{ticket.as_text()}\n\nDiff:\n\n```diff\n{_clip(diff, 20_000)}\n```"
            ),
        ],
    )


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…(truncated)…"
