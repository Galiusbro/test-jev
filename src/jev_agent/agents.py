"""Planner, implementer, repairer and reviewer agents.

Tool-using agents run on the LangChain `create_agent` harness (`harness.py`):
bounded steps, context compaction and Jev middleware. The planner's final plan
and the review are structured outputs with a corrective retry.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from jev_agent.harness import LoopResult, run_agent
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

    @model_validator(mode="after")
    def _names_files(self) -> Plan:
        # A plan without files is an investigation note, not a plan; the policy
        # check would wave it through and the implementer would work blind.
        if not self.files_to_change and not self.files_to_create:
            raise ValueError(
                "the plan must name at least one file in files_to_change or files_to_create"
            )
        return self


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
    explore = run_agent(
        model,
        tools,
        _PLANNER_SYSTEM.format(rules=instructions.text),
        f"Ticket:\n\n{ticket.as_text()}",
        max_steps,
    )
    if explore.error:
        raise LLMError(explore.error)
    # Keep the findings, drop raw tool traffic: the structured call only needs
    # the ticket and the planner's conclusions.
    findings = _findings(explore)
    request = [
        SystemMessage(_PLANNER_SYSTEM.format(rules=instructions.text)),
        HumanMessage(f"Ticket:\n\n{ticket.as_text()}"),
        HumanMessage(f"Your investigation notes:\n\n{findings}\n\n{_PLAN_REQUEST}"),
    ]
    return structured(model, Plan, request), explore


def _findings(explore: LoopResult) -> str:
    """The planner's notes; if exploration hit the step limit, keep what it got."""
    if explore.finished and explore.messages:
        return explore.messages[-1].text
    notes = [m.text for m in explore.messages if isinstance(m, AIMessage) and m.text.strip()]
    read = sorted(
        {
            str(tc["args"].get("path", ""))
            for m in explore.messages
            if isinstance(m, AIMessage)
            for tc in m.tool_calls
            if tc["name"] == "read_file"
        }
    )
    parts = ["(Exploration stopped at the step limit.)"]
    if read:
        parts.append("Files inspected: " + ", ".join(read))
    if notes:
        parts.append("Notes so far:\n" + "\n".join(notes))
    return "\n".join(parts)


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
    middleware: Sequence[AgentMiddleware] = (),
) -> LoopResult:
    return run_agent(
        model,
        tools,
        _IMPLEMENTER_SYSTEM.format(rules=instructions.text),
        f"Ticket:\n\n{ticket.as_text()}\n\nApproved plan:\n\n{plan.model_dump_json(indent=2)}",
        max_steps,
        middleware,
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
    middleware: Sequence[AgentMiddleware] = (),
) -> LoopResult:
    return run_agent(
        model,
        tools,
        _REPAIR_SYSTEM.format(rules=instructions.text),
        f"Ticket:\n\n{ticket.as_text()}\n\nPlan summary: {plan.summary}\n\n"
        f"Current diff:\n\n```diff\n{_clip(diff, 12_000)}\n```\n\n"
        f"Problems to fix:\n\n{problems}",
        max_steps,
        middleware,
    )


class Finding(BaseModel):
    severity: Literal["blocker", "major", "minor"] = Field(
        description="blocker: wrong/unsafe; major: violates ticket or project rules; "
        "minor: style or nice-to-have."
    )
    file: str
    issue: str
    suggestion: str = ""
    evidence: str = Field(
        default="",
        description="The exact line(s) copied from the diff that show the problem. "
        "Required for blocker/major findings.",
    )


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

    def grounded(self, diff: str) -> Review:
        """Downgrade blocking findings whose evidence is not in the diff.

        Reviewers hallucinate; a blocking finding must quote the code it is
        about. Unsupported ones are kept for the record as `minor`.
        """
        haystack = _normalize(diff)
        findings = []
        for f in self.findings:
            quoted = [_normalize(line) for line in f.evidence.splitlines() if line.strip()]
            supported = bool(quoted) and all(q in haystack for q in quoted)
            if f.severity in ("blocker", "major") and not supported:
                f = f.model_copy(update={"severity": "minor", "issue": f"[unverified] {f.issue}"})
            findings.append(f)
        return self.model_copy(update={"findings": findings})


_REVIEW_SYSTEM = """\
You are an independent code reviewer. You did not write this change. Review
the diff against the ticket and the project rules. Check for: requirement
mismatches (including off-by-one and boundary behaviour), missing tests,
security problems, violations of the project conventions, and changes outside
the ticket's scope. Automated tests, lint and type checks already pass.

Security checklist — each of these is at least `major`:
- trusting client-controlled input for a security decision (headers such as
  `X-Forwarded-For`, query/body fields used for identity, rate limiting or
  authorization);
- secrets or credentials in code, logs or responses;
- injection (SQL, shell, path traversal) or missing input validation at a
  trust boundary;
- authentication or authorization bypasses, including via error paths.

Report only real problems. Use `blocker`/`major` only for issues that must be
fixed before merge, and for those copy the exact diff line(s) that show the
problem into `evidence` — findings without matching evidence are discarded as
unverified. Style preferences are `minor`. An empty findings list means
approve.

Project rules (AGENTS.md):
{rules}
"""


def review_change(
    model: BaseChatModel,
    ticket: Ticket,
    instructions: ProjectInstructions,
    diff: str,
) -> Review:
    review = structured(
        model,
        Review,
        [
            SystemMessage(_REVIEW_SYSTEM.format(rules=instructions.text)),
            HumanMessage(
                f"Ticket:\n\n{ticket.as_text()}\n\nDiff:\n\n```diff\n{_clip(diff, 20_000)}\n```"
            ),
        ],
        retries=2,
    )
    return review.grounded(diff)


def _normalize(text: str) -> str:
    """Whitespace-insensitive form; strips diff markers so quotes match either way."""
    lines = (line[1:] if line[:1] in "+- " else line for line in text.splitlines())
    return " ".join(" ".join(lines).split())


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…(truncated)…"
