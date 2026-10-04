"""Ticket-to-diff workflow as a LangGraph state machine.

    prepare_workspace → plan → policy_check → implement → validate ─┬→ review ─┬→ report
                                     │                   ↑            │          │
                                     └→ report           └── repair ←─┴──────────┘
                                     (rejected)          (bounded: max_repair_attempts)

`validate` runs the project's autofix commands, then its checks. Failed checks
or blocking review findings send the change to `repair` while attempts remain.
`policy_check` applies the target's AGENTS.md rules to the plan (forbidden
files reject the run; approval-required changes ask the human), and the same
policy guards every tool call. Jev decision edges plug in later.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from jev_agent.agents import (
    Plan,
    Review,
    StructuredOutputError,
    implement_plan,
    plan_ticket,
    repair_change,
    review_change,
)
from jev_agent.config import ModelTier
from jev_agent.llm import CallLog, LLMError
from jev_agent.policy import Approver, Policy, deny_all, parse_rules
from jev_agent.project import ProjectInstructions, load_instructions
from jev_agent.tickets import Ticket
from jev_agent.tools import make_tools
from jev_agent.workspace import CommandResult, Workspace

Status = Literal[
    "approved", "changes_requested", "validation_failed", "no_changes", "rejected", "error"
]
ModelFactory = Callable[[ModelTier, CallLog], BaseChatModel]


@dataclass
class RunConfig:
    repo: Path
    run_dir: Path
    models: ModelFactory
    call_log: CallLog = field(default_factory=CallLog)
    plan_max_steps: int = 12
    implement_max_steps: int = 30
    repair_max_steps: int = 15
    max_repair_attempts: int = 3
    setup_command: str | None = "uv sync --quiet"
    approver: Approver = deny_all


class RunState(TypedDict, total=False):
    ticket: Ticket
    workspace: Workspace
    instructions: ProjectInstructions
    policy: Policy
    setup: CommandResult
    plan: Plan
    implement_finished: bool
    implement_steps: int
    tool_calls: list[str]
    changed_files: list[str]
    autofix: list[CommandResult]
    validation: list[CommandResult]
    checks_passed: bool
    review: Review | None  # None: the latest review failed — never route on a stale one
    repair_attempts: int
    repairs: list[dict[str, Any]]
    status: Status
    error: str
    started_at: float


def build_graph(cfg: RunConfig) -> Any:
    def model(tier: ModelTier) -> BaseChatModel:
        return cfg.models(tier, cfg.call_log)

    def writer_tools(state: RunState) -> list[BaseTool]:
        instructions = state["instructions"]
        return make_tools(
            state["workspace"],
            writable=True,
            policy=state["policy"],
            checks=instructions.commands,
            autofix=list(instructions.autofix.values()),
        )

    def prepare_workspace(state: RunState) -> RunState:
        ws = Workspace.create(cfg.repo, cfg.run_dir)
        instructions = load_instructions(ws.root)
        update: RunState = {
            "workspace": ws,
            "instructions": instructions,
            "policy": Policy(parse_rules(instructions.text), cfg.approver, ws.baseline),
            "repair_attempts": 0,
            "repairs": [],
        }
        if cfg.setup_command:
            update["setup"] = ws.run(cfg.setup_command)
        return update

    def plan(state: RunState) -> RunState:
        tools = make_tools(state["workspace"], writable=False, policy=state["policy"])
        try:
            result, _ = plan_ticket(
                model(ModelTier.STRONG),
                tools,
                state["ticket"],
                state["instructions"],
                cfg.plan_max_steps,
            )
        except (LLMError, StructuredOutputError) as exc:
            return {"status": "error", "error": f"planning failed: {exc}"}
        return {"plan": result}

    def policy_check(state: RunState) -> RunState:
        verdict = state["policy"].check_plan(state["plan"])
        if verdict.approved:
            return {}
        return {"status": "rejected", "error": "plan rejected: " + "; ".join(verdict.reasons)}

    def implement(state: RunState) -> RunState:
        tools = writer_tools(state)
        loop = implement_plan(
            model(ModelTier.CODER),
            tools,
            state["ticket"],
            state["plan"],
            state["instructions"],
            cfg.implement_max_steps,
        )
        update: RunState = {
            "implement_finished": loop.finished,
            "implement_steps": loop.steps,
            "tool_calls": loop.tool_calls,
            "changed_files": state["workspace"].changed_files(),
        }
        if loop.error:
            update["error"] = f"implementation stopped: {loop.error}"
        return update

    def validate(state: RunState) -> RunState:
        ws = state["workspace"]
        fixes = [ws.run(cmd) for cmd in state["instructions"].autofix.values()]
        results = [ws.run(cmd) for cmd in state["instructions"].commands.values()]
        return {
            "autofix": fixes,
            "validation": results,
            "checks_passed": all(r.ok for r in results),
            "changed_files": ws.changed_files(),
        }

    def review(state: RunState) -> RunState:
        try:
            result = review_change(
                model(ModelTier.STRONG),
                state["ticket"],
                state["instructions"],
                state["workspace"].diff(),
            )
        except (LLMError, StructuredOutputError) as exc:
            return {"review": None, "error": f"review failed: {exc}"}
        return {"review": result}

    def repair(state: RunState) -> RunState:
        if not state.get("checks_passed"):
            reason = "checks"
            problems = _failed_checks(state.get("validation", []))
        else:
            reason = "review"
            current = state.get("review")
            assert current is not None, "after_review only routes here with a review"
            problems = _blocking_findings(current)
        loop = repair_change(
            model(ModelTier.CODER),
            writer_tools(state),
            state["ticket"],
            state["plan"],
            state["instructions"],
            problems,
            state["workspace"].diff(),
            cfg.repair_max_steps,
        )
        record = {"reason": reason, "steps": loop.steps, "finished": loop.finished}
        if loop.error:
            record["error"] = loop.error
        return {
            "repair_attempts": state.get("repair_attempts", 0) + 1,
            "repairs": [*state.get("repairs", []), record],
        }

    def report(state: RunState) -> RunState:
        status = final_status(state)
        write_report(cfg, {**state, "status": status})
        return {"status": status}

    def after_plan(state: RunState) -> str:
        return "report" if state.get("status") == "error" else "policy_check"

    def after_policy(state: RunState) -> str:
        return "report" if state.get("status") == "rejected" else "implement"

    def can_repair(state: RunState) -> bool:
        return state.get("repair_attempts", 0) < cfg.max_repair_attempts

    def after_validate(state: RunState) -> str:
        if not state.get("changed_files"):
            return "report"
        if state.get("checks_passed"):
            return "review"
        return "repair" if can_repair(state) else "report"

    def after_review(state: RunState) -> str:
        result = state.get("review")
        if result is None or result.approved:
            return "report"
        return "repair" if can_repair(state) else "report"

    graph = StateGraph(RunState)
    for name, node in [
        ("prepare_workspace", prepare_workspace),
        ("plan", plan),
        ("policy_check", policy_check),
        ("implement", implement),
        ("validate", validate),
        ("review", review),
        ("repair", repair),
        ("report", report),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "prepare_workspace")
    graph.add_edge("prepare_workspace", "plan")
    graph.add_conditional_edges("plan", after_plan, ["policy_check", "report"])
    graph.add_conditional_edges("policy_check", after_policy, ["implement", "report"])
    graph.add_edge("implement", "validate")
    graph.add_conditional_edges("validate", after_validate, ["review", "repair", "report"])
    graph.add_conditional_edges("review", after_review, ["repair", "report"])
    graph.add_edge("repair", "validate")
    graph.add_edge("report", END)
    return graph.compile()


def final_status(state: RunState) -> Status:
    if state.get("status") in ("error", "rejected"):
        return state["status"]
    if not state.get("changed_files"):
        return "no_changes"
    if not state.get("checks_passed"):
        return "validation_failed"
    result = state.get("review")
    if result is None:
        return "error"  # review itself failed
    return "approved" if result.approved else "changes_requested"


def _failed_checks(results: list[CommandResult]) -> str:
    parts = [
        f"`{r.command}` failed (exit {r.exit_code}):\n```\n{r.output[-3000:]}\n```"
        for r in results
        if not r.ok
    ]
    return "Automated checks failed:\n\n" + "\n\n".join(parts)


def _blocking_findings(result: Review) -> str:
    lines = [
        f"- [{f.severity}] {f.file}: {f.issue} Suggested fix: {f.suggestion}"
        for f in result.blocking
    ]
    return "A code review requested changes:\n\n" + "\n".join(lines)


def policy_report(policy: Policy) -> dict[str, Any]:
    return {
        "rules": [{"kind": r.kind, "text": r.text, "enforced": r.enforced} for r in policy.rules],
        "audit": [asdict(d) for d in policy.audit],
        "approved_paths": sorted(policy.approved_paths),
    }


def metrics(log: CallLog) -> dict[str, Any]:
    ok = [r for r in log.records if r.ok]
    return {
        "llm_calls": len(ok),
        "failed_attempts": len(log.records) - len(ok),
        "input_tokens": sum(r.input_tokens or 0 for r in ok),
        "output_tokens": sum(r.output_tokens or 0 for r in ok),
        "llm_seconds": round(sum(r.elapsed_s for r in log.records), 1),
        "models": sorted({r.model for r in ok}),
    }


def write_report(cfg: RunConfig, state: RunState) -> dict[str, Any]:
    ws = state["workspace"]
    diff = ws.diff()
    (cfg.run_dir / "changes.diff").write_text(diff)
    plan = state.get("plan")
    review = state.get("review")
    data = {
        "ticket": state["ticket"].model_dump(),
        "status": state.get("status"),
        "error": state.get("error"),
        "duration_s": round(time.time() - state.get("started_at", time.time()), 1),
        "plan": plan.model_dump() if plan else None,
        "changed_files": state.get("changed_files", []),
        "implement": {
            "finished": state.get("implement_finished"),
            "steps": state.get("implement_steps"),
            "tool_calls": state.get("tool_calls", []),
        },
        "repairs": state.get("repairs", []),
        "validation": [asdict(r) for r in state.get("validation", [])],
        "review": review.model_dump() | {"approved": review.approved} if review else None,
        "policy": policy_report(state["policy"]) if "policy" in state else None,
        "metrics": metrics(cfg.call_log),
        "llm_attempts": [asdict(r) for r in cfg.call_log.records],
    }
    (cfg.run_dir / "report.json").write_text(json.dumps(data, indent=2))
    return data
