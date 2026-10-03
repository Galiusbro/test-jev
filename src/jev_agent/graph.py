"""Ticket-to-diff workflow as a LangGraph state machine (M2: linear happy path).

    load → prepare_workspace → plan → implement → validate → report

Policy checks, repair loops, review and Jev decision edges plug into this graph
in later milestones.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from jev_agent.agents import Plan, StructuredOutputError, implement_plan, plan_ticket
from jev_agent.config import ModelTier
from jev_agent.llm import CallLog, LLMError
from jev_agent.project import ProjectInstructions, load_instructions
from jev_agent.tickets import Ticket
from jev_agent.tools import make_tools
from jev_agent.workspace import CommandResult, Workspace

Status = Literal["validated", "validation_failed", "incomplete", "no_changes", "error"]
ModelFactory = Callable[[ModelTier, CallLog], BaseChatModel]


@dataclass
class RunConfig:
    repo: Path
    run_dir: Path
    models: ModelFactory
    call_log: CallLog = field(default_factory=CallLog)
    plan_max_steps: int = 12
    implement_max_steps: int = 30
    setup_command: str | None = "uv sync --quiet"


class RunState(TypedDict, total=False):
    ticket: Ticket
    workspace: Workspace
    instructions: ProjectInstructions
    plan: Plan
    implement_finished: bool
    implement_steps: int
    tool_calls: list[str]
    setup: CommandResult
    validation: list[CommandResult]
    changed_files: list[str]
    status: Status
    error: str
    started_at: float


def build_graph(cfg: RunConfig) -> Any:
    def model(tier: ModelTier) -> BaseChatModel:
        return cfg.models(tier, cfg.call_log)

    def prepare_workspace(state: RunState) -> RunState:
        ws = Workspace.create(cfg.repo, cfg.run_dir)
        update: RunState = {"workspace": ws, "instructions": load_instructions(ws.root)}
        if cfg.setup_command:
            update["setup"] = ws.run(cfg.setup_command)
        return update

    def plan(state: RunState) -> RunState:
        tools = make_tools(state["workspace"], writable=False)
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

    def after_plan(state: RunState) -> str:
        return "report" if state.get("status") == "error" else "implement"

    def implement(state: RunState) -> RunState:
        tools = make_tools(state["workspace"], writable=True)
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
        results = [ws.run(cmd) for cmd in state["instructions"].commands.values()]
        if not state.get("changed_files"):
            status: Status = "no_changes"
        elif not state.get("implement_finished"):
            status = "incomplete"
        elif all(r.ok for r in results):
            status = "validated"
        else:
            status = "validation_failed"
        return {"validation": results, "status": status}

    def report(state: RunState) -> RunState:
        write_report(cfg, state)
        return {}

    graph = StateGraph(RunState)
    graph.add_node("prepare_workspace", prepare_workspace)
    graph.add_node("plan", plan)
    graph.add_node("implement", implement)
    graph.add_node("validate", validate)
    graph.add_node("report", report)
    graph.add_edge(START, "prepare_workspace")
    graph.add_edge("prepare_workspace", "plan")
    graph.add_conditional_edges("plan", after_plan, ["implement", "report"])
    graph.add_edge("implement", "validate")
    graph.add_edge("validate", "report")
    graph.add_edge("report", END)
    return graph.compile()


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
        "validation": [asdict(r) for r in state.get("validation", [])],
        "metrics": metrics(cfg.call_log),
        "llm_attempts": [asdict(r) for r in cfg.call_log.records],
    }
    (cfg.run_dir / "report.json").write_text(json.dumps(data, indent=2))
    return data
