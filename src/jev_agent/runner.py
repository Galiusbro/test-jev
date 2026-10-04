"""One workflow run, shared by `jev-agent run` and `jev-agent eval`."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.context import collect_runs

from jev_agent.agents import StructuredOutputError
from jev_agent.config import ModelTier, Settings
from jev_agent.decisions import TypeSafeJevClient
from jev_agent.decisions.fabric import Decisions
from jev_agent.graph import RunConfig, build_graph
from jev_agent.llm import LLMConfigError, LLMError, chat_model
from jev_agent.observability import configure_tracing, flush, run_url
from jev_agent.policy import ApprovalRequest, Approver
from jev_agent.tickets import load_ticket


class Approvals(StrEnum):
    ASK = "ask"
    ALL = "all"
    NONE = "none"


def make_approver(
    mode: Approvals,
    *,
    log: Callable[[str], None],
    ask: Callable[[], bool] | None = None,
) -> Approver:
    """`ask` prompts a human; without it (no terminal) ASK means deny."""

    def decide(request: ApprovalRequest) -> bool:
        reasons = "".join(f"\n      - {r}" for r in request.reasons)
        log(f"  [yellow]approval needed[/] ({request.action}):{reasons}")
        if mode is Approvals.ALL:
            log("    [green]auto-approved[/] (approvals: all)")
            return True
        if mode is Approvals.NONE or ask is None:
            log("    [red]denied[/] (no interactive approval)")
            return False
        return ask()

    return decide


def make_decisions(settings: Settings, *, enabled: bool) -> Decisions:
    if not enabled or settings.typesafe_api_key is None:
        return Decisions(None)
    return Decisions(TypeSafeJevClient.from_settings(settings), settings.jev_min_confidence)


@dataclass
class RunOutcome:
    status: str
    run_dir: Path
    report: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)
    trace_url: str | None = None
    jev_enabled: bool = False
    tracing: bool = False


def execute_run(
    ticket_path: Path,
    *,
    settings: Settings,
    repo: Path,
    runs_dir: Path,
    approver: Approver,
    jev: bool,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    on_start: Callable[[RunOutcome], None] | None = None,
    on_update: Callable[[str, dict[str, Any]], None] | None = None,
) -> RunOutcome:
    """Run the graph once. LLM failures end as status `error`, never raise."""
    ticket = load_ticket(ticket_path)
    run_dir = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{ticket.id}"
    run_dir.mkdir(parents=True)
    cfg = RunConfig(
        repo=repo,
        run_dir=run_dir,
        models=lambda tier, log: chat_model(tier, settings, call_log=log),
        approver=approver,
        decisions=make_decisions(settings, enabled=jev),
    )
    tracing = configure_tracing(settings)
    outcome = RunOutcome("running", run_dir, {}, jev_enabled=cfg.decisions.enabled, tracing=tracing)
    if on_start:
        on_start(outcome)

    state: dict[str, Any] = {"ticket": ticket, "started_at": time.time()}
    config: RunnableConfig = {
        "run_name": f"ticket {ticket.id}",
        "tags": ["jev" if cfg.decisions.enabled else "no-jev", *tags],
        "metadata": {
            "ticket": ticket.id,
            "jev": cfg.decisions.enabled,
            "run_dir": str(run_dir),
            "models": {t.value: settings.models_for(t) for t in ModelTier},
            **(metadata or {}),
        },
    }
    crash: str | None = None
    traced: list[Any] = []
    try:
        with collect_runs() as runs:
            for update in build_graph(cfg).stream(state, config, stream_mode="updates"):
                for node, delta in update.items():
                    state.update(delta or {})
                    if on_update:
                        on_update(node, state)
        traced = list(runs.traced_runs)
    except (LLMError, LLMConfigError, StructuredOutputError) as exc:
        crash = str(exc)
    finally:
        if tracing:
            flush()

    report_path = run_dir / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if crash:
        report |= {"status": "error", "error": crash}
    if tracing and traced and (url := run_url(traced[0], settings.langsmith_project)):
        report["langsmith_url"] = url
        outcome.trace_url = url
    report_path.write_text(json.dumps(report, indent=2))

    outcome.status = str(report.get("status") or state.get("status") or "error")
    outcome.report = report
    outcome.state = state
    return outcome
