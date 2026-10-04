"""`jev-agent` command line."""

from __future__ import annotations

import json
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from jev_agent.agents import StructuredOutputError
from jev_agent.bench import CANDIDATES, run_bench
from jev_agent.config import ModelTier, get_settings
from jev_agent.decisions import (
    BooleanQuestion,
    ChoiceQuestion,
    HttpJevClient,
    JevError,
    ScoreQuestion,
)
from jev_agent.decisions.client import State
from jev_agent.graph import RunConfig, build_graph
from jev_agent.llm import CallLog, LLMConfigError, LLMError, chat_model, list_models
from jev_agent.policy import ApprovalRequest, Approver
from jev_agent.tickets import load_ticket

app = typer.Typer(help="Controlled ticket-to-PR agent.", no_args_is_help=True)
console = Console()


@app.command()
def doctor() -> None:
    """Check API keys and make one live call to NVIDIA and to Jev."""
    settings = get_settings()
    ok = True

    for tier in ModelTier:
        log = CallLog()
        try:
            reply = chat_model(tier, settings, call_log=log).invoke(
                "Reply with the single word: pong"
            )
        except (LLMConfigError, LLMError) as exc:
            ok = False
            console.print(f"[red]✗[/] NVIDIA {tier}: {exc}")
            continue
        answered = log.records[-1]
        skipped = [f"{r.model} ({r.error})" for r in log.records[:-1]]
        console.print(
            f"[green]✓[/] NVIDIA {tier}: {answered.model} in {answered.elapsed_s:.1f}s → "
            f"{str(reply.content).strip()[:30]!r}"
            + (f" [yellow]after fallback from {', '.join(skipped)}[/]" if skipped else "")
        )

    try:
        evaluation = HttpJevClient.from_settings(settings).evaluate(
            {
                "ticket": "Login returns HTTP 500 when the password field is empty.",
                "result": "Fix applied; all tests pass; diff touches only auth.py and a test.",
            },
            {
                "done": BooleanQuestion(instructions="Is the task complete?"),
                "kind": ChoiceQuestion(
                    instructions="What kind of ticket is this?",
                    options={"bug": "Broken behaviour", "feature": "New behaviour"},
                ),
                "risk": ScoreQuestion(
                    instructions="How risky is this change?", levels=["low", "medium", "high"]
                ),
            },
        )
        done = evaluation.boolean("done")
        kind = evaluation.choice("kind")
        risk = evaluation.score("risk")
        console.print(
            f"[green]✓[/] Jev ({evaluation.model or settings.jev_model}): "
            f"done p={done.probability:.2f} · kind={kind.value} ({kind.confidence}) · "
            f"risk={risk.value} score={risk.score:.2f} ({risk.confidence})"
        )
    except JevError as exc:
        ok = False
        console.print(f"[red]✗[/] Jev: {exc}")

    raise typer.Exit(0 if ok else 1)


class Approvals(StrEnum):
    ASK = "ask"
    ALL = "all"
    NONE = "none"


@app.command()
def run(
    ticket_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Ticket .md")],
    repo: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("demo-api"),
    runs_dir: Annotated[Path, typer.Option()] = Path("runs"),
    approvals: Annotated[
        Approvals, typer.Option(help="ask: prompt (deny without a terminal); all; none")
    ] = Approvals.ASK,
) -> None:
    """Run the ticket-to-diff workflow on a copy of REPO."""
    settings = get_settings()
    ticket = load_ticket(ticket_path)
    run_dir = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{ticket.id}"
    run_dir.mkdir(parents=True)
    cfg = RunConfig(
        repo=repo,
        run_dir=run_dir,
        models=lambda tier, log: chat_model(tier, settings, call_log=log),
        approver=_approver(approvals),
    )
    console.print(f"[bold]{ticket.title}[/] → {run_dir}")
    state: dict[str, Any] = {"ticket": ticket, "started_at": time.time()}
    try:
        for update in build_graph(cfg).stream(state, stream_mode="updates"):
            for node, delta in update.items():
                state.update(delta or {})
                console.print(f"  [green]✓[/] {node}{_describe(node, state)}")
    except (LLMError, LLMConfigError, StructuredOutputError) as exc:
        console.print(f"  [red]✗[/] {exc}")
        raise typer.Exit(1) from exc
    if policy := state.get("policy"):
        for d in policy.audit:
            if d.verdict in ("deny", "rejected") and d.action != "plan":
                console.print(f"  [red]policy blocked[/] {d.action} {d.target}: {d.reason}")
    status = state.get("status")
    color = "green" if status == "approved" else "red"
    if error := state.get("error"):
        console.print(f"  [red]{error}[/]")
    console.print(f"[{color}]{status}[/] · report: {run_dir / 'report.json'}")
    raise typer.Exit(0 if status == "approved" else 1)


def _approver(mode: Approvals) -> Approver:
    def decide(request: ApprovalRequest) -> bool:
        reasons = "".join(f"\n      - {r}" for r in request.reasons)
        console.print(f"  [yellow]approval needed[/] ({request.action}):{reasons}")
        if mode is Approvals.ALL:
            console.print("    [green]auto-approved[/] (--approvals all)")
            return True
        if mode is Approvals.NONE or not sys.stdin.isatty():
            console.print("    [red]denied[/] (no interactive approval)")
            return False
        return Confirm.ask("    Approve?", default=False, console=console)

    return decide


def _describe(node: str, state: dict[str, Any]) -> str:
    if node == "prepare_workspace" and (setup := state.get("setup")) and not setup.ok:
        return f" [yellow](setup failed: {setup.output[-200:]})[/]"
    if node == "plan":
        if "plan" not in state:
            return " [red]failed[/]"
        plan = state["plan"]
        return f": {plan.summary} [dim](risk={plan.risk}, files={plan.files_to_change})[/]"
    if node == "policy_check":
        if state.get("status") == "rejected":
            return " [red]rejected[/]"
        plan_decision = [d for d in state["policy"].audit if d.action == "plan"][-1]
        return f": {plan_decision.verdict} — {plan_decision.reason}"
    if node == "implement":
        done = "" if state["implement_finished"] else " [yellow]did not finish[/]"
        return f": {state['implement_steps']} steps, changed {state['changed_files']}{done}"
    if node == "validate":
        return ": " + ", ".join(f"{'✓' if r.ok else '✗'} {r.command}" for r in state["validation"])
    if node == "review":
        if "review" not in state:
            return f" [red]{state.get('error', 'failed')}[/]"
        review = state["review"]
        verdict = "[green]approved[/]" if review.approved else "[yellow]changes requested[/]"
        lines = "".join(f"\n      [{f.severity}] {f.file}: {f.issue}" for f in review.findings)
        return f": {verdict} — {review.summary}{lines}"
    if node == "repair":
        last = state["repairs"][-1]
        return f" #{state['repair_attempts']} ({last['reason']}): {last['steps']} steps"
    return ""


@app.command()
def models(pattern: str = typer.Argument("", help="Substring to filter model ids.")) -> None:
    """List NVIDIA models available to your key."""
    try:
        ids = list_models(get_settings())
    except LLMConfigError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    for model_id in ids:
        if pattern.lower() in model_id.lower():
            console.print(model_id)


@app.command()
def bench(
    models: Annotated[
        list[str] | None, typer.Argument(help="Model ids. Default: built-in shortlist.")
    ] = None,
    trials: Annotated[int, typer.Option("--trials", "-n", min=1)] = 3,
    timeout: Annotated[float, typer.Option("--timeout", "-t", help="Seconds per request.")] = 90.0,
) -> None:
    """Measure latency and tool-calling reliability of NVIDIA models."""
    targets = models or list(dict.fromkeys(m for ms in CANDIDATES.values() for m in ms))
    try:
        results = run_bench(get_settings(), targets, trials, timeout_s=timeout)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    table = Table("model", "median s", "tool calls", "errors")
    for r in sorted(results, key=lambda r: (r.median_s is None, r.median_s or 0)):
        median = f"{r.median_s:.1f}" if r.median_s is not None else "—"
        table.add_row(r.model, median, f"{r.tool_calls_ok}/{r.trials}", ", ".join(r.errors))
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Yes/no question for Jev."),
    state: str = typer.Option(..., "--state", "-s", help="Evidence: text or JSON."),
) -> None:
    """Ask Jev a single yes/no question — handy for tuning question wording."""
    try:
        parsed: State = json.loads(state)
    except json.JSONDecodeError:
        parsed = state
    try:
        evaluation = HttpJevClient.from_settings(get_settings()).evaluate(
            parsed,
            {"q": BooleanQuestion(instructions=question)},
        )
    except JevError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    answer = evaluation.boolean("q")
    table = Table("probability", "confidence")
    table.add_row(f"{answer.probability:.3f}", str(answer.confidence))
    console.print(table)


if __name__ == "__main__":
    app()
