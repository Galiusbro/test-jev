"""`jev-agent` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from jev_agent.bench import CANDIDATES, run_bench
from jev_agent.config import ModelTier, get_settings
from jev_agent.decisions import (
    BooleanQuestion,
    ChoiceQuestion,
    JevError,
    ScoreQuestion,
    TypeSafeJevClient,
)
from jev_agent.decisions.client import State
from jev_agent.evals import MODES, load_cases, read_results, run_series, summarize
from jev_agent.llm import CallLog, LLMConfigError, LLMError, chat_model, list_models
from jev_agent.mcp_server import build_server
from jev_agent.observability import configure_tracing
from jev_agent.pr import PullRequestError, open_pull_request
from jev_agent.runner import (
    Approvals,
    RunOutcome,
    execute_run,
    make_approver,
    make_decisions,
)
from jev_agent.scaffold import init_repo

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
        evaluation = TypeSafeJevClient.from_settings(settings).evaluate(
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

    if configure_tracing(settings):
        try:
            from langsmith import Client

            next(iter(Client().list_projects(limit=1)), None)
            console.print(f"[green]✓[/] LangSmith: project {settings.langsmith_project!r}")
        except Exception as exc:  # noqa: BLE001 — report, keep exit code honest
            ok = False
            console.print(f"[red]✗[/] LangSmith: {exc}")
    else:
        console.print("[dim]- LangSmith: off (no LANGSMITH_API_KEY)[/]")

    raise typer.Exit(0 if ok else 1)


@app.command()
def run(
    ticket_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Ticket .md")],
    repo: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("demo-api"),
    runs_dir: Annotated[Path, typer.Option()] = Path("runs"),
    approvals: Annotated[
        Approvals, typer.Option(help="ask: prompt (deny without a terminal); all; none")
    ] = Approvals.ASK,
    jev: Annotated[bool, typer.Option(help="Use Jev decisions (off = pre-Jev baseline)")] = True,
    open_pr: Annotated[
        bool, typer.Option(help="Open a GitHub PR for the target repo when the run is approved.")
    ] = False,
) -> None:
    """Run the ticket-to-diff workflow on a copy of REPO."""
    settings = get_settings()
    ask = (
        (lambda: Confirm.ask("    Approve?", default=False, console=console))
        if sys.stdin.isatty()
        else None
    )

    def started(outcome: RunOutcome) -> None:
        console.print(
            f"[bold]{ticket_path.stem}[/] → {outcome.run_dir} "
            f"[dim](jev {'on' if outcome.jev_enabled else 'off'}, "
            f"langsmith {'on' if outcome.tracing else 'off'})[/]"
        )

    outcome = execute_run(
        ticket_path,
        settings=settings,
        repo=repo,
        runs_dir=runs_dir,
        approver=make_approver(approvals, log=console.print, ask=ask),
        jev=jev,
        tags=[f"approvals:{approvals.value}"],
        on_start=started,
        on_update=lambda node, state: console.print(
            f"  [green]✓[/] {node}{_describe(node, state)}"
        ),
    )
    if outcome.trace_url:
        console.print(f"  [dim]trace:[/] {outcome.trace_url}")
    if policy := outcome.state.get("policy"):
        for d in policy.audit:
            if d.verdict in ("deny", "rejected") and d.action != "plan":
                console.print(f"  [red]policy blocked[/] {d.action} {d.target}: {d.reason}")
    color = "green" if outcome.status == "approved" else "red"
    if error := outcome.report.get("error"):
        console.print(f"  [red]{error}[/]")
    console.print(f"[{color}]{outcome.status}[/] · report: {outcome.run_dir / 'report.json'}")
    if open_pr and outcome.status == "approved":
        try:
            url = open_pull_request(
                target_repo=repo,
                diff=(outcome.run_dir / "changes.diff").read_text(),
                report=outcome.report,
            )
        except PullRequestError as exc:
            console.print(f"[red]PR not opened:[/] {exc}")
            raise typer.Exit(1) from exc
        _add_to_report(outcome.run_dir, {"pull_request": url})
        console.print(f"[green]pull request:[/] {url}")
    raise typer.Exit(0 if outcome.status == "approved" else 1)


@app.command()
def init(
    repo: Annotated[Path, typer.Argument(exists=True, file_okay=False, help="Repo to set up")],
) -> None:
    """Bootstrap REPO with AGENTS.md, CLAUDE.md and .mcp.json (never overwrites)."""
    result = init_repo(repo, Path(__file__).resolve().parents[2])
    stack = result.stack
    console.print(f"[bold]{repo}[/]: detected [cyan]{stack.name}[/]")
    for name, cmd in stack.commands.items():
        console.print(f"  {name}: [dim]{cmd}[/]")
    for name in result.written:
        console.print(f"  [green]created[/] {name}")
    for name in result.skipped:
        console.print(f"  [yellow]kept existing[/] {name}")
    console.print(
        "Next: fill in the TODOs in AGENTS.md, review the rules, then try "
        f"`jev-agent run <ticket.md> --repo {repo}`."
    )


@app.command()
def mcp(
    repo: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("demo-api"),
    approvals: Annotated[
        Approvals, typer.Option(help="none: refuse approval-required writes; all: allow them")
    ] = Approvals.NONE,
    jev: Annotated[bool, typer.Option(help="Enable the Jev triage_ticket tool")] = True,
) -> None:
    """Serve REPO's governed tools over MCP (stdio) for any MCP client."""
    if approvals is Approvals.ASK:
        console.print("[red]--approvals ask needs a terminal; use none or all for MCP[/]")
        raise typer.Exit(2)
    server = build_server(
        repo, approvals=approvals, decisions=make_decisions(get_settings(), enabled=jev)
    )
    server.run("stdio")


@app.command("eval")
def eval_(
    cases_file: Annotated[Path, typer.Option("--cases", exists=True)] = Path("evals/cases.json"),
    only: Annotated[list[str] | None, typer.Option("--case", help="Case id; repeatable.")] = None,
    modes: Annotated[list[str] | None, typer.Option("--mode", help="jev / no-jev")] = None,
    repeats: Annotated[int, typer.Option(min=1)] = 1,
    results: Annotated[Path, typer.Option()] = Path("evals/results/latest.jsonl"),
    repo: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path("demo-api"),
    runs_dir: Annotated[Path, typer.Option()] = Path("runs"),
    retry_errors: Annotated[
        bool, typer.Option(help="Rerun runs lost to a provider outage (infra errors).")
    ] = False,
) -> None:
    """Run the eval cases in each mode; resumable (appends to RESULTS)."""
    chosen = modes or list(MODES)
    unknown = set(chosen) - set(MODES)
    if unknown:
        console.print(f"[red]unknown mode(s): {sorted(unknown)}; use {MODES}[/]")
        raise typer.Exit(2)
    cases = load_cases(cases_file, only or ())
    run_series(
        cases,
        settings=get_settings(),
        modes=chosen,
        repeats=repeats,
        results=results,
        repo=repo,
        runs_dir=runs_dir,
        log=console.print,
        retry_errors=retry_errors,
    )
    console.print(summarize(read_results(results)))


@app.command("eval-report")
def eval_report(
    results: Annotated[Path, typer.Argument(exists=True)] = Path("evals/results/latest.jsonl"),
) -> None:
    """Print the summary tables for an eval results file (Markdown)."""
    console.print(summarize(read_results(results)), markup=False, highlight=False)


def _add_to_report(run_dir: Path, fields: dict[str, Any]) -> None:
    path = run_dir / "report.json"
    if path.exists():
        path.write_text(json.dumps(json.loads(path.read_text()) | fields, indent=2))


def _describe(node: str, state: dict[str, Any]) -> str:
    if node == "prepare_workspace" and (setup := state.get("setup")) and not setup.ok:
        return f" [yellow](setup failed: {setup.output[-200:]})[/]"
    if node == "plan":
        if "plan" not in state:
            return " [red]failed[/]"
        plan = state["plan"]
        return f": {plan.summary} [dim](risk={plan.risk}, files={plan.files_to_change})[/]"
    if node == "triage":
        t = state["triage"]
        stop = f" [yellow]→ human: {t.stop_reason}[/]" if t.stop_reason else ""
        return f": {t.kind}, actionable={t.actionable}{stop}"
    if node == "policy_check":
        a = state.get("assessment")
        jev = f" [dim](complexity={a.complexity})[/]" if a else ""
        if state.get("status") == "rejected":
            return f" [red]rejected[/]{jev}"
        plan_decision = [d for d in state["policy"].audit if d.action == "plan"][-1]
        return f": {plan_decision.verdict} — {plan_decision.reason}{jev}"
    if node == "prove_tests":
        proof = state["proof"]
        mark = "[green]✓[/]" if proof.acceptable else "[red]✗[/]"
        return f": {mark} {proof.status} — {proof.detail}"
    if node == "diagnose":
        return f": failure looks like a [bold]{state['failure_kind']}[/] problem"
    if node == "implement":
        done = "" if state["implement_finished"] else " [yellow]did not finish[/]"
        return f": {state['implement_steps']} steps, changed {state['changed_files']}{done}"
    if node == "validate":
        return ": " + ", ".join(f"{'✓' if r.ok else '✗'} {r.command}" for r in state["validation"])
    if node == "review":
        review = state.get("review")
        if review is None:
            return f" [red]{state.get('error', 'failed')}[/]"
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
        evaluation = TypeSafeJevClient.from_settings(get_settings()).evaluate(
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
