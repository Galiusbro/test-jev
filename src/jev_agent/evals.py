"""Evaluation: run the case set in several modes and summarize the results.

Results are appended to a JSONL file one run at a time, so a long series can
be resumed. `summarize` turns them into the evaluation report.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_agent.config import Settings
from jev_agent.runner import Approvals, RunOutcome, execute_run, make_approver

MODES = ("jev", "no-jev")


@dataclass(frozen=True)
class Case:
    id: str
    ticket: Path
    approvals: Approvals
    expect: tuple[str, ...]
    checks: str = ""


def load_cases(path: Path, only: Sequence[str] = ()) -> list[Case]:
    data = json.loads(path.read_text())
    cases = [
        Case(
            id=c["id"],
            ticket=path.parent.parent / c["ticket"],  # relative to the project root
            approvals=Approvals(c["approvals"]),
            expect=tuple(c["expect"]),
            checks=c.get("checks", ""),
        )
        for c in data["cases"]
    ]
    return [c for c in cases if not only or c.id in only]


def done_keys(results: Path) -> set[tuple[str, str, int]]:
    if not results.exists():
        return set()
    rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    return {(r["case"], r["mode"], r["repeat"]) for r in rows}


def result_row(case: Case, mode: str, repeat: int, outcome: RunOutcome) -> dict[str, Any]:
    report = outcome.report
    metrics = report.get("metrics") or {}
    decisions = (report.get("jev") or {}).get("decisions") or []
    return {
        "case": case.id,
        "mode": mode,
        "repeat": repeat,
        "status": outcome.status,
        "passed": outcome.status in case.expect,
        "expect": list(case.expect),
        "duration_s": report.get("duration_s"),
        "llm_calls": metrics.get("llm_calls", 0),
        "failed_attempts": metrics.get("failed_attempts", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "repairs": len(report.get("repairs") or []),
        "jev_decisions": len(decisions),
        "jev_seconds": round(sum(d.get("latency_s", 0) for d in decisions), 2),
        "error": report.get("error"),
        "run_dir": str(outcome.run_dir),
        "trace_url": outcome.trace_url,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def run_series(
    cases: Sequence[Case],
    *,
    settings: Settings,
    modes: Sequence[str],
    repeats: int,
    results: Path,
    repo: Path,
    runs_dir: Path,
    log: Callable[[str], None],
) -> None:
    results.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(results)
    for repeat in range(1, repeats + 1):
        for case in cases:
            for mode in modes:
                if (case.id, mode, repeat) in skip:
                    continue
                log(f"[bold]{case.id}[/] · {mode} · #{repeat}")
                outcome = execute_run(
                    case.ticket,
                    settings=settings,
                    repo=repo,
                    runs_dir=runs_dir,
                    approver=make_approver(case.approvals, log=lambda _m: None),
                    jev=mode == "jev",
                    tags=["eval", f"case:{case.id}", f"approvals:{case.approvals.value}"],
                    metadata={"eval_case": case.id, "eval_mode": mode, "eval_repeat": repeat},
                    on_update=lambda node, _s: log(f"    {node}"),
                )
                row = result_row(case, mode, repeat, outcome)
                with results.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                verdict = "[green]pass[/]" if row["passed"] else "[red]FAIL[/]"
                log(f"  → {row['status']} {verdict} ({row['duration_s']}s)")


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def summarize(rows: Sequence[dict[str, Any]]) -> str:
    """Markdown report: overall per mode, then per case and mode."""
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
        by_case[(r["case"], r["mode"])].append(r)

    lines = [
        "| Mode | Runs | Correct outcome | Median time, s | Total time, min "
        "| Input tokens (total) | Repairs | Failed LLM attempts | Jev calls (total s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in sorted(by_mode):
        rs = by_mode[mode]
        passed = sum(r["passed"] for r in rs)
        durations = [r["duration_s"] for r in rs if r["duration_s"] is not None]
        lines.append(
            f"| {mode} | {len(rs)} | {passed}/{len(rs)} | {_fmt(_median(durations))} "
            f"| {_fmt(sum(durations) / 60, 1)} | {_fmt(sum(r['input_tokens'] for r in rs))} "
            f"| {sum(r['repairs'] for r in rs)} | {sum(r['failed_attempts'] for r in rs)} "
            f"| {sum(r['jev_decisions'] for r in rs)} "
            f"({_fmt(sum(r['jev_seconds'] for r in rs), 1)}) |"
        )

    lines += [
        "",
        "| Case | Mode | Statuses | Correct | Median time, s | Median input tokens | Repairs |",
        "|---|---|---|---|---|---|---|",
    ]
    for (case, mode), rs in sorted(by_case.items()):
        statuses = ", ".join(r["status"] for r in rs)
        passed = sum(r["passed"] for r in rs)
        durations = [r["duration_s"] for r in rs if r["duration_s"] is not None]
        tokens = [r["input_tokens"] for r in rs]
        lines.append(
            f"| {case} | {mode} | {statuses} | {passed}/{len(rs)} "
            f"| {_fmt(_median(durations))} | {_fmt(_median(tokens))} "
            f"| {sum(r['repairs'] for r in rs)} |"
        )
    return "\n".join(lines) + "\n"


def read_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
