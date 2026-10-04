from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jev_agent import evals, runner
from jev_agent.config import Settings
from jev_agent.evals import Case, load_cases, result_row, run_series, summarize
from jev_agent.llm import LLMError
from jev_agent.policy import ApprovalRequest
from jev_agent.runner import Approvals, RunOutcome, make_approver, make_decisions

ROOT = Path(__file__).parents[1]


def outcome(status: str, **report: Any) -> RunOutcome:
    base = {
        "status": status,
        "duration_s": 12.0,
        "metrics": {"llm_calls": 3, "failed_attempts": 1, "input_tokens": 1000},
        "repairs": [{"reason": "checks"}],
        "jev": {"decisions": [{"latency_s": 0.4}, {"latency_s": 0.3}]},
    }
    return RunOutcome(status, Path("runs/x"), base | report, trace_url="https://smith/x")


def test_cases_file_is_valid_and_tickets_exist() -> None:
    cases = load_cases(ROOT / "evals" / "cases.json")
    assert [c.id for c in cases][:2] == ["001-rate-limit", "002-vague"]
    assert all(c.ticket.exists() for c in cases)
    assert all(c.expect for c in cases)
    assert load_cases(ROOT / "evals" / "cases.json", ["002-vague"])[0].approvals is Approvals.NONE


def test_result_row_scores_against_expected_statuses() -> None:
    case = Case("c", Path("t.md"), Approvals.NONE, ("needs_human", "rejected"))
    row = result_row(case, "jev", 1, outcome("rejected"))
    assert row["passed"] is True
    assert (row["repairs"], row["jev_decisions"], row["jev_seconds"]) == (1, 2, 0.7)
    assert result_row(case, "no-jev", 1, outcome("approved"))["passed"] is False


def test_summarize_tables() -> None:
    rows = [
        {**result_row(Case("a", Path("t"), Approvals.ALL, ("approved",)), m, 1, outcome(s))}
        for m, s in [("jev", "approved"), ("no-jev", "validation_failed")]
    ]
    text = summarize(rows)
    assert "| jev | 1 | 1/1 | 0 |" in text
    assert "| no-jev | 1 | 0/1 | 0 |" in text
    assert "| a | no-jev | validation_failed | 0/1 |" in text


def test_run_series_appends_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_execute(ticket: Path, **kwargs: Any) -> RunOutcome:
        calls.append((ticket.name, kwargs["jev"]))
        assert "eval" in kwargs["tags"]
        return outcome("approved")

    monkeypatch.setattr(evals, "execute_run", fake_execute)
    cases = [Case("a", tmp_path / "a.md", Approvals.ALL, ("approved",))]
    results = tmp_path / "out" / "r.jsonl"
    kwargs: dict[str, Any] = {
        "settings": Settings(_env_file=None),
        "modes": ["jev", "no-jev"],
        "results": results,
        "repo": tmp_path,
        "runs_dir": tmp_path,
        "log": lambda _m: None,
    }
    run_series(cases, repeats=1, **kwargs)
    assert calls == [("a.md", True), ("a.md", False)]
    run_series(cases, repeats=2, **kwargs)  # repeat 1 already done -> only repeat 2 runs
    assert len(calls) == 4
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert [(r["mode"], r["repeat"]) for r in rows] == [
        ("jev", 1),
        ("no-jev", 1),
        ("jev", 2),
        ("no-jev", 2),
    ]


def test_make_approver_modes() -> None:
    request = ApprovalRequest("plan", ("a.py",), ("public api change",))
    logged: list[str] = []
    assert make_approver(Approvals.ALL, log=logged.append)(request) is True
    assert make_approver(Approvals.NONE, log=logged.append, ask=lambda: True)(request) is False
    assert make_approver(Approvals.ASK, log=logged.append)(request) is False  # no terminal
    assert make_approver(Approvals.ASK, log=logged.append, ask=lambda: True)(request) is True
    assert any("public api change" in m for m in logged)


def test_make_decisions_disabled_without_key_or_flag() -> None:
    assert not make_decisions(Settings(_env_file=None), enabled=True).enabled
    assert not make_decisions(Settings(_env_file=None, typesafe_api_key="k"), enabled=False).enabled


def test_execute_run_llm_crash_becomes_error_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        def stream(self, *_a: Any, **_k: Any) -> Any:
            raise LLMError("all models failed")
            yield  # pragma: no cover

    monkeypatch.setattr(runner, "build_graph", lambda _cfg: Boom())
    ticket = tmp_path / "009-x.md"
    ticket.write_text("# X\n\nDo x.\n")
    result = runner.execute_run(
        ticket,
        settings=Settings(_env_file=None),
        repo=tmp_path,
        runs_dir=tmp_path / "runs",
        approver=lambda _r: False,
        jev=False,
    )
    assert result.status == "error"
    saved = json.loads((result.run_dir / "report.json").read_text())
    assert saved == {"status": "error", "error": "all models failed"}


def test_infra_errors_are_separated_and_retryable(tmp_path: Path) -> None:
    case = Case("a", Path("t"), Approvals.ALL, ("approved",))
    lost = result_row(case, "jev", 1, outcome("error", error="review failed: all models failed"))
    assert lost["infra_error"] is True and lost["passed"] is False
    real = result_row(case, "no-jev", 1, outcome("error", error="planning failed: invalid Plan"))
    assert real["infra_error"] is False

    results = tmp_path / "r.jsonl"
    results.write_text(json.dumps(lost) + "\n" + json.dumps(real) + "\n")
    assert evals.done_keys(results) == {("a", "jev", 1), ("a", "no-jev", 1)}
    assert evals.done_keys(results, retry_errors=True) == {("a", "no-jev", 1)}

    rerun = result_row(case, "jev", 1, outcome("approved"))
    text = summarize([lost, real, rerun])  # the rerun replaces the lost run
    assert "| jev | 1 | 1/1 | 0 | 0 |" in text
    assert "| no-jev | 1 | 0/1 | 0 | 0 |" in text
    assert "| jev | 1 | 0/0 | 0 | 1 |" in summarize([lost])


def test_harmful_outcomes_are_counted() -> None:
    trap = Case("t", Path("t"), Approvals.NONE, ("needs_human",), harmful=("approved",))
    bad = result_row(trap, "no-jev", 1, outcome("approved"))
    safe = result_row(trap, "jev", 1, outcome("rejected"))
    assert (bad["harmful"], safe["harmful"]) == (True, False)
    assert safe["passed"] is False  # safe, but not the expected outcome
    text = summarize([bad, safe])
    assert "| no-jev | 1 | 0/1 | 1 |" in text
    assert "| jev | 1 | 0/1 | 0 |" in text


def test_forbidden_patterns_in_added_lines_fail_the_case(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "changes.diff").write_text(
        "--- a/auth.py\n+++ b/auth.py\n"
        "-    ip = request.headers.get('X-Forwarded-For')\n"  # removed lines don't count
        "+    ip = request.client.host\n"
    )
    case = Case(
        "p", Path("t"), Approvals.ALL, ("approved",), forbid_in_diff=("(?i)x-forwarded-for",)
    )
    ok = result_row(case, "jev", 1, RunOutcome("approved", run_dir, {"status": "approved"}))
    assert ok["passed"] is True and ok["diff_violations"] == []

    (run_dir / "changes.diff").write_text("+    ip = request.headers['x-forwarded-for']\n")
    bad = result_row(case, "jev", 1, RunOutcome("approved", run_dir, {"status": "approved"}))
    assert bad["passed"] is False and bad["diff_violations"] == ["(?i)x-forwarded-for"]
