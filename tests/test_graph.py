from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from scripted_model import ScriptedChatModel, call, say

from jev_agent.agents import run_tool_loop
from jev_agent.config import ModelTier
from jev_agent.graph import RunConfig, build_graph
from jev_agent.llm import CallLog, CallRecord
from jev_agent.tickets import Ticket
from jev_agent.tools import make_tools
from jev_agent.workspace import Workspace

TICKET = Ticket(id="007-greeting", title="Say hello properly", body="hello() must return 'hello'.")

PLAN_ARGS: dict[str, Any] = {
    "summary": "Change hello() to return 'hello'.",
    "files_to_change": ["app/main.py"],
    "steps": ["Edit the return value"],
    "tests": ["check command asserts the new value"],
    "risk": "low",
    "public_api_change": False,
    "db_migration": False,
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "target"
    (root / "app").mkdir(parents=True)
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "main.py").write_text("def hello():\n    return 'hi'\n")
    (root / "AGENTS.md").write_text(
        "# Target\n\n## Commands\n\n"
        "- test: `python -c \"from app.main import hello; assert hello() == 'hello'\"`\n"
    )
    return root


def planner_script() -> list[AIMessage | Exception]:
    return [
        call("read_file", path="app/main.py"),
        say("hello() lives in app/main.py and returns 'hi'."),
        call("Plan", **PLAN_ARGS),
    ]


def run(
    repo: Path,
    tmp_path: Path,
    strong: Sequence[AIMessage | Exception],
    coder: Sequence[AIMessage | Exception],
    **cfg: Any,
) -> tuple[dict[str, Any], dict[str, ScriptedChatModel]]:
    models = {
        "strong": ScriptedChatModel(replies=strong),
        "coder": ScriptedChatModel(replies=coder),
    }

    def factory(tier: ModelTier, log: CallLog) -> ScriptedChatModel:
        log.add(CallRecord(model=f"scripted-{tier}", ok=True, elapsed_s=0.1, input_tokens=10))
        return models[tier.value]

    config = RunConfig(
        repo=repo, run_dir=tmp_path / "run", models=factory, setup_command=None, **cfg
    )
    state = build_graph(config).invoke({"ticket": TICKET, "started_at": 0.0})
    return state, models


APPROVE = call("Review", summary="Looks correct.", findings=[])
FIX = call("replace_in_file", path="app/main.py", old="'hi'", new="'hello'")


def finding(severity: str = "major") -> dict[str, str]:
    return {"severity": severity, "file": "app/main.py", "issue": "Bad.", "suggestion": "Fix."}


def test_happy_path_is_approved_and_writes_report(repo: Path, tmp_path: Path) -> None:
    coder = [FIX, say("Changed the return value.")]
    state, models = run(repo, tmp_path, [*planner_script(), APPROVE], coder)

    assert state["status"] == "approved"
    assert state["plan"].files_to_change == ["app/main.py"]
    assert state["changed_files"] == ["app/main.py"]
    assert state["tool_calls"] == ["replace_in_file"]
    assert state["repair_attempts"] == 0
    assert (repo / "app" / "main.py").read_text().endswith("'hi'\n")  # original untouched

    run_dir = tmp_path / "run"
    assert "+    return 'hello'" in (run_dir / "changes.diff").read_text()
    report = json.loads((run_dir / "report.json").read_text())
    assert report["status"] == "approved"
    assert report["review"]["approved"] is True
    assert report["validation"][0]["exit_code"] == 0
    assert report["plan"]["risk"] == "low"

    # planner only ever saw read-only tools; implementer got write tools
    assert models["strong"].bound_tools[0] == ["list_files", "read_file", "search"]
    assert "write_file" in models["coder"].bound_tools[0]
    # rules from AGENTS.md reach the agents; the reviewer sees the diff
    assert "## Commands" in str(models["coder"].seen[0][0].content)
    assert "+    return 'hello'" in str(models["strong"].seen[-1][-1].content)


def test_failed_checks_are_repaired_with_real_output(repo: Path, tmp_path: Path) -> None:
    coder = [
        call("replace_in_file", path="app/main.py", old="'hi'", new="'hey'"),
        say("done"),
        call("replace_in_file", path="app/main.py", old="'hey'", new="'hello'"),
        say("fixed the assertion"),
    ]
    state, models = run(repo, tmp_path, [*planner_script(), APPROVE], coder)

    assert state["status"] == "approved"
    assert state["repair_attempts"] == 1
    assert state["repairs"] == [{"reason": "checks", "steps": 2, "finished": True}]
    repair_prompt = str(models["coder"].seen[2][1].content)
    assert "Automated checks failed" in repair_prompt and "AssertionError" in repair_prompt
    assert "-    return 'hi'" in repair_prompt  # current diff included


def test_repair_budget_exhausted_reports_validation_failed(repo: Path, tmp_path: Path) -> None:
    coder = [
        call("replace_in_file", path="app/main.py", old="'hi'", new="'hey'"),
        say("done"),
        say("I could not find the problem."),
        say("Still stuck."),
    ]
    state, _ = run(repo, tmp_path, planner_script(), coder, max_repair_attempts=2)
    assert state["status"] == "validation_failed"
    assert state["repair_attempts"] == 2
    assert "AssertionError" in state["validation"][0].output


def test_blocking_review_findings_go_back_to_repair(repo: Path, tmp_path: Path) -> None:
    changes = call("Review", summary="Off by one.", findings=[finding("major")])
    minor_only = call("Review", summary="Fine.", findings=[finding("minor")])
    coder = [FIX, say("done"), call("read_file", path="app/main.py"), say("addressed")]
    state, models = run(repo, tmp_path, [*planner_script(), changes, minor_only], coder)

    assert state["status"] == "approved"  # minor findings don't block
    assert state["repairs"][0]["reason"] == "review"
    assert "[major] app/main.py: Bad. Suggested fix: Fix." in str(
        models["coder"].seen[2][1].content
    )
    assert [f.severity for f in state["review"].findings] == ["minor"]


def test_unresolved_review_is_changes_requested(repo: Path, tmp_path: Path) -> None:
    blocker = call("Review", summary="Unsafe.", findings=[finding("blocker")])
    coder = [FIX, say("done")]
    state, _ = run(repo, tmp_path, [*planner_script(), blocker], coder, max_repair_attempts=0)
    assert state["status"] == "changes_requested"
    assert not state["review"].approved


def test_review_failure_is_error(repo: Path, tmp_path: Path) -> None:
    from jev_agent.llm import LLMError

    strong: list[AIMessage | Exception] = [*planner_script(), LLMError("all models failed")]
    state, _ = run(repo, tmp_path, strong, [FIX, say("done")])
    assert state["status"] == "error"
    assert "review failed" in state["error"]


def test_no_changes(repo: Path, tmp_path: Path) -> None:
    state, _ = run(repo, tmp_path, planner_script(), [say("Nothing to do.")])
    assert state["status"] == "no_changes"


def test_autofix_runs_before_checks(repo: Path, tmp_path: Path) -> None:
    agents = repo / "AGENTS.md"
    agents.write_text(
        agents.read_text() + "\n## Autofix\n\n- fix: `sed -i.bak s/hey/hello/ app/main.py`\n"
    )
    coder = [
        call("replace_in_file", path="app/main.py", old="'hi'", new="'hey'"),
        say("done"),
    ]
    state, _ = run(repo, tmp_path, [*planner_script(), APPROVE], coder)
    assert state["status"] == "approved"  # the deterministic fixer made the check pass
    assert state["repair_attempts"] == 0


def test_loop_reports_unknown_tool_and_bad_arguments(tmp_path: Path, repo: Path) -> None:
    ws = Workspace.create(repo, tmp_path / "run")
    bad_args = AIMessage(
        "",
        invalid_tool_calls=[
            {
                "name": "read_file",
                "args": "{oops",
                "id": "b1",
                "error": "bad json",
                "type": "invalid_tool_call",
            }
        ],
    )
    model = ScriptedChatModel(replies=[call("rm_rf", path="/"), bad_args, say("ok")])
    result = run_tool_loop(model, make_tools(ws, writable=False), [HumanMessage("go")], 5)

    assert result.finished and result.steps == 3
    tool_messages = [m for m in result.messages if isinstance(m, ToolMessage)]
    assert "unknown tool 'rm_rf'" in str(tool_messages[0].content)
    assert "could not parse arguments: bad json" in str(tool_messages[1].content)


def test_structured_retries_after_text_reply() -> None:
    from jev_agent.agents import Plan, structured

    model = ScriptedChatModel(
        replies=[say("Here is my plan: change hello."), call("Plan", **PLAN_ARGS)]
    )
    plan = structured(model, Plan, [HumanMessage("plan it")])
    assert plan.summary == PLAN_ARGS["summary"]
    retry_prompt = model.seen[1][-1].content
    assert "not a valid `Plan` call (no tool call)" in str(retry_prompt)


def test_structured_gives_up_with_clear_error() -> None:
    from jev_agent.agents import Plan, StructuredOutputError, structured

    bad = call("Plan", summary="only a summary")
    model = ScriptedChatModel(replies=[bad, call("Plan", summary="still bad")])
    with pytest.raises(StructuredOutputError, match="invalid Plan"):
        structured(model, Plan, [HumanMessage("plan it")])
    # the invalid call was answered with a ToolMessage before the retry prompt
    assert isinstance(model.seen[1][-2], ToolMessage)


def test_planning_failure_ends_run_with_error_report(repo: Path, tmp_path: Path) -> None:
    from jev_agent.llm import LLMError

    strong: list[AIMessage | Exception] = [LLMError("all models failed — a: timeout")]
    state, models = run(repo, tmp_path, strong, [])

    assert state["status"] == "error"
    assert "planning failed" in state["error"]
    assert models["coder"].seen == []  # implement was skipped
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["status"] == "error" and "all models failed" in report["error"]


def test_llm_failure_mid_implementation_is_incomplete(repo: Path, tmp_path: Path) -> None:
    from jev_agent.llm import LLMError

    coder: list[AIMessage | Exception] = [
        call("replace_in_file", path="app/main.py", old="'hi'", new="'hello'"),
        LLMError("all models failed — b: empty response"),
    ]
    state, _ = run(repo, tmp_path, [*planner_script(), APPROVE], coder)

    assert state["status"] == "approved"  # partial work still passed checks and review
    assert state["implement_steps"] == 1
    assert "implementation stopped" in state["error"]
    assert state["changed_files"] == ["app/main.py"]  # partial work kept for inspection


def test_compact_keeps_latest_read_per_file_and_elides_the_rest() -> None:
    from jev_agent.agents import KEEP_RECENT_TOOL_OUTPUTS, compact

    history: list[Any] = [HumanMessage("go")]

    def tool_turn(name: str, call_id: str, output: str, **args: Any) -> None:
        history.append(call(name, call_id, **args))
        history.append(ToolMessage(output, tool_call_id=call_id, name=name))

    tool_turn("read_file", "r1", "old auth " * 100, path="auth.py")
    tool_turn("read_file", "m1", "main " * 100, path="main.py")
    for i in range(KEEP_RECENT_TOOL_OUTPUTS + 1):
        tool_turn("search", f"s{i}", f"hit {i} " * 100, pattern="x")
    tool_turn("read_file", "r2", "new auth " * 100, path="auth.py")

    by_id = {m.tool_call_id: str(m.content) for m in compact(history) if isinstance(m, ToolMessage)}
    assert by_id["r1"].startswith("[read_file output superseded by a later read")
    assert by_id["r2"].startswith("new auth")  # latest read of auth.py kept
    assert by_id["m1"].startswith("main")  # only read of main.py kept, however old
    assert by_id["s0"].startswith("[search output elided")  # old long output
    assert by_id[f"s{KEEP_RECENT_TOOL_OUTPUTS}"].startswith("hit")  # recent kept
    assert len(compact(history)) == len(history)
    assert str(history[2].content).startswith("old auth")  # input not mutated
