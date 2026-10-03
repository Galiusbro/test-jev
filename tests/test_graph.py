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


def test_happy_path_validates_and_writes_report(repo: Path, tmp_path: Path) -> None:
    coder = [
        call("replace_in_file", path="app/main.py", old="'hi'", new="'hello'"),
        say("Changed the return value."),
    ]
    state, models = run(repo, tmp_path, planner_script(), coder)

    assert state["status"] == "validated"
    assert state["plan"].files_to_change == ["app/main.py"]
    assert state["changed_files"] == ["app/main.py"]
    assert state["tool_calls"] == ["replace_in_file"]
    assert (repo / "app" / "main.py").read_text().endswith("'hi'\n")  # original untouched

    run_dir = tmp_path / "run"
    assert "+    return 'hello'" in (run_dir / "changes.diff").read_text()
    report = json.loads((run_dir / "report.json").read_text())
    assert report["status"] == "validated"
    assert report["validation"][0]["exit_code"] == 0
    assert report["metrics"]["llm_calls"] == 2  # one factory call per tier in this fake
    assert report["plan"]["risk"] == "low"

    # planner only ever saw read-only tools; implementer got write tools
    assert models["strong"].bound_tools[0] == ["list_files", "read_file", "search"]
    assert "write_file" in models["coder"].bound_tools[0]
    # rules from AGENTS.md reach both agents
    assert "## Commands" in str(models["coder"].seen[0][0].content)


def test_broken_change_fails_validation(repo: Path, tmp_path: Path) -> None:
    coder = [
        call("replace_in_file", path="app/main.py", old="'hi'", new="'hey'"),
        say("done"),
    ]
    state, _ = run(repo, tmp_path, planner_script(), coder)
    assert state["status"] == "validation_failed"
    assert "AssertionError" in state["validation"][0].output


def test_no_changes(repo: Path, tmp_path: Path) -> None:
    state, _ = run(repo, tmp_path, planner_script(), [say("Nothing to do.")])
    assert state["status"] == "no_changes"


def test_step_budget_exhausted_is_incomplete(repo: Path, tmp_path: Path) -> None:
    coder = [
        call("replace_in_file", "c1", path="app/main.py", old="'hi'", new="'hello'"),
        call("read_file", "c2", path="app/main.py"),
    ]
    state, _ = run(repo, tmp_path, planner_script(), coder, implement_max_steps=2)
    assert state["status"] == "incomplete"
    assert state["implement_steps"] == 2


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
    state, _ = run(repo, tmp_path, planner_script(), coder)

    assert state["status"] == "incomplete"
    assert state["implement_steps"] == 1
    assert "implementation stopped" in state["error"]
    assert state["changed_files"] == ["app/main.py"]  # partial work kept for inspection
