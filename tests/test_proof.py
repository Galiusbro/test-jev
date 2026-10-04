from __future__ import annotations

import sys
from pathlib import Path

import pytest
from scripted_model import ScriptedChatModel, call, say

from jev_agent.config import ModelTier
from jev_agent.graph import RunConfig, build_graph
from jev_agent.llm import CallLog
from jev_agent.proof import is_test_file, prove_tests
from jev_agent.tickets import Ticket
from jev_agent.workspace import Workspace

PYTEST = f"{sys.executable} -m pytest -q -p no:cacheprovider"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "target"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "__init__.py").write_text("")
    (root / "app" / "main.py").write_text("def hello():\n    return 'hi'\n")
    (root / "tests" / "test_main.py").write_text(
        "from app.main import hello\n\n\ndef test_hello_exists():\n    assert hello()\n"
    )
    (root / "AGENTS.md").write_text(f"# T\n\n## Commands\n\n- test: `{PYTEST}`\n")
    return root


@pytest.fixture
def ws(repo: Path, tmp_path: Path) -> Workspace:
    return Workspace.create(repo, tmp_path / "run")


def fix_code(ws: Workspace) -> None:
    (ws.root / "app" / "main.py").write_text("def hello():\n    return 'hello'\n")


GOOD_TEST = (
    "from app.main import hello\n\n\ndef test_hello_word():\n    assert hello() == 'hello'\n"
)
WEAK_TEST = (
    "from app.main import hello\n\n\ndef test_hello_callable():\n    assert callable(hello)\n"
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("tests/test_a.py", True), ("a_test.py", True), ("src/app.py", False), ("tests/x.md", False)],
)
def test_is_test_file(path: str, expected: bool) -> None:
    assert is_test_file(path) is expected


def test_tests_that_fail_on_base_are_proven(ws: Workspace) -> None:
    fix_code(ws)
    (ws.root / "tests" / "test_word.py").write_text(GOOD_TEST)
    proof = prove_tests(ws, ws.changed_files(), PYTEST)
    assert proof.status == "proven" and proof.acceptable
    assert proof.tests == ("tests/test_word.py",)
    assert "return 'hello'" in (ws.root / "app" / "main.py").read_text()  # change restored


def test_tests_that_pass_on_base_are_not_proven(ws: Workspace) -> None:
    fix_code(ws)
    (ws.root / "tests" / "test_weak.py").write_text(WEAK_TEST)
    proof = prove_tests(ws, ws.changed_files(), PYTEST)
    assert proof.status == "not_proven" and not proof.acceptable
    assert "also pass on the original code" in proof.detail


def test_new_source_file_is_removed_for_the_proof_and_restored(ws: Workspace) -> None:
    (ws.root / "app" / "extra.py").write_text("VALUE = 1\n")
    (ws.root / "tests" / "test_extra.py").write_text(
        "from app.extra import VALUE\n\n\ndef test_value():\n    assert VALUE == 1\n"
    )
    proof = prove_tests(ws, ws.changed_files(), PYTEST)
    assert proof.status == "proven"  # ImportError on the original code
    assert (ws.root / "app" / "extra.py").exists()


def test_no_tests_skipped_and_missing_command(ws: Workspace) -> None:
    fix_code(ws)
    assert prove_tests(ws, ws.changed_files(), PYTEST).status == "no_tests"
    assert prove_tests(ws, ["docs/api.md"], PYTEST).status == "skipped"
    assert prove_tests(ws, ["app/main.py", "tests/test_a.py"], None).status == "skipped"


def test_no_tests_collected_is_not_proof(ws: Workspace) -> None:
    fix_code(ws)
    (ws.root / "tests" / "test_empty.py").write_text("X = 1\n")
    proof = prove_tests(ws, ws.changed_files(), PYTEST)
    assert proof.status == "not_proven" and proof.detail == "no tests were collected"


def test_graph_sends_weak_tests_back_to_repair(repo: Path, tmp_path: Path) -> None:
    plan = call(
        "Plan",
        summary="Return 'hello'.",
        files_to_change=["app/main.py"],
        files_to_create=["tests/test_word.py"],
        steps=["edit"],
        tests=["assert the word"],
        risk="low",
        public_api_change=False,
        db_migration=False,
    )
    strong = [say("hello() is in app/main.py"), plan, call("Review", summary="ok", findings=[])]
    coder = [
        call("replace_in_file", "e1", path="app/main.py", old="'hi'", new="'hello'"),
        call("write_file", "w1", path="tests/test_word.py", content=WEAK_TEST),
        say("done"),
        call("write_file", "w2", path="tests/test_word.py", content=GOOD_TEST),
        say("tests now check the value"),
    ]
    models = {
        "strong": ScriptedChatModel(replies=strong),
        "coder": ScriptedChatModel(replies=coder),
    }

    def factory(tier: ModelTier, _log: CallLog) -> ScriptedChatModel:
        return models[tier.value]

    cfg = RunConfig(repo=repo, run_dir=tmp_path / "run", models=factory, setup_command=None)
    ticket = Ticket(id="t", title="Say hello", body="hello() must return 'hello'.")
    state = build_graph(cfg).invoke({"ticket": ticket, "started_at": 0.0})

    assert state["status"] == "approved"
    assert state["repairs"][0]["reason"] == "regression_proof"
    assert "also pass on the original code" in str(models["coder"].seen[3][1].content)
    assert state["proof"].status == "proven"
