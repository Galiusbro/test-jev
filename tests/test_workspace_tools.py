from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from jev_agent.project import load_instructions
from jev_agent.tickets import load_ticket
from jev_agent.tools import make_tools
from jev_agent.workspace import Workspace, WorkspaceError


@pytest.fixture
def source(tmp_path: Path) -> Path:
    repo = tmp_path / "src-repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "main.py").write_text("def hello():\n    return 'hi'\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "junk").write_text("x")
    (repo / "AGENTS.md").write_text(
        "# Rules\n\n## Commands\n\n- test: `python -c 'print(1)'`\n- lint: `true`\n\n"
        "## Allowed\n\n- anything: `nope`\n"
    )
    return repo


@pytest.fixture
def ws(source: Path, tmp_path: Path) -> Workspace:
    return Workspace.create(source, tmp_path / "run")


def tool(ws: Workspace, name: str, *, writable: bool = True) -> BaseTool:
    return {t.name: t for t in make_tools(ws, writable=writable)}[name]


def test_copy_is_isolated_and_skips_venv(ws: Workspace, source: Path) -> None:
    assert ws.files() == ["AGENTS.md", "app/main.py"]
    (ws.root / "app" / "main.py").write_text("changed")
    assert (source / "app" / "main.py").read_text().startswith("def hello")


def test_build_artifacts_are_not_changes(ws: Workspace) -> None:
    (ws.root / "app" / "__pycache__").mkdir()
    (ws.root / "app" / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"x")
    (ws.root / ".venv").mkdir()
    (ws.root / ".venv" / "pyvenv.cfg").write_text("x")
    (ws.root / ".pytest_cache").mkdir()
    assert ws.changed_files() == [] and ws.diff() == ""


def test_diff_and_changed_files(ws: Workspace) -> None:
    assert ws.diff() == "" and ws.changed_files() == []
    (ws.root / "app" / "main.py").write_text("def hello():\n    return 'hey'\n")
    (ws.root / "app" / "new.py").write_text("x = 1\n")
    assert ws.changed_files() == ["app/main.py", "app/new.py"]
    assert "+    return 'hey'" in ws.diff()


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", ".git/config"])
def test_resolve_rejects_escapes(ws: Workspace, path: str) -> None:
    with pytest.raises(WorkspaceError):
        ws.resolve(path)


def test_create_refuses_existing_run(source: Path, tmp_path: Path) -> None:
    Workspace.create(source, tmp_path / "run")
    with pytest.raises(WorkspaceError):
        Workspace.create(source, tmp_path / "run")


def test_run_command_captures_output_and_exit(ws: Workspace) -> None:
    ok = ws.run("echo hello")
    assert ok.ok and ok.output == "hello"
    bad = ws.run("echo boom >&2; exit 3")
    assert bad.exit_code == 3 and "boom" in bad.output
    long = ws.run("python -c \"print('x' * 9000)\"", max_chars=100)
    assert long.output.startswith("…(truncated)…") and len(long.output) < 200
    slow = ws.run("sleep 2", timeout_s=0.1)
    assert slow.exit_code == 124


def test_run_does_not_leak_our_virtualenv(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/somewhere/.venv")
    assert ws.run('echo "[$VIRTUAL_ENV]"').output == "[]"


def test_read_only_tools_have_no_write(ws: Workspace) -> None:
    names = [t.name for t in make_tools(ws, writable=False)]
    assert names == ["list_files", "read_file", "search"]


def test_read_search_and_list(ws: Workspace) -> None:
    assert tool(ws, "list_files").invoke({}) == "AGENTS.md\napp/main.py"
    assert "return 'hi'" in tool(ws, "read_file").invoke({"path": "app/main.py"})
    assert tool(ws, "read_file").invoke({"path": "../x"}).startswith("ERROR")
    assert tool(ws, "search").invoke({"pattern": r"def \w+"}) == "app/main.py:1: def hello():"
    assert tool(ws, "search").invoke({"pattern": "zzz"}) == "no matches"
    assert tool(ws, "search").invoke({"pattern": "("}).startswith("ERROR")


def test_write_and_replace(ws: Workspace) -> None:
    write, replace = tool(ws, "write_file"), tool(ws, "replace_in_file")
    assert write.invoke({"path": "pkg/new.py", "content": "a = 1\na = 1\n"}).startswith("wrote")
    assert "2 times" in replace.invoke({"path": "pkg/new.py", "old": "a = 1", "new": "b"})
    assert (
        replace.invoke({"path": "app/main.py", "old": "'hi'", "new": "'hello'"})
        == "edited app/main.py"
    )
    assert "'hello'" in (ws.root / "app" / "main.py").read_text()
    assert write.invoke({"path": "../evil.py", "content": "x"}).startswith("ERROR")


def test_load_instructions_commands_in_order(ws: Workspace) -> None:
    instructions = load_instructions(ws.root)
    assert instructions.commands == {"test": "python -c 'print(1)'", "lint": "true"}
    assert "# Rules" in instructions.text


def test_load_instructions_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_instructions(tmp_path)
    (tmp_path / "AGENTS.md").write_text("# no commands\n")
    with pytest.raises(ValueError, match="Commands"):
        load_instructions(tmp_path)


def test_demo_api_instructions_parse() -> None:
    instructions = load_instructions(Path(__file__).parents[1] / "demo-api")
    assert list(instructions.commands) == ["test", "lint", "format", "typecheck"]
    assert list(instructions.autofix) == ["format", "lint-fix"]


def test_load_ticket(tmp_path: Path) -> None:
    path = tmp_path / "042-thing.md"
    path.write_text("# Do the thing\n\nDetails here.\n")
    ticket = load_ticket(path)
    assert (ticket.id, ticket.title, ticket.body) == ("042-thing", "Do the thing", "Details here.")
    assert ticket.as_text() == "# Do the thing\n\nDetails here."
    path.write_text("no heading")
    with pytest.raises(ValueError):
        load_ticket(path)


def test_edits_that_break_python_syntax_are_rejected(ws: Workspace) -> None:
    write, replace = tool(ws, "write_file"), tool(ws, "replace_in_file")
    before = (ws.root / "app" / "main.py").read_text()

    result = replace.invoke(
        {"path": "app/main.py", "old": "    return 'hi'", "new": "  return 'hi'\n    x"}
    )
    assert result.startswith("ERROR: edit rejected") and "line" in result
    assert (ws.root / "app" / "main.py").read_text() == before

    assert write.invoke({"path": "app/new.py", "content": "def f(:\n"}).startswith("ERROR")
    assert not (ws.root / "app" / "new.py").exists()
    # non-Python files are not syntax-checked
    assert write.invoke({"path": "notes.md", "content": "def f(:"}).startswith("wrote")
