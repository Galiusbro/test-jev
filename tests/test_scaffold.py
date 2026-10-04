from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from jev_agent import cli
from jev_agent.policy import parse_rules
from jev_agent.project import load_instructions
from jev_agent.scaffold import detect_stack, init_repo

PYPROJECT = """
[project]
name = "shop-api"
dependencies = ["fastapi"]

[dependency-groups]
dev = ["pytest>=8", "ruff", "mypy"]

[tool.ruff]
line-length = 100
"""


def python_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "shop"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "migrations").mkdir()
    (repo / "pyproject.toml").write_text(PYPROJECT)
    (repo / "uv.lock").write_text("")
    return repo


def test_detects_python_tooling(tmp_path: Path) -> None:
    stack = detect_stack(python_repo(tmp_path))
    assert stack.name == "python"
    assert stack.commands == {
        "test": "uv run pytest",
        "lint": "uv run ruff check .",
        "format": "uv run ruff format --check .",
        "typecheck": "uv run mypy .",
    }
    assert stack.autofix["format"] == "uv run ruff format ."
    assert (stack.code_dirs, stack.test_dirs) == (["src"], ["tests"])


def test_generated_agents_md_drives_the_policy_engine(tmp_path: Path) -> None:
    repo = python_repo(tmp_path)
    result = init_repo(repo, tmp_path)
    assert result.written == ["AGENTS.md", "CLAUDE.md", ".mcp.json"]

    instructions = load_instructions(repo)  # parses Commands / Autofix
    assert instructions.commands["test"] == "uv run pytest"
    assert "lint-fix" in instructions.autofix
    rules = parse_rules(instructions.text)
    allowed = [r for r in rules if r.kind == "allowed"]
    assert any(r.matches("src/shop/app.py") for r in allowed)
    approval = [r for r in rules if r.kind == "approval"]
    assert any(r.matches("migrations/001.sql") for r in approval)
    assert any(r.matches("pyproject.toml") for r in approval)
    forbidden = [r for r in rules if r.kind == "forbidden"]
    assert any(r.matches(".env") for r in forbidden)
    assert any(r.trigger == "existing_tests" for r in forbidden)
    assert "@AGENTS.md" in (repo / "CLAUDE.md").read_text()
    server = json.loads((repo / ".mcp.json").read_text())["mcpServers"]["jev-agent"]
    assert server["args"][-2:] == ["--repo", str(repo.resolve())]


def test_never_overwrites(tmp_path: Path) -> None:
    repo = python_repo(tmp_path)
    (repo / "AGENTS.md").write_text("# mine\n")
    result = init_repo(repo, tmp_path)
    assert result.skipped == ["AGENTS.md"]
    assert (repo / "AGENTS.md").read_text() == "# mine\n"


def test_node_and_unknown_stacks(tmp_path: Path) -> None:
    node = tmp_path / "web"
    (node / "src").mkdir(parents=True)
    (node / "package.json").write_text(
        json.dumps({"scripts": {"test": "vitest", "lint": "eslint ."}})
    )
    stack = detect_stack(node)
    assert stack.commands == {"test": "npm run test", "lint": "npm run lint"}
    assert stack.manifest == "package.json"

    bare = tmp_path / "bare"
    bare.mkdir()
    init_repo(bare, tmp_path)
    assert load_instructions(bare).commands["test"].startswith("echo 'TODO")


def test_cli_init(tmp_path: Path) -> None:
    repo = python_repo(tmp_path)
    result = CliRunner().invoke(cli.app, ["init", str(repo)])
    assert result.exit_code == 0, result.output
    assert "detected python" in result.output and "created AGENTS.md" in result.output
