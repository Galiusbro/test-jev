from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from jev_agent.decisions import FakeJevClient
from jev_agent.decisions.fabric import Decisions
from jev_agent.mcp_server import build_server
from jev_agent.runner import Approvals

AGENTS = """# Repo

## Commands

- test: `echo tests-ran`

## Allowed

- Modify code under `app/`
- Add or modify tests under `tests/`

## Approval required

- Changes to `app/db.py`

## Forbidden

- Reading or writing `.env`
- Editing or deleting existing tests to make them pass
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A live git checkout (MCP serves it in place, baseline = HEAD)."""
    root = tmp_path / "repo"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "AGENTS.md").write_text(AGENTS)
    (root / ".env").write_text("SECRET=1\n")
    (root / "app" / "main.py").write_text("x = 1\n")
    (root / "app" / "db.py").write_text("SCHEMA = ''\n")
    (root / "tests" / "test_main.py").write_text("def test_x():\n    assert True\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@x"]
    subprocess.run([*git, "init", "-q"], cwd=root, check=True)
    subprocess.run([*git, "add", "-A"], cwd=root, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], cwd=root, check=True)
    return root


def call(server: Any, tool: str, /, **args: Any) -> str:
    result = asyncio.run(server.call_tool(tool, args))
    return "".join(getattr(part, "text", "") for part in result.content)


def test_exposes_governed_tools(repo: Path) -> None:
    server = build_server(repo)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "list_files",
        "read_file",
        "search",
        "write_file",
        "replace_in_file",
        "run_check",
        "project_rules",
        "triage_ticket",
    }


def test_policy_applies_to_mcp_clients(repo: Path) -> None:
    server = build_server(repo)
    assert ".env" not in call(server, "list_files")
    assert call(server, "read_file", path=".env").startswith("ERROR: denied by policy")
    assert call(server, "write_file", path="AGENTS.md", content="x").startswith("ERROR")
    denied = call(server, "replace_in_file", path="app/db.py", old="''", new="'users'")
    assert "approval was not given" in denied
    removed = call(server, "write_file", path="tests/test_main.py", content="X = 1\n")
    assert "would remove existing tests: test_x" in removed
    assert call(server, "replace_in_file", path="app/main.py", old="x = 1", new="x = 2") == (
        "edited app/main.py"
    )
    assert (repo / "app" / "main.py").read_text() == "x = 2\n"


def test_approvals_all_allows_approval_paths(repo: Path) -> None:
    server = build_server(repo, approvals=Approvals.ALL)
    assert call(server, "replace_in_file", path="app/db.py", old="''", new="'u'") == (
        "edited app/db.py"
    )


def test_run_check_and_rules(repo: Path) -> None:
    server = build_server(repo)
    assert "tests-ran" in call(server, "run_check", name="test")
    assert call(server, "run_check", name="rm").startswith("ERROR: unknown check")
    assert "## Forbidden" in call(server, "project_rules")


def test_triage_ticket_uses_jev_when_available(repo: Path) -> None:
    def respond(_state: Any, _questions: Any) -> dict[str, Any]:
        return {
            "kind": {"type": "choice", "value": "out_of_scope", "probability": 1, "confidence": 1},
            "actionable": {"type": "boolean", "probability": 0.1},
        }

    server = build_server(repo, decisions=Decisions(FakeJevClient(respond)))
    result = json.loads(call(server, "triage_ticket", title="Deploy", body="Ship to prod"))
    assert result["kind"] == "out_of_scope" and result["stop_reason"] and result["jev"]

    plain = json.loads(call(build_server(repo), "triage_ticket", title="t", body="b"))
    assert plain == {"kind": "feature", "actionable": True, "stop_reason": None, "jev": False}
