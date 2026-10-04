from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from jev_agent.pr import PullRequestError, open_pull_request, pr_body


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@x", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, Path]:
    """A git repo whose target app lives in demo/, with a bare remote."""
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    root = tmp_path / "project"
    (root / "demo" / "app").mkdir(parents=True)
    (root / "demo" / "app" / "main.py").write_text("def hello():\n    return 'hi'\n")
    git(root.parent, "init", "-q", "-b", "main", str(root))
    git(root, "config", "user.name", "t")
    git(root, "config", "user.email", "t@x")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-q", "origin", "main")
    return root, remote


DIFF = """diff --git a/app/main.py b/app/main.py
--- a/app/main.py
+++ b/app/main.py
@@ -1,2 +1,2 @@
 def hello():
-    return 'hi'
+    return 'hello'
"""

REPORT: dict[str, Any] = {
    "ticket": {"id": "007-hello", "title": "Say hello properly", "body": "Return 'hello'."},
    "plan": {
        "summary": "Change the return value.",
        "steps": ["edit"],
        "risk": "low",
        "public_api_change": False,
        "db_migration": False,
    },
    "changed_files": ["app/main.py"],
    "validation": [{"command": "pytest", "exit_code": 0}],
    "regression_proof": {"status": "proven", "detail": "fails on the original code"},
    "repairs": [],
    "review": {
        "summary": "Looks right.",
        "findings": [{"severity": "minor", "file": "app/main.py", "issue": "Docstring missing."}],
    },
    "policy": {
        "audit": [
            {
                "action": "plan",
                "target": "plan",
                "verdict": "approved",
                "reason": "public api change",
            }
        ]
    },
    "jev": {
        "decisions": [
            {"name": "triage", "outcome": "proceed (bug)"},
            {"name": "write_gate:app/main.py", "outcome": "allow"},
        ]
    },
}


def test_opens_pr_from_a_worktree_without_touching_the_checkout(
    project: tuple[Path, Path],
) -> None:
    root, remote = project
    calls: list[list[str]] = []

    def fake_gh(args: Sequence[str], cwd: Path) -> str:
        calls.append(list(args))
        assert cwd == root
        return "https://github.com/acme/demo/pull/1"

    url = open_pull_request(target_repo=root / "demo", diff=DIFF, report=REPORT, gh=fake_gh)

    assert url.endswith("/pull/1")
    branch = calls[0][calls[0].index("--head") + 1]
    assert branch.startswith("jev/007-hello-")
    assert calls[0][calls[0].index("--base") + 1] == "main"
    # the branch reached the remote with the change under demo/
    assert "return 'hello'" in git(remote, "show", f"{branch}:demo/app/main.py")
    # the user's checkout is untouched and no worktree is left behind
    assert "return 'hi'" in (root / "demo" / "app" / "main.py").read_text()
    assert git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert len(git(root, "worktree", "list").splitlines()) == 1


def test_empty_diff_is_refused(project: tuple[Path, Path]) -> None:
    root, _ = project
    with pytest.raises(PullRequestError, match="empty diff"):
        open_pull_request(target_repo=root / "demo", diff="", report=REPORT, gh=lambda a, c: "")


def test_bad_diff_cleans_up_the_worktree(project: tuple[Path, Path]) -> None:
    root, _ = project
    broken = DIFF.replace("return 'hi'", "return 'nope'")
    with pytest.raises(PullRequestError, match="git apply"):
        open_pull_request(target_repo=root / "demo", diff=broken, report=REPORT, gh=lambda a, c: "")
    assert len(git(root, "worktree", "list").splitlines()) == 1


def test_pr_body_contains_the_evidence() -> None:
    body = pr_body(REPORT)
    for expected in (
        "## Summary",
        "Change the return value.",
        "- `app/main.py`",
        "✅ `pytest`",
        "Regression proof: **proven**",
        "[minor] `app/main.py`: Docstring missing.",
        "plan `plan`: public api change",
        "`triage` → proceed (bug)",
        "write gate: 1 edits screened",
    ):
        assert expected in body, expected


def test_pr_body_prefixes_paths_and_does_not_double_number() -> None:
    body = pr_body(REPORT, "demo")
    assert "- `demo/app/main.py`" in body
    assert "1. Edit main.py" in body and "1. 1." not in body
    assert "2. Run tests" in body
    assert "- `app/main.py`" in pr_body(REPORT, ".")
