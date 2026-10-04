"""Open a GitHub pull request for an approved run.

The change lives in the run's workspace copy. To publish it without touching
the user's checkout, a temporary `git worktree` is created on a new branch,
the workspace diff is applied under the target repo's path inside the git
repository (e.g. `demo-api/`), committed, pushed, and `gh pr create` opens the
PR with a body built from the run report.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

GhRunner = Callable[[Sequence[str], Path], str]


class PullRequestError(RuntimeError):
    pass


def _git(cwd: Path, *args: str, input_text: str | None = None) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, input=input_text)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise PullRequestError(f"git {' '.join(args)}: {detail}")
    return proc.stdout.strip()


def run_gh(args: Sequence[str], cwd: Path) -> str:
    proc = subprocess.run(["gh", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PullRequestError(f"gh {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


_NUMBERED = re.compile(r"^\s*\d+[.)]\s+")


def pr_body(report: dict[str, Any], prefix: str = "") -> str:
    """Markdown PR description; `prefix` is the target's path inside the git repo."""
    base = f"{prefix.rstrip('/')}/" if prefix and prefix != "." else ""
    plan = report.get("plan") or {}
    review = report.get("review") or {}
    proof = report.get("regression_proof") or {}
    policy = report.get("policy") or {}
    jev = (report.get("jev") or {}).get("decisions") or []
    approvals = [a for a in policy.get("audit", []) if a["verdict"] == "approved"]
    lines = [
        "## Summary",
        "",
        plan.get("summary", "") or report.get("ticket", {}).get("title", ""),
        "",
        "## Ticket",
        "",
        f"**{report['ticket']['title']}**",
        "",
        report["ticket"]["body"],
        "",
        "## Changes",
        "",
        *[f"- `{base}{path}`" for path in report.get("changed_files", [])],
        "",
        "## Plan",
        "",
        # Models often number their steps already; don't double the numbering.
        *[f"{i}. {_NUMBERED.sub('', step)}" for i, step in enumerate(plan.get("steps", []), 1)],
        "",
        f"Risk: **{plan.get('risk', '?')}** · public API change: "
        f"**{plan.get('public_api_change')}** · DB migration: **{plan.get('db_migration')}**",
        "",
        "## Validation",
        "",
        *[
            f"- {'✅' if v['exit_code'] == 0 else '❌'} `{v['command']}`"
            for v in report.get("validation", [])
        ],
        f"- Regression proof: **{proof.get('status', 'n/a')}** — {proof.get('detail', '')}",
        f"- Repair attempts: {len(report.get('repairs', []))}",
        "",
        "## Review",
        "",
        review.get("summary", "No review recorded."),
        "",
        *[f"- [{f['severity']}] `{f['file']}`: {f['issue']}" for f in review.get("findings", [])],
        "",
        "## Approvals",
        "",
        *([f"- {a['action']} `{a['target']}`: {a['reason']}" for a in approvals] or ["- none"]),
        "",
    ]
    if jev:
        lines += [
            "## Jev decisions",
            "",
            *[
                f"- `{d['name']}` → {d['outcome']}"
                for d in jev
                if not d["name"].startswith("write_gate")
            ],
            f"- write gate: {sum(d['name'].startswith('write_gate') for d in jev)} edits screened",
            "",
        ]
    lines += ["---", "Opened by jev-agent from a validated, reviewed run."]
    return "\n".join(lines)


def open_pull_request(
    *,
    target_repo: Path,
    diff: str,
    report: dict[str, Any],
    gh: GhRunner = run_gh,
    remote: str = "origin",
) -> str:
    """Create branch + commit + push + PR. Returns the PR URL."""
    if not diff.strip():
        raise PullRequestError("nothing to publish: empty diff")
    repo_root = Path(_git(target_repo, "rev-parse", "--show-toplevel"))
    prefix = target_repo.resolve().relative_to(repo_root.resolve()).as_posix()
    base = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    ticket = report["ticket"]
    branch = f"jev/{ticket['id']}-{time.strftime('%Y%m%d-%H%M%S')}"

    with tempfile.TemporaryDirectory(prefix="jev-pr-") as tmp:
        worktree = Path(tmp) / "wt"
        _git(repo_root, "worktree", "add", "-q", "-b", branch, str(worktree), base)
        try:
            apply = ["apply", "--index"]
            if prefix != ".":
                apply.append(f"--directory={prefix}")
            _git(worktree, *apply, "-", input_text=diff if diff.endswith("\n") else diff + "\n")
            _git(worktree, "commit", "-q", "-m", f"{ticket['title']}\n\nTicket: {ticket['id']}")
            _git(worktree, "push", "-q", "-u", remote, branch)
        finally:
            _git(repo_root, "worktree", "remove", "--force", str(worktree))
    return gh(
        [
            "pr",
            "create",
            "--base",
            base,
            "--head",
            branch,
            "--title",
            ticket["title"],
            "--body",
            pr_body(report, prefix),
        ],
        repo_root,
    )
