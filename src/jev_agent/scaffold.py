"""`jev-agent init`: bootstrap a repository for governed AI-assisted work.

Generates the files the rest of jev-agent reads, tailored to the detected
stack, and never overwrites existing ones:

- AGENTS.md  — commands, autofix, Allowed / Approval required / Forbidden,
               definition of done (the policy engine's input)
- CLAUDE.md  — points Claude Code at AGENTS.md
- .mcp.json  — connects MCP clients to `jev-agent mcp` for this repo
"""

from __future__ import annotations

import json
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Stack:
    name: str
    commands: dict[str, str] = field(default_factory=dict)
    autofix: dict[str, str] = field(default_factory=dict)
    code_dirs: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    manifest: str | None = None


def detect_stack(repo: Path) -> Stack:
    if (repo / "pyproject.toml").exists():
        return _python(repo)
    if (repo / "package.json").exists():
        return _node(repo)
    return Stack("unknown", code_dirs=_existing(repo, ["src", "lib", "app"]))


def _existing(repo: Path, names: list[str]) -> list[str]:
    return [n for n in names if (repo / n).is_dir()]


def _python(repo: Path) -> Stack:
    text = (repo / "pyproject.toml").read_text()
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        data = {}
    run = "uv run " if (repo / "uv.lock").exists() else "python -m "
    tools = set(data.get("tool", {}))
    deps = " ".join([*data.get("project", {}).get("dependencies", []), *_dev_deps(data)]).lower()

    def uses(name: str) -> bool:
        return name in tools or name in deps

    commands: dict[str, str] = {}
    autofix: dict[str, str] = {}
    if uses("pytest") or (repo / "tests").is_dir():
        commands["test"] = f"{run}pytest"
    if uses("ruff"):
        commands["lint"] = f"{run}ruff check ."
        commands["format"] = f"{run}ruff format --check ."
        autofix["format"] = f"{run}ruff format ."
        autofix["lint-fix"] = f"{run}ruff check --fix --quiet . || true"
    if uses("mypy"):
        commands["typecheck"] = f"{run}mypy ."
    package = data.get("project", {}).get("name", "").replace("-", "_")
    code = _existing(repo, ["src", package] if package else ["src"]) or _existing(
        repo, ["app", "lib"]
    )
    return Stack(
        "python",
        commands,
        autofix,
        code_dirs=code,
        test_dirs=_existing(repo, ["tests", "test"]),
        manifest="pyproject.toml",
    )


def _dev_deps(data: dict[str, object]) -> list[str]:
    groups = data.get("dependency-groups", {})
    if not isinstance(groups, dict):
        return []
    return [str(d) for deps in groups.values() if isinstance(deps, list) for d in deps]


def _node(repo: Path) -> Stack:
    try:
        scripts = json.loads((repo / "package.json").read_text()).get("scripts", {})
    except json.JSONDecodeError:
        scripts = {}
    commands = {k: f"npm run {k}" for k in ("test", "lint", "typecheck") if k in scripts}
    autofix = {"format": "npm run format"} if "format" in scripts else {}
    return Stack(
        "node",
        commands,
        autofix,
        code_dirs=_existing(repo, ["src", "lib", "app"]),
        test_dirs=_existing(repo, ["tests", "test", "__tests__"]),
        manifest="package.json",
    )


def agents_md(repo: Path, stack: Stack) -> str:
    def bullets(items: dict[str, str]) -> str:
        return "\n".join(f"- {k}: `{v}`" for k, v in items.items())

    commands = bullets(stack.commands) or "- test: `echo 'TODO: add your test command'`"
    allowed = [f"- Modify application code under `{d}/`" for d in stack.code_dirs]
    allowed += [f"- Add or modify tests under `{d}/`" for d in stack.test_dirs]
    allowed += ["- Update documentation under `docs/` and `README.md`"]
    approval = []
    if stack.manifest:
        approval.append(f"- Adding or upgrading dependencies in `{stack.manifest}`")
    for d in _existing(repo, ["migrations", "alembic", "db/migrate"]):
        approval.append(f"- Database schema changes in `{d}/`")
    approval += [
        "- Changes to CI configuration in `.github/`",
        "- Public API changes: new, removed or renamed endpoints, fields or status codes",
        "- Deleting any file",
    ]
    parts = [
        f"# AGENTS.md — {repo.resolve().name}",
        "",
        "Instructions for any coding agent working in this repository. The sections",
        "**Allowed**, **Approval required** and **Forbidden** are enforced by jev-agent's",
        "policy engine — keep one rule per bullet, paths in backticks.",
        "",
        "## Project",
        "",
        "TODO: one paragraph on what this repository does and how it is structured.",
        "",
        "## Conventions",
        "",
        "- Keep diffs minimal: no drive-by refactors or renames outside the task.",
        "- Every behaviour change ships with a test that fails without the change.",
        "- Never trust client-supplied input for security decisions.",
        "",
        "## Commands",
        "",
        commands,
        "",
    ]
    if stack.autofix:
        parts += ["## Autofix", "", bullets(stack.autofix), ""]
    parts += [
        "## Allowed",
        "",
        *allowed,
        "",
        "## Approval required",
        "",
        *approval,
        "",
        "## Forbidden",
        "",
        "- Editing or deleting existing tests to make them pass",
        "- Disabling lint, type checks or tests",
        "- Reading or writing `.env` or any secrets",
        "- Modifying `AGENTS.md`",
        "",
        "## Definition of done",
        "",
        "1. All commands in **Commands** pass.",
        "2. New behaviour is covered by tests that fail on the base commit.",
        "3. Documentation reflects any user-visible change.",
        "",
    ]
    return "\n".join(parts)


CLAUDE_MD = """@AGENTS.md

## Claude Code specifics

- The rules in AGENTS.md are enforced when you work through the `jev-agent` MCP
  server (`.mcp.json`); prefer its tools for reading, editing and running checks.
"""


def mcp_json(repo: Path, jev_agent_dir: Path) -> str:
    uv = shutil.which("uv") or "uv"
    config = {
        "mcpServers": {
            "jev-agent": {
                "command": uv,
                "args": [
                    "run",
                    "--directory",
                    str(jev_agent_dir.resolve()),
                    "jev-agent",
                    "mcp",
                    "--repo",
                    str(repo.resolve()),
                ],
            }
        }
    }
    return json.dumps(config, indent=2) + "\n"


@dataclass
class InitResult:
    stack: Stack
    written: list[str]
    skipped: list[str]


def init_repo(repo: Path, jev_agent_dir: Path) -> InitResult:
    stack = detect_stack(repo)
    files = {
        "AGENTS.md": agents_md(repo, stack),
        "CLAUDE.md": CLAUDE_MD,
        ".mcp.json": mcp_json(repo, jev_agent_dir),
    }
    written, skipped = [], []
    for name, content in files.items():
        path = repo / name
        if path.exists():
            skipped.append(name)
            continue
        path.write_text(content)
        written.append(name)
    return InitResult(stack, written, skipped)
