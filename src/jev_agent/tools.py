"""Capabilities the agents get — scoped to one workspace.

Agents never get a shell. Read tools are always available; write tools only
for the implementer. Every path goes through `Workspace.resolve`, so nothing
outside the working copy can be read or written, and edits that would break
Python syntax are rejected before they touch the file.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from jev_agent.policy import Policy
from jev_agent.workspace import Workspace, WorkspaceError

MAX_READ_CHARS = 20_000
MAX_SEARCH_HITS = 50
# Test selectors only: paths, `::` ids, parametrize brackets. No spaces or shell syntax.
_TARGET = re.compile(r"^[\w./\-]+(::[\w\-\[\].]+)*$")


def make_tools(
    ws: Workspace,
    *,
    writable: bool,
    policy: Policy | None = None,
    checks: Mapping[str, str] | None = None,
    autofix: Sequence[str] = (),
) -> list[BaseTool]:
    """Tools for one workspace.

    With a `policy`, every read and write is checked. With `checks` (the
    project's declared commands), writers also get `run_check` — a way to
    verify their own work without getting a shell.
    """

    def visible() -> list[str]:
        return [f for f in ws.files() if policy is None or policy.readable(f)]

    def rel(target: Path) -> str:
        # Policy rules see the normalized path, so `./.env` or `app/../.env`
        # cannot slip past a rule written for `.env`.
        return target.relative_to(ws.root).as_posix()

    def list_files() -> str:
        """List every file in the repository (paths relative to the repo root)."""
        return "\n".join(visible())

    def read_file(path: str) -> str:
        """Read a text file. `path` is relative to the repo root."""
        try:
            target = ws.resolve(path)
            if policy and (denied := policy.check_read(rel(target))):
                return denied
            text = target.read_text()
        except (WorkspaceError, OSError, UnicodeDecodeError) as exc:
            return f"ERROR: {exc}"
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + "\n…(truncated)…"
        return text

    def search(pattern: str) -> str:
        """Search all files for a regular expression. Returns `path:line: text` hits."""
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"ERROR: invalid regex: {exc}"
        hits: list[str] = []
        for rel in visible():
            try:
                lines = ws.resolve(rel).read_text().splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for number, line in enumerate(lines, 1):
                if regex.search(line):
                    hits.append(f"{rel}:{number}: {line.strip()}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return "\n".join([*hits, "…(more hits truncated)…"])
        return "\n".join(hits) or "no matches"

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file with the full `content`."""
        try:
            target = ws.resolve(path)
        except WorkspaceError as exc:
            return f"ERROR: {exc}"
        if problem := _syntax_problem(path, content):
            return problem
        if policy and (denied := policy.check_write(rel(target), content)):
            return denied
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"wrote {path} ({len(content)} chars)"

    def replace_in_file(path: str, old: str, new: str) -> str:
        """Replace one exact occurrence of `old` with `new` in a file.

        `old` must appear exactly once; include enough surrounding lines to make
        it unique.
        """
        try:
            target = ws.resolve(path)
            if policy and (denied := policy.check_read(rel(target))):
                return denied
            text = target.read_text()
        except (WorkspaceError, OSError) as exc:
            return f"ERROR: {exc}"
        count = text.count(old)
        if count != 1:
            return f"ERROR: `old` found {count} times in {path}; it must match exactly once"
        updated = text.replace(old, new, 1)
        if problem := _syntax_problem(path, updated):
            return problem
        if policy and (denied := policy.check_write(rel(target), updated)):
            return denied
        target.write_text(updated)
        return f"edited {path}"

    def run_check(name: str, target: str = "") -> str:
        """Run one of the project's declared checks and return its output.

        `name` is one of the check names listed in the tool description.
        `target` optionally narrows a test run to a file or test id, e.g.
        `tests/test_api.py` or `tests/test_api.py::test_login`.
        """
        assert checks is not None
        command = checks.get(name)
        if command is None:
            return f"ERROR: unknown check {name!r}; available: {sorted(checks)}"
        if target:
            if not _TARGET.match(target):
                return f"ERROR: invalid target {target!r}"
            try:
                ws.resolve(target.split("::", 1)[0])
            except WorkspaceError as exc:
                return f"ERROR: {exc}"
            command = f"{command} {target}"
        for fix in autofix:
            ws.run(fix)
        result = ws.run(command, max_chars=4000)
        verdict = "PASSED" if result.ok else f"FAILED (exit {result.exit_code})"
        return f"$ {command}\n{verdict}\n{result.output}"

    funcs: list[Callable[..., str]] = [list_files, read_file, search]
    if writable:
        funcs += [write_file, replace_in_file]
    tools: list[BaseTool] = [StructuredTool.from_function(f) for f in funcs]
    if writable and checks:
        tools.append(
            StructuredTool.from_function(
                run_check,
                description=(run_check.__doc__ or "") + f"\nAvailable checks: {sorted(checks)}.",
            )
        )
    return tools


def _syntax_problem(path: str, content: str) -> str | None:
    """Reject edits that would leave a Python file unparseable."""
    if not path.endswith(".py"):
        return None
    try:
        compile(content, path, "exec", dont_inherit=True)
    except SyntaxError as exc:
        return (
            f"ERROR: edit rejected — {path} would not parse: {exc.msg} at line {exc.lineno}. "
            "The file is unchanged; check indentation and brackets."
        )
    return None
