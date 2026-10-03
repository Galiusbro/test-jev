"""Capabilities the agents get — scoped to one workspace.

Agents never get a shell. Read tools are always available; write tools only
for the implementer. Every path goes through `Workspace.resolve`, so nothing
outside the working copy can be read or written.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool

from jev_agent.workspace import Workspace, WorkspaceError

MAX_READ_CHARS = 20_000
MAX_SEARCH_HITS = 50


def make_tools(ws: Workspace, *, writable: bool) -> list[BaseTool]:
    def list_files() -> str:
        """List every file in the repository (paths relative to the repo root)."""
        return "\n".join(ws.files())

    def read_file(path: str) -> str:
        """Read a text file. `path` is relative to the repo root."""
        try:
            target = ws.resolve(path)
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
        for rel in ws.files():
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
            text = target.read_text()
        except (WorkspaceError, OSError) as exc:
            return f"ERROR: {exc}"
        count = text.count(old)
        if count != 1:
            return f"ERROR: `old` found {count} times in {path}; it must match exactly once"
        target.write_text(text.replace(old, new, 1))
        return f"edited {path}"

    funcs: list[Callable[..., str]] = [list_files, read_file, search]
    if writable:
        funcs += [write_file, replace_in_file]
    return [StructuredTool.from_function(f) for f in funcs]
