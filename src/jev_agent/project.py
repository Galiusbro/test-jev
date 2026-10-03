"""Project instructions read from the target repo's AGENTS.md.

The text is agent context, `## Commands` is the validation suite and the
optional `## Autofix` section lists deterministic fixers (formatters, lint
autofix) run before every validation so models don't spend tokens on style.
The policy sections (Allowed / Approval required / Forbidden) become
enforceable rules in M3.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

_COMMAND = re.compile(r"^-\s*([\w-]+):\s*`([^`]+)`\s*$")


class ProjectInstructions(BaseModel):
    text: str
    commands: dict[str, str]  # name -> shell command, in file order
    autofix: dict[str, str] = {}


def _section(text: str, heading: str) -> list[str]:
    lines, inside = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            inside = line[3:].strip().lower() == heading.lower()
            continue
        if inside:
            lines.append(line)
    return lines


def load_instructions(repo: Path) -> ProjectInstructions:
    path = repo / "AGENTS.md"
    if not path.exists():
        raise FileNotFoundError(f"{repo} has no AGENTS.md — the agent needs project rules")
    text = path.read_text()
    commands = _commands(text, "Commands")
    if not commands:
        raise ValueError(f"{path}: no '- name: `command`' entries under '## Commands'")
    return ProjectInstructions(text=text, commands=commands, autofix=_commands(text, "Autofix"))


def _commands(text: str, heading: str) -> dict[str, str]:
    commands = {}
    for line in _section(text, heading):
        match = _COMMAND.match(line.strip())
        if match:
            commands[match.group(1)] = match.group(2)
    return commands
