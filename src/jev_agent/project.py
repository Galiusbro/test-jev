"""Project instructions read from the target repo's AGENTS.md.

M2 uses the text as agent context and the `## Commands` section as the
validation suite. The policy sections (Allowed / Approval required / Forbidden)
become enforceable rules in M3.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

_COMMAND = re.compile(r"^-\s*([\w-]+):\s*`([^`]+)`\s*$")


class ProjectInstructions(BaseModel):
    text: str
    commands: dict[str, str]  # name -> shell command, in file order


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
    commands = {}
    for line in _section(text, "Commands"):
        match = _COMMAND.match(line.strip())
        if match:
            commands[match.group(1)] = match.group(2)
    if not commands:
        raise ValueError(f"{path}: no '- name: `command`' entries under '## Commands'")
    return ProjectInstructions(text=text, commands=commands)
