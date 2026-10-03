"""Tickets: the task the agent works on.

A ticket file is Markdown: the first `# heading` is the title, the rest is the
body. GitHub Issues will map onto the same model later.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Ticket(BaseModel):
    id: str
    title: str
    body: str

    def as_text(self) -> str:
        return f"# {self.title}\n\n{self.body}".strip()


def load_ticket(path: Path) -> Ticket:
    lines = path.read_text().strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{path}: ticket must start with a '# Title' line")
    return Ticket(id=path.stem, title=lines[0][2:].strip(), body="\n".join(lines[1:]).strip())
