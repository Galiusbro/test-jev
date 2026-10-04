"""MCP server: jev-agent's governed capabilities for any MCP client.

A developer's own coding agent (Claude Code, Cursor, …) can connect to this
server and work on a repository through the same tools jev-agent's agents
use — so the same guarantees hold for it:

- reads and writes are checked against the repo's AGENTS.md rules (forbidden
  files hidden, approval-required paths refused unless approvals are on,
  edits that remove existing tests denied, edits that break Python syntax
  rejected);
- `run_check` runs only the commands AGENTS.md declares, never a shell;
- `triage_ticket` asks Jev whether a ticket is actionable and in scope.

It serves the live checkout (no copy): the baseline for test protection is
HEAD.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from jev_agent.decisions.fabric import Decisions
from jev_agent.policy import Policy, parse_rules
from jev_agent.project import load_instructions
from jev_agent.runner import Approvals, make_approver
from jev_agent.tickets import Ticket
from jev_agent.tools import make_tools
from jev_agent.workspace import Workspace


def build_server(
    repo: Path,
    *,
    approvals: Approvals = Approvals.NONE,
    decisions: Decisions | None = None,
) -> MCPServer:
    ws = Workspace(repo, base_ref="HEAD")
    instructions = load_instructions(ws.root)
    policy = Policy(
        parse_rules(instructions.text),
        make_approver(approvals, log=lambda _m: None),
        ws.baseline,
    )
    server = MCPServer(
        name="jev-agent",
        instructions=(
            f"Governed access to the repository at {ws.root.name}. Call `project_rules` "
            "first. Writes are checked against those rules; denied calls return an "
            "ERROR explaining why. Use `run_check` to run the project's tests and linters."
        ),
    )

    for tool in make_tools(
        ws,
        writable=True,
        policy=policy,
        checks=instructions.commands,
        autofix=list(instructions.autofix.values()),
    ):
        func = getattr(tool, "func", None)
        if func is not None:
            server.tool(name=tool.name, description=tool.description)(func)

    @server.tool()
    def project_rules() -> str:
        """The repository's AGENTS.md: conventions, commands and what agents may change."""
        return instructions.text

    jev = decisions or Decisions(None)

    @server.tool()
    def triage_ticket(title: str, body: str) -> str:
        """Ask Jev whether a ticket is in scope and concrete enough to implement.

        Returns JSON with `kind`, `actionable` and `stop_reason` (null when the
        ticket can proceed). Without a Jev key every ticket proceeds.
        """
        result = jev.triage(Ticket(id="mcp", title=title, body=body), instructions.text)
        return json.dumps(
            {
                "kind": result.kind,
                "actionable": result.actionable,
                "stop_reason": result.stop_reason,
                "jev": jev.enabled,
            }
        )

    return server
