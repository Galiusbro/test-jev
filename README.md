# jev-agent

> Controlled autonomous software engineering: from ticket to validated pull request.

LangGraph orchestrates the workflow, **Jev** (TypeSafe API) makes typed
decisions on its edges, free **NVIDIA API Catalog** models write the code, and
plain Python enforces policy. Full design: [PROJECT_IDEA.md](PROJECT_IDEA.md) ·
results: [evaluation report](docs/evaluation-report.md) · walkthrough: [demo script](docs/demo.md) · file map: [project map](docs/project-map.md) ·
example output: [PR #1](https://github.com/Galiusbro/test-jev/pull/1).

## Setup

```bash
uv sync
cp .env.example .env   # add NVIDIA_API_KEY and TYPESAFE_API_KEY
uv run jev-agent doctor
```

## Commands

| Command | What it does |
|---|---|
| `jev-agent doctor` | Checks keys; one live call per model tier (shows fallbacks) and to Jev |
| `jev-agent models [pattern]` | Lists NVIDIA models your key can use |
| `jev-agent bench [models…] -n 3` | Median latency + tool-call success per model (default: shortlist) |
| `jev-agent run tickets/001-….md [--repo demo-api] [--approvals ask\|all\|none] [--no-jev] [--open-pr]` | Ticket → plan → implement → validate on a copy; writes `runs/<id>/report.json` + `changes.diff` |
| `jev-agent eval [--case ID] [--mode jev\|no-jev] [--repeats N]` | Run the eval cases; results to `evals/results/latest.jsonl` (resumable) |
| `jev-agent eval-report [results.jsonl]` | Summary tables (Markdown) |
| `jev-agent init DIR` | Bootstrap a repo: AGENTS.md (detected commands + policy), CLAUDE.md, .mcp.json; never overwrites |
| `jev-agent mcp [--repo DIR] [--approvals none\|all]` | Serve the repo's governed tools over MCP (stdio) |
| `jev-agent ask "Question?" -s '<state>'` | One yes/no Jev decision — for tuning question wording |

## Use from any MCP client

`jev-agent mcp` serves a repository's governed tools over MCP, so your own
coding agent works under the same rules as jev-agent's agents: AGENTS.md
policy on every read and write, `.env` hidden, existing tests protected,
`run_check` limited to the declared commands, plus Jev-backed `triage_ticket`.

Claude Code (`.mcp.json` in the project you work on):

```json
{
  "mcpServers": {
    "jev-agent": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/Jev-agent",
               "jev-agent", "mcp", "--repo", "/path/to/target-repo"]
    }
  }
}
```

`--approvals none` (default) refuses approval-required writes; `--approvals all`
allows them.

## Development

```bash
uv run pytest && uv run ruff check . && uv run mypy src tests
cd demo-api && uv run pytest
```

Agent instructions: [AGENTS.md](AGENTS.md).

## Status

- [x] M1 — skeleton, Jev + NVIDIA clients, demo-api, CI
- [x] M2 — linear happy path (ticket → context → plan → implement → validate)
- [x] M3 — policy engine from AGENTS.md, scoped capabilities (incl. run_check), human approvals
- [x] M4 — bounded repair, autofix, review agent, regression proof, GitHub PR (`--open-pr`)
- [x] M5 — Jev decision fabric: triage, plan risk/complexity → approval + model choice, semantic write gate, failure diagnosis, review verification
- [x] M6 — LangSmith tracing, eval harness, Jev vs no-Jev comparison → [evaluation report](docs/evaluation-report.md)
- [x] M7 — `jev-agent init`, MCP server, [demo script](docs/demo.md)
