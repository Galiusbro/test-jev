# jev-agent

> Controlled autonomous software engineering: from ticket to validated pull request.

LangGraph orchestrates the workflow, **Jev** (TypeSafe API) makes typed
decisions on its edges, free **NVIDIA API Catalog** models write the code, and
plain Python enforces policy. Full design: [PROJECT_IDEA.md](PROJECT_IDEA.md).

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
| `jev-agent ask "Question?" -s '<state>'` | One yes/no Jev decision — for tuning question wording |

## Development

```bash
uv run pytest && uv run ruff check . && uv run mypy src tests
cd demo-api && uv run pytest
```

Agent instructions: [AGENTS.md](AGENTS.md).

## Status

- [x] M1 — skeleton, Jev + NVIDIA clients, demo-api, CI
- [ ] M2 — linear happy path (ticket → context → plan → implement → validate)
- [ ] M3 — policy engine from AGENTS.md, capabilities, approval interrupts
- [ ] M4 — bounded repair, regression proof, review agent, PR
- [ ] M5 — Jev decision fabric + model router
- [ ] M6 — LangSmith tracing, evals, experiments
- [ ] M7 — `jev-agent init`, MCP server, demo script
