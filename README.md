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
| `jev-agent run tickets/001-….md [--repo demo-api] [--approvals ask\|all\|none] [--no-jev]` | Ticket → plan → implement → validate on a copy; writes `runs/<id>/report.json` + `changes.diff` |
| `jev-agent eval [--case ID] [--mode jev\|no-jev] [--repeats N]` | Run the eval cases; results to `evals/results/latest.jsonl` (resumable) |
| `jev-agent eval-report [results.jsonl]` | Summary tables (Markdown) |
| `jev-agent ask "Question?" -s '<state>'` | One yes/no Jev decision — for tuning question wording |

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
- [x] M4 — bounded repair, autofix, review agent (regression proof + PR still to do)
- [x] M5 — Jev decision fabric: triage, plan risk/complexity → approval + model choice, semantic write gate, failure diagnosis, review verification
- [x] M6 — LangSmith tracing, eval harness, Jev vs no-Jev comparison → [evaluation report](docs/evaluation-report.md)
- [ ] M7 — `jev-agent init`, MCP server, demo script
