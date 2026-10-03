# AGENTS.md — jev-agent

Shared instructions for coding agents (Claude Code, Codex, Cursor, …) working on
this repository. Product context and roadmap: [PROJECT_IDEA.md](PROJECT_IDEA.md).

## What this repo is

A controlled ticket-to-PR agent. LangGraph owns the workflow, Jev (TypeSafe API)
makes typed decisions on graph edges, NVIDIA API Catalog models do the
generative work, and plain Python enforces policy.

```text
src/jev_agent/
  config.py       # Settings from env/.env; ModelTier -> model id
  llm/            # ResilientChatModel: streaming, fallback chain, call log
                  #   reasoning.py maps on/off/low to each model family's flags
  decisions/      # Jev: typed questions/answers, HTTP client, fake client
  cli.py          # `jev-agent` Typer app
demo-api/         # separate uv project — the repo the agent works ON
tests/            # unit tests; no network
```

## Ground rules

- **The model is not the security boundary.** Permissions are enforced in code
  (policy engine, tool layer), never by prompt wording or a Jev answer alone.
- Jev answers are judgments, not facts. Every Jev decision has a confidence
  threshold and a conservative fallback (more context, approval, or human).
  `JevError` is handled like a low-confidence answer.
- Keep math, counting, dates and control flow in code; ask Jev only for
  judgment calls (see Reactify "state and questions" guidance).
- Every LLM call goes through `llm.chat_model(tier, …)`: never call a single
  NVIDIA model directly — free-tier queues stall at random and the fallback
  chain is what keeps runs alive.
- Tests never hit the network: use `FakeJevClient` and `respx` for HTTP.
- `demo-api/` has its own `AGENTS.md`, venv and tooling. Do not import from it.

## Commands

- install: `uv sync`
- test: `uv run pytest`
- lint: `uv run ruff check . && uv run ruff format --check .`
- typecheck: `uv run mypy src tests`
- demo-api tests: `cd demo-api && uv run pytest`
- live smoke check (needs keys): `uv run jev-agent doctor`

## Conventions

- Python 3.12, `mypy --strict`, ruff (config in `pyproject.toml`).
- Pydantic models for anything crossing a boundary (LLM output, Jev I/O, config).
- New external integration = interface (Protocol) + real impl + fake for tests.
- Small, focused diffs. Record notable design choices in `docs/adr/`.

## Ask before

- Adding dependencies.
- Changing the Jev wire format in `decisions/types.py` (verify against the live API).
- Touching CI workflows or `.claude/settings.json`.

## Never

- Commit `.env` or print secret values.
- Weaken tests, lint or type settings to get green.

## Definition of done

All commands above pass, new behaviour has tests, docs updated if user-visible.
