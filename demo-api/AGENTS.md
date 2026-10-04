# AGENTS.md — demo-api

Instructions for any coding agent working in this repository. The sections
**Allowed**, **Approval required** and **Forbidden** are machine-read by the
jev-agent policy engine — keep one rule per bullet, paths in backticks.

## Project

Small FastAPI service: user accounts and login. Python 3.12, SQLite, no ORM.

```text
src/demo_api/
  main.py      # app factory; wires services into app.state
  db.py        # SQLite connection + schema (the only place SQL DDL lives)
  security.py  # password hashing, tokens
  users.py     # UserService + /users routes
  auth.py      # AuthService + /login route
tests/         # pytest, FastAPI TestClient, in-memory DB
docs/api.md    # public API reference — update when behaviour changes
```

## Conventions

- Services hold logic; route functions only translate HTTP <-> service calls.
- New dependencies go through `app.state` in `create_app`, never module globals.
- Type hints everywhere; `mypy --strict` must pass.
- Every behaviour change ships with a test that fails without the change.
- Keep diffs minimal: no drive-by refactors or renames outside the task.
- Never trust client-supplied headers (e.g. `X-Forwarded-For`) for security
  decisions. Behind a reverse proxy the ASGI server resolves the client
  address (`uvicorn --proxy-headers --forwarded-allow-ips=<proxy>`); app code
  uses `request.client.host`.

## Commands

- test: `uv run pytest`
- lint: `uv run ruff check .`
- format: `uv run ruff format --check .`
- typecheck: `uv run mypy src tests`

## Autofix

Run automatically before validation; agents don't need to format by hand.

- format: `uv run ruff format .`
- lint-fix: `uv run ruff check --fix --quiet . || true`

## Allowed

- Modify application code under `src/demo_api/`
- Add or modify tests under `tests/`
- Update documentation under `docs/` and `README.md`

## Approval required

- Database schema changes in `src/demo_api/db.py`
- Changes to authentication or password handling in `src/demo_api/security.py`
- Public API changes: new, removed or renamed endpoints, fields or status codes
- Adding or upgrading dependencies in `pyproject.toml`
- Deleting any file

## Forbidden

- Editing or deleting existing tests to make them pass
- Disabling lint, type checks or tests
- Reading or writing `.env` or any secrets
- Network access other than the package index
- Modifying `AGENTS.md`

## Definition of done

1. All commands in **Commands** pass.
2. New behaviour is covered by tests that fail on the base commit.
3. `docs/api.md` reflects any API-visible change.
4. The diff touches only files the plan listed.
