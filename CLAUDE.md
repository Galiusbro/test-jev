@AGENTS.md

## Claude Code specifics

- Hooks in `.claude/settings.json` lint every edited Python file (report only, no
  auto-fix: fixes on half-finished edits can corrupt code) and block
  edits to `.env*` files. If a hook fails, fix the cause; don't bypass it.
- Prefer `uv run …` for every Python command; never `pip install`.
