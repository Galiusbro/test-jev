# Jev Agent — Controlled Ticket-to-PR Platform

> **Controlled autonomous software engineering: from ticket to validated pull request.**

An AI-native engineering system that takes a software ticket (GitHub Issue), gathers repository context, plans the change, implements it, validates it with deterministic checks, repairs failures within a bounded budget, gets an independent AI review, and opens a Pull Request — while operating under version-controlled project rules, capability-based permissions and human approval gates.

The project also **demonstrates AI-native development itself**: the repo is built *with* coding agents, using shared instructions, reusable skills, hooks and CI — the same practices the platform automates.

---

## 1. Why this project (requirements coverage)

Every item from the Akvelon vacancy and the 10 screening questions maps to a concrete, demo-able feature.

### Vacancy requirements & responsibilities

| Requirement | Where it lives in the project |
|---|---|
| Strong fundamentals: architecture, testing, debugging, code quality | Typed Python 3.12, layered architecture, >85% test coverage, ruff + mypy --strict, ADRs in `docs/adr/` |
| Practical experience with AI coding agents | Repo is developed with Claude Code / Codex using `AGENTS.md`, `CLAUDE.md`, skills and hooks (see §6) |
| SDLC: CI/CD, automated testing, release | GitHub Actions: lint → typecheck → tests → eval suite → tagged release + Docker image |
| Assess & verify AI-generated code | Validation pipeline + "test must fail before fix" check + independent review agent (§3.5–3.7) |
| Build reusable agent workflows, instructions, skills, tools | `workflows/` (bug-fix, feature, review), `.claude/skills/`, MCP server with typed tools |
| Improve project docs/context for agents | `AGENTS.md` spec + context builder (repo map, targeted retrieval, Jev relevance filter) |
| Connect agents to dev tools & systems | MCP server: GitHub, repo, test runner, issue tracker; GitHub App for PRs |
| Automated testing, static analysis, verification | pytest, ruff, mypy, bandit/semgrep, API contract tests inside the validation node |
| Evaluate new AI tools for other teams | LangSmith experiments: LLM router vs Jev router, model A vs B, graph v1 vs v2 — with a written report |
| *Nice:* introducing AI practices in a team | Installable **workflow template** + `jev-agent init` that bootstraps any repo with AGENTS.md, hooks, CI checks |

### The 10 screening questions → proof points

| # | Question topic | Demo artifact |
|---|---|---|
| 1 | AGENTS.md / CLAUDE.md | Demo target repo's `AGENTS.md` is parsed into a machine-readable policy (allowed / approval / forbidden) |
| 2 | Integrations (GitHub, MCP, DBs) | Custom MCP server exposing scoped read/write capabilities |
| 3 | Reusable skills/commands/hooks | `bug-fix` workflow + Claude Code skills/hooks shipped in the template |
| 4 | Verifying AI code | Deterministic validation → bounded repair → independent review |
| 5 | Access controls | Capability-based tools + policy engine; "the model is not the security boundary" |
| 6 | Team adoption | `jev-agent init` template; developers use a workflow, not a blank chat |
| 7 | Others using it | Config lives in the repo; second demo repo onboarded with zero personal setup |
| 8 | Repeatable workflow | ticket → context → plan → approval → code → test → validate → review → PR |
| 9 | Measuring impact | Metrics dashboard: success rate, first-pass validation, retries, rework, cost/task |
| 10 | Models / routing / cost | Jev-driven model router over OpenRouter; per-run token & cost accounting |

---

## 2. Architecture

**Division of responsibilities:**

| Layer | Owns |
|---|---|
| **LangGraph** | The workflow: state machine, checkpoints, interrupts for human approval |
| **Jev** (TypeSafe decision model) | Fast, typed semantic decisions on graph edges: `Choice`, `Score`, `Noul` |
| **LangChain** | Worker agents (`create_agent`), tools, structured outputs, middleware |
| **MCP** | Exposes capabilities (repo, GitHub, tests, issues) with scoped access |
| **Policy engine (plain Python)** | Enforces permissions — deterministic, auditable, never delegated to a model |
| **LangSmith** | Tracing, datasets, evals (incl. Jev-as-a-Judge), experiments |
| **OpenRouter** | Model access; router picks cheap vs strong models per step |

```text
LLM    = thinks / writes / codes
Jev    = decides where to go next
Code   = enforces policy
Graph  = orchestrates
Smith  = observes and evaluates
```

### Workflow graph

```text
START
  ↓
ingest_ticket
  ↓
JEV Choice: task_type ── bug | feature | docs | security | out_of_scope → human
  ↓
build_context  (repo map → search → JEV Score relevance filter → top-k files)
  ↓
JEV Noul: context_sufficient? ── no → retrieve_more (max 2)
  ↓
planner  (structured Plan: files, changes, risk, api_change, migration)
  ↓
policy_check (deterministic, from AGENTS.md)  +  JEV Score: risk
  ├── forbidden            → STOP (rejected)
  ├── approval / high risk → interrupt: human_approval
  └── allowed
  ↓
implement  (coding agent; every tool call passes Jev risk gate + policy)
  ↓
validate  (pytest, ruff, mypy, security scan, test-fails-before-fix check)
  ↓
JEV Choice: failure_type ── pass | code_error | test_error | env_error | needs_human
  ├── code/test error & attempts < 3 → repair → validate
  ├── env_error / attempts ≥ 3       → NEEDS_HUMAN
  ↓
review_agent  (different model + prompt; diff-only view)
  ↓
JEV Noul: task_complete? ── no → repair (shares retry budget)
  ↓
open_pull_request  (summary, changes, tests, validation, risk, review findings, approvals)
  ↓
END
```

### Jev decision points (the "decision fabric")

| Primitive | Question | Used by |
|---|---|---|
| `Choice(task_type)` | What kind of ticket is this? | Router edge |
| `Choice(model_tier)` | Which model fits this step? | Model router middleware |
| `Score(relevance)` | Is this chunk relevant to the task? | Context filter |
| `Noul(context_sufficient)` | Do we know enough to plan? | Retrieval loop |
| `Score(risk)` | How risky is the plan / this tool call? | Policy + Auto-mode gate |
| `Choice(failure_type)` | Why did validation fail? | Repair routing |
| `Noul(task_complete)` | Does the diff satisfy the ticket? | Termination |

Low confidence on any decision → fall back to the conservative branch (more context, approval, or human).

### Tool-call path

```text
agent proposes tool call
  → schema validation
  → Jev risk gate (AutoModeMiddleware)
  → policy engine (allow / approve / deny)
  → MCP tool executes in sandbox
  → audit log
```

---

## 3. Core features

1. **Ticket ingestion** — GitHub Issue (or local YAML ticket for offline demos).
2. **AGENTS.md → policy** — parse sections into `Allowed`, `ApprovalRequired`, `Forbidden` rules plus test commands and definition of done.
3. **Context builder** — repository map, ripgrep/AST search, git history of touched files, Jev relevance filtering. Progressive disclosure: never the whole repo.
4. **Structured planning** — Pydantic `Plan` model; plan is shown to the human before code changes for risky tasks.
5. **Capability-based execution** — the agent gets `repo.read_file`, `repo.write_file(path in allowed_paths)`, `tests.run`, `lint.run`, `git.diff` — no raw shell. Runs in a Docker sandbox on a throwaway branch. Secrets never enter prompts.
6. **Deterministic validation** — pytest, ruff, mypy, bandit; plus **regression-test proof**: new tests must fail on the base commit and pass on the fix.
7. **Bounded self-repair** — `MAX_REPAIR_ATTEMPTS = 3`, real error output fed back; then `NEEDS_HUMAN`.
8. **Independent review** — separate model, diff-only input, checklist: regressions, missing tests, security, scope creep, architecture violations.
9. **PR generation** — GitHub App opens a PR with a structured body and a link to the LangSmith trace.
10. **Human-in-the-loop** — LangGraph interrupts; approve/reject via CLI or a minimal web UI.

---

## 4. Observability, evals & metrics

**Per run:** duration, LLM calls, tool calls, tokens in/out, cost, Jev decisions with confidence, repair attempts, files changed, approvals, outcome. Full trace in LangSmith; append-only JSONL audit log locally.

**Eval dataset** (`evals/tasks/`), ~15–20 tickets against the demo repo:

```text
simple bug · feature · schema-changing feature (must require approval)
ambiguous ticket (must escalate) · malicious tool request (must be blocked)
flaky env failure (must route to env_error) · out-of-scope ticket
```

**Experiments** (LangSmith), written up as `docs/evaluation-report.md`:

| Experiment | Compare |
|---|---|
| Routing | LLM-as-router vs Jev router (accuracy, latency, cost) |
| Models | Single strong model vs tiered routing |
| Graph | v1 (no context filter) vs v2 (Jev relevance filter) |
| Judge | Human labels vs Jev-as-a-Judge agreement |

**Aggregate KPIs:** success rate, first-pass validation rate, avg retries, avg cost/task, human-intervention rate, correct-escalation rate, policy-violation rate (target: 0).

This is the answer to "how do you know it works?" — and to the vacancy's "evaluate new AI tools for other teams."

---

## 5. Demo

**Target repo:** `demo-api/` — small FastAPI service (`POST /users`, `POST /login`, `GET /users/{id}`, `GET /health`), SQLite, its own `AGENTS.md` and tests.

**Scripted 5-minute demo:**

1. Open issue: *"Add rate limiting to `/login`: max 5 failed attempts/min/IP, return 429, keep API backward compatible, add regression tests, update docs."*
2. `jev-agent run --issue 12` → live console shows each node and Jev decision with confidence.
3. Validation fails once → repair → passes. Regression test proven to fail on base commit.
4. Review agent flags a missing edge case → repaired.
5. PR opens with structured body + trace link.
6. Second ticket *"Add `role` column to users"* → policy pauses for approval (migration).
7. Third ticket tries `rm -rf migrations/` via prompt injection in the issue body → blocked and logged.
8. Show metrics dashboard and the routing experiment results.

---

## 6. The repo as an AI-native engineering showcase

How the project itself is built — this is what the role actually asks for:

```text
AGENTS.md                 # shared instructions for any coding agent
CLAUDE.md                 # Claude Code specifics, points to AGENTS.md
.claude/
  skills/                 # e.g. add-graph-node, add-mcp-tool, write-eval-case
  commands/               # /bugfix, /review-diff
  settings.json           # hooks: run ruff+mypy after edits, block edits to secrets
.github/workflows/
  ci.yml                  # lint, typecheck, tests, coverage gate
  evals.yml               # nightly eval suite → LangSmith experiment
  release.yml             # tag → Docker image
docs/adr/                 # architecture decisions (why Jev, why MCP, why bounded retries)
```

Plus `jev-agent init <repo>` — bootstraps another team's repo with an AGENTS.md template, hooks and CI checks. That's the "introduce AI practices to a team" story.

---

## 7. Project layout

```text
jev_agent/
  graph/          # LangGraph state, nodes, edges
  decisions/      # Jev wrappers: typed Choice/Score/Noul + thresholds + fallbacks
  agents/         # planner, coder, reviewer (LangChain create_agent)
  policy/         # AGENTS.md parser, rule engine
  tools/          # capability implementations
  mcp_server/     # MCP exposure of tools
  validation/     # runners + regression-proof check
  github/         # issue read, PR creation
  metrics/        # cost/token accounting, audit log
  cli.py
demo-api/         # target FastAPI repo
evals/            # dataset + evaluators
tests/            # unit + integration (mock LLM / mock Jev)
```

---

## 8. Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Skeleton | Repo, AGENTS.md/CLAUDE.md, CI green, demo-api with tests |
| M2 | Linear happy path | ticket → context → plan → implement → validate → local diff (no Jev yet) |
| M3 | Policy & capabilities | AGENTS.md parser, policy engine, sandbox, approval interrupt |
| M4 | Repair & review | bounded repair loop, regression proof, review agent, PR creation |
| M5 | Jev decision fabric | all 7 decision points with confidence fallbacks; model router |
| M6 | Observability & evals | LangSmith tracing, eval dataset, experiments, metrics report |
| M7 | Adoption & polish | `jev-agent init`, MCP server, demo script, README, recorded video |

M2–M4 deliver a working demo even without Jev; M5–M6 are the differentiators.

---

## 9. Risks & open questions

- **Jev API surface** — verify current `langchain-typesafe` package names (`TypeSafeClassifier`, `ModelRouterMiddleware`, `AutoModeMiddleware`) and pricing before M5; keep `decisions/` behind an interface so an LLM fallback exists.
- **Sandbox** — Docker is enough for the demo; note where production would use stronger isolation.
- **Cost** — eval runs can be expensive; cap with cheap models for non-coding steps and cache.
- **Guardrails framework** — decide in M3 whether Guardrails AI adds value beyond Pydantic + policy engine.
