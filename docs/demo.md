# Demo script (≈5 minutes)

A walkthrough for showing jev-agent live or from recorded runs. Each step has
the command, what to point at, and one sentence to say.

> Prep: `uv sync`, keys in `.env`, `uv run jev-agent doctor` all green. Live
> runs on the free NVIDIA tier take 4–20 min, so start step 2 before you talk,
> or show a finished run from `runs/` and its LangSmith trace.

## 1. The problem (30 s)

Open [`demo-api/AGENTS.md`](../demo-api/AGENTS.md).

> "Teams adopt coding agents one chat window at a time. This project turns the
> agent into a governed workflow: the repo's own AGENTS.md defines what an
> agent may change, and that is enforced in code — the model is not the
> security boundary."

## 2. Ticket → validated PR (90 s)

```bash
uv run jev-agent run tickets/004-trim-user-names.md --approvals all --open-pr
```

Point at the node lines as they appear, then at
[PR #1](https://github.com/Galiusbro/test-jev/pull/1):

- `triage` — Jev classifies the ticket (bug, actionable) in ~0.4 s.
- `policy_check` — the plan declares a public API change, so a human approves.
- `validate` → `prove_tests` — checks pass, and the new tests **fail on the
  original code**, so they really test the change.
- `review` — an independent reviewer; findings must quote the diff.
- PR body: plan, validation, regression proof, approvals, Jev decisions.

> "The output isn't 'the model says it's done' — it's evidence: checks, a
> regression proof, a grounded review and an audit trail."

## 3. Stopping the wrong work (60 s)

```bash
uv run jev-agent run tickets/003-deploy-to-production.md --approvals none
uv run jev-agent run tickets/005-skip-flaky-test.md --approvals none
```

Both stop at `triage` in under a second with zero LLM tokens.

> "Without Jev the same deploy ticket became an approved code change after 17
> minutes; the skip-the-test ticket was only caught by luck in review."

Show [`docs/evaluation-report.md`](evaluation-report.md): Jev 6/6 correct, 0
harmful; baseline 4/6 with one harmful change, twice the time and tokens.

## 4. Where LLM, Jev and code each decide (45 s)

Open [`src/jev_agent/graph.py`](../src/jev_agent/graph.py) docstring and
[`decisions/fabric.py`](../src/jev_agent/decisions/fabric.py).

> "LLMs write. Jev makes the fast typed judgments on the graph's edges —
> triage, risk, model routing, failure diagnosis, review verification — each
> with a confidence threshold and a fallback. Deterministic code enforces
> policy. LangGraph orchestrates, LangChain `create_agent` with middleware
> runs the agents, LangSmith traces everything."

## 5. Bring it to any repo and any agent (45 s)

```bash
uv run jev-agent init ../some-other-repo     # AGENTS.md, CLAUDE.md, .mcp.json
uv run jev-agent mcp --repo demo-api          # same guardrails for Claude Code / Cursor
```

> "Adoption is one command: it detects the stack and writes the rules file the
> policy engine enforces. The MCP server gives a developer's own agent the
> same governed tools — `.env` hidden, existing tests protected, only declared
> commands runnable."

## 6. Honest limits (30 s)

> "Free-tier models are slow and flaky — the harness has first-token
> timeouts, fallback chains and model cooldowns because of that. The eval is
> one run per case on a toy repo; the robust result is the qualitative one:
> Jev stops wrong work early, deterministic policy stays the backbone."

---

## Likely follow-up questions

| Question | Short answer | Where |
|---|---|---|
| Why not let the LLM decide routing? | Typed answers with calibrated confidence in ~0.3 s, no extra LLM call; low confidence takes a safe fallback. | `decisions/fabric.py` |
| What if Jev is down? | Every decision has a fallback that reproduces the pre-Jev behaviour; that's also the eval baseline. | `--no-jev` |
| How do you stop agents deleting tests? | AST diff of test functions against the base commit; the write is refused. | `policy.py` |
| How do you know the tests test anything? | Regression proof: changed tests must fail on the original code. | `proof.py` |
| Reviewer hallucinations? | Blocking findings must quote diff lines; Jev verifies them; unsupported ones are downgraded. | `agents.py` (`Review.grounded`) |
| Why not AutoModeMiddleware? | It sends 30 messages of full files to Jev with a fixed threshold and no human approval; our gate sends only the edit's diff and escalates to a human. | `harness.py` |
| Cost? | Per-run tokens, latency and failed attempts in `report.json`; aggregated by `jev-agent eval-report`. | `evals.py` |
