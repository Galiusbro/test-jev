"""Decision fabric: the fast judgment calls Jev makes on the workflow's edges.

LLMs write; Jev decides where to go next; code enforces the result. Every
decision here is a small typed question with:

- a confidence threshold — below it, or on any Jev error, the decision takes
  its documented conservative fallback;
- a record in the decision log (question, answer, confidence, outcome).

Without a Jev client every decision returns its fallback, which reproduces the
pre-Jev behaviour — that is the baseline for A/B comparisons.
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from jev_agent.agents import Finding, Plan
from jev_agent.decisions.client import JevClient, JevError, State
from jev_agent.decisions.types import (
    BooleanCriteria,
    BooleanQuestion,
    ChoiceQuestion,
    Evaluation,
    Question,
    ScoreQuestion,
)
from jev_agent.tickets import Ticket

TicketKind = Literal["bug", "feature", "docs", "refactor", "out_of_scope"]
FailureKind = Literal["code", "test", "environment"]
Complexity = Literal["low", "medium", "high"]

MAX_STATE_CHARS = 8_000
WRITE_GATE_THRESHOLD = 0.75  # Jev degrades on large, noisy state; keep evidence focused


@dataclass
class DecisionRecord:
    name: str
    answers: dict[str, Any]
    outcome: str
    fallback: bool = False
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class Triage:
    kind: TicketKind
    actionable: bool
    stop_reason: str | None  # set when the run should go to a human instead


@dataclass
class PlanAssessment:
    complexity: Complexity
    escalate: str | None  # approval reason, when the plan looks high-risk


@dataclass
class Decisions:
    client: JevClient | None
    min_confidence: float = 0.7
    log: list[DecisionRecord] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _ask(
        self, name: str, state: State, questions: Mapping[str, Question]
    ) -> tuple[Evaluation | None, float, str | None]:
        if self.client is None:
            return None, 0.0, "jev disabled"
        start = time.perf_counter()
        try:
            evaluation = self.client.evaluate(state, questions, name=name)
        except JevError as exc:
            return None, time.perf_counter() - start, str(exc)
        return evaluation, time.perf_counter() - start, None

    def _record(
        self,
        name: str,
        evaluation: Evaluation | None,
        outcome: str,
        *,
        fallback: bool,
        latency: float,
        error: str | None,
    ) -> None:
        answers = (
            {
                k: a.model_dump(include={"type", "probability", "confidence"}) | _value(a)
                for k, a in evaluation.answers.items()
            }
            if evaluation
            else {}
        )
        self.log.append(DecisionRecord(name, answers, outcome, fallback, error, round(latency, 3)))

    def _sure(self, answer: Any) -> bool:
        return bool(answer.is_confident(self.min_confidence))

    # 1 — triage ---------------------------------------------------------------

    def triage(self, ticket: Ticket, rules: str) -> Triage:
        """Fallback: treat the ticket as an actionable feature (pre-Jev behaviour)."""
        evaluation, latency, error = self._ask(
            "triage",
            {"ticket": ticket.as_text(), "project_rules": _clip(rules)},
            {
                "kind": ChoiceQuestion(
                    instructions="What kind of work does the `ticket` ask for in this repository?",
                    options={
                        "bug": "Existing behaviour is broken and must be fixed",
                        "feature": "New or changed behaviour",
                        "docs": "Documentation only",
                        "refactor": "Restructure code without changing behaviour",
                        "out_of_scope": "Not a code change to this repository, or something "
                        "the `project_rules` forbid",
                    },
                ),
                "actionable": BooleanQuestion(
                    instructions="Does the `ticket` state concrete requirements that a "
                    "developer could implement and test without asking for clarification?"
                ),
            },
        )
        if evaluation is None:
            result = Triage("feature", True, None)
            self._record("triage", None, "proceed", fallback=True, latency=latency, error=error)
            return result
        kind, actionable = evaluation.choice("kind"), evaluation.boolean("actionable")
        stop = None
        if kind.value == "out_of_scope" and self._sure(kind):
            stop = "ticket is out of scope for this repository"
        elif not actionable.yes and self._sure(actionable):
            stop = "ticket is too vague to implement without clarification"
        result = Triage(kind.value, actionable.yes, stop)  # type: ignore[arg-type]
        self._record(
            "triage",
            evaluation,
            f"stop: {stop}" if stop else f"proceed ({kind.value})",
            fallback=False,
            latency=latency,
            error=None,
        )
        return result

    # 2 + 3 — plan risk and complexity (one fan-out request) -------------------

    def assess_plan(self, ticket: Ticket, plan: Plan) -> PlanAssessment:
        """Fallback: medium complexity, no extra escalation (policy rules still apply)."""
        evaluation, latency, error = self._ask(
            "assess_plan",
            {"ticket": ticket.as_text(), "plan": plan.model_dump()},
            {
                "risk": ScoreQuestion(
                    instructions="How risky is carrying out the `plan` in this codebase?",
                    levels=[
                        "Low: internal change, easy to review and revert",
                        "Medium: user-visible behaviour change with limited blast radius",
                        "High: security-sensitive, data-destroying, or hard to reverse",
                    ],
                ),
                "complexity": ScoreQuestion(
                    instructions="How complex is implementing the `plan` correctly?",
                    levels=[
                        "Small, localized change in one or two files",
                        "Moderate change across a few files with new tests",
                        "Large or cross-cutting change, or subtle logic",
                    ],
                ),
            },
        )
        if evaluation is None:
            self._record(
                "assess_plan",
                None,
                "medium, no escalation",
                fallback=True,
                latency=latency,
                error=error,
            )
            return PlanAssessment("medium", None)
        risk, complexity = evaluation.score("risk"), evaluation.score("complexity")
        level: Complexity = "medium"
        if self._sure(complexity):
            if complexity.score < 0.5:
                level = "low"
            elif complexity.score >= 1.5:
                level = "high"
        escalate = None
        if risk.score >= 1.5 and self._sure(risk):
            escalate = f"Jev rates the plan high-risk ({risk.value})"
        self._record(
            "assess_plan",
            evaluation,
            f"complexity={level}, escalate={bool(escalate)}",
            fallback=False,
            latency=latency,
            error=None,
        )
        return PlanAssessment(level, escalate)

    # 4 — semantic write gate --------------------------------------------------

    def write_gate(
        self,
        ticket: Ticket,
        plan: Plan,
        advisory_rules: Sequence[str],
        path: str,
        before: str | None,
        after: str,
    ) -> str | None:
        """Return an approval reason if the edit looks like it breaks a rule.

        Covers the rules code cannot check (e.g. "Disabling lint, type checks
        or tests"). Fallback: allow — deterministic policy already ran.
        """
        edit = "".join(
            difflib.unified_diff(
                (before or "").splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=path,
                tofile=path,
            )
        )
        evaluation, latency, error = self._ask(
            "write_gate",
            {
                "ticket": ticket.title,
                "plan_summary": plan.summary,
                "rules": list(advisory_rules),
                "edit": _clip(edit),
            },
            {
                "violates": BooleanQuestion(
                    instructions="Does the `edit` break one of the `rules`, or do something "
                    "the `ticket` clearly does not ask for (such as disabling checks or "
                    "deleting unrelated code)?",
                    criteria=BooleanCriteria(
                        true="The edit breaks a rule or is clearly out of scope",
                        false="The edit is a plausible part of implementing the ticket",
                    ),
                )
            },
        )
        if evaluation is None:
            self._record(
                f"write_gate:{path}", None, "allow", fallback=True, latency=latency, error=error
            )
            return None
        violates = evaluation.boolean("violates")
        reason = None
        # Probability alone: measured violations scored 0.80-0.96, a normal edit
        # 0.09; requiring confidence too would push the bar to ~0.85 and miss
        # `@pytest.mark.skip` (0.80).
        if violates.probability >= WRITE_GATE_THRESHOLD:
            reason = "Jev flags this edit as breaking a project rule or the ticket's scope"
        self._record(
            f"write_gate:{path}",
            evaluation,
            "escalate" if reason else "allow",
            fallback=False,
            latency=latency,
            error=None,
        )
        return reason

    # 5 — failure diagnosis ----------------------------------------------------

    def diagnose(self, ticket: Ticket, failures: str, diff: str) -> FailureKind:
        """Fallback: "code" — the normal repair path (pre-Jev behaviour)."""
        evaluation, latency, error = self._ask(
            "diagnose",
            {"ticket": ticket.as_text(), "failures": _clip(failures), "diff": _clip(diff)},
            {
                "cause": ChoiceQuestion(
                    instructions="Given the `ticket`, the `diff` and the check `failures`, "
                    "what most likely needs to change?",
                    options={
                        "code": "The application code; the tests correctly describe the ticket",
                        "test": "The tests added for this ticket; they contradict the ticket",
                        "environment": "Nothing in the change: missing tools, dependencies, "
                        "network or environment problems",
                    },
                )
            },
        )
        if evaluation is None:
            self._record("diagnose", None, "code", fallback=True, latency=latency, error=error)
            return "code"
        cause = evaluation.choice("cause")
        kind: FailureKind = cause.value if self._sure(cause) else "code"  # type: ignore[assignment]
        self._record(
            "diagnose",
            evaluation,
            kind,
            fallback=not self._sure(cause),
            latency=latency,
            error=None,
        )
        return kind

    # 6 — review finding verification -----------------------------------------

    def verify_findings(self, ticket: Ticket, diff: str, findings: Sequence[Finding]) -> list[bool]:
        """One supported/unsupported verdict per finding. Fallback: keep (True)."""
        if not findings:
            return []
        questions: dict[str, Question] = {
            f"finding_{i}": BooleanQuestion(
                instructions={
                    "finding": f"{f.file}: {f.issue}",
                    "question": "Is the `finding` a real problem that the `diff` actually "
                    "contains, given what the `ticket` requires?",
                }
            )
            for i, f in enumerate(findings)
        }
        evaluation, latency, error = self._ask(
            "verify_findings", {"ticket": ticket.as_text(), "diff": _clip(diff)}, questions
        )
        if evaluation is None:
            self._record(
                "verify_findings", None, "keep all", fallback=True, latency=latency, error=error
            )
            return [True] * len(findings)
        verdicts = []
        for name in questions:
            answer = evaluation.boolean(name)
            verdicts.append(not (not answer.yes and self._sure(answer)))
        self._record(
            "verify_findings",
            evaluation,
            f"supported {sum(verdicts)}/{len(verdicts)}",
            fallback=False,
            latency=latency,
            error=None,
        )
        return verdicts


def _value(answer: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("value", "score"):
        if hasattr(answer, key):
            out[key] = getattr(answer, key)
    return out


def _clip(text: str) -> str:
    return text if len(text) <= MAX_STATE_CHARS else text[:MAX_STATE_CHARS] + "\n…(truncated)…"
