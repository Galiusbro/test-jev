from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from jev_agent.agents import Finding, Plan
from jev_agent.decisions import BooleanQuestion, FakeJevClient, JevError, Question
from jev_agent.decisions.client import State
from jev_agent.decisions.fabric import Decisions
from jev_agent.tickets import Ticket

TICKET = Ticket(id="t", title="Rate-limit logins", body="Max 5 failed attempts per minute.")
PLAN = Plan(
    summary="Add a limiter.",
    files_to_change=["src/app.py"],
    steps=["x"],
    tests=["y"],
    risk="medium",
    public_api_change=True,
    db_migration=False,
)


def yes(p: float = 0.95) -> dict[str, Any]:
    return {"type": "boolean", "probability": p}  # confidence derived: |2p - 1|


def choice(value: str, confidence: float = 0.9) -> dict[str, Any]:
    return {"type": "choice", "value": value, "probability": 0.9, "confidence": confidence}


def score(value: float, confidence: float = 0.9) -> dict[str, Any]:
    return {
        "type": "score",
        "value": str(value),
        "score": value,
        "probability": 0.8,
        "confidence": confidence,
    }


def jev(**answers: Any) -> tuple[Decisions, FakeJevClient]:
    def respond(_state: State, questions: Mapping[str, Question]) -> Mapping[str, Any]:
        return {name: answers[name] for name in questions}

    client = FakeJevClient(respond)
    return Decisions(client, min_confidence=0.7), client


def failing() -> Decisions:
    def boom(_s: State, _q: Mapping[str, Question]) -> Mapping[str, Any]:
        raise JevError("HTTP 529 overloaded")

    return Decisions(FakeJevClient(boom))


def test_disabled_fabric_returns_pre_jev_fallbacks() -> None:
    d = Decisions(None)
    assert not d.enabled
    assert d.triage(TICKET, "rules").stop_reason is None
    assert d.assess_plan(TICKET, PLAN).complexity == "medium"
    assert d.write_gate(TICKET, PLAN, [], "a.py", "x", "y") is None
    assert d.diagnose(TICKET, "failed", "diff") == "code"
    finding = Finding(severity="major", file="a.py", issue="i")
    assert d.verify_findings(TICKET, "diff", [finding]) == [True]
    assert all(r.fallback and r.error == "jev disabled" for r in d.log)


def test_jev_errors_fall_back_and_are_logged() -> None:
    d = failing()
    assert d.triage(TICKET, "rules").stop_reason is None
    assert d.diagnose(TICKET, "f", "d") == "code"
    assert d.log[0].fallback and "529" in (d.log[0].error or "")


@pytest.mark.parametrize(
    ("kind", "actionable", "stop"),
    [
        (choice("out_of_scope"), yes(0.9), "out of scope"),
        (choice("feature"), yes(0.05), "too vague"),
        (choice("out_of_scope", confidence=0.4), yes(0.9), None),  # unsure → proceed
        (choice("feature"), yes(0.4), None),  # vague-ish but unsure → proceed
        (choice("bug"), yes(0.9), None),
    ],
)
def test_triage(kind: dict[str, Any], actionable: dict[str, Any], stop: str | None) -> None:
    d, client = jev(kind=kind, actionable=actionable)
    result = d.triage(TICKET, "## Forbidden\n- x")
    assert (result.stop_reason is not None) == (stop is not None)
    if stop:
        assert stop in (result.stop_reason or "")
    state = client.calls[0][0]
    assert isinstance(state, dict) and "Rate-limit logins" in state["ticket"]


@pytest.mark.parametrize(
    ("complexity", "risk", "level", "escalates"),
    [
        (score(1.8), score(0.3), "high", False),
        (score(0.2), score(0.3), "low", False),
        (score(1.8, confidence=0.3), score(0.3), "medium", False),  # unsure → medium
        (score(1.0), score(1.9), "medium", True),
        (score(1.0), score(1.9, confidence=0.2), "medium", False),
    ],
)
def test_assess_plan(
    complexity: dict[str, Any], risk: dict[str, Any], level: str, escalates: bool
) -> None:
    d, client = jev(complexity=complexity, risk=risk)
    result = d.assess_plan(TICKET, PLAN)
    assert result.complexity == level
    assert (result.escalate is not None) is escalates
    assert set(client.calls[0][1]) == {"risk", "complexity"}  # one fan-out request


def test_write_gate_flags_confident_violations_only() -> None:
    d, client = jev(violates=yes(0.97))
    reason = d.write_gate(
        TICKET, PLAN, ["Disabling lint"], "pyproject.toml", "[lint]\n", "[lint]\nignore=['ALL']\n"
    )
    assert reason is not None and "Jev" in reason
    state = client.calls[0][0]
    assert isinstance(state, dict)
    assert state["rules"] == ["Disabling lint"]
    assert "+ignore=['ALL']" in state["edit"]

    d, _ = jev(violates=yes(0.7))  # leaning yes, below the bar
    assert d.write_gate(TICKET, PLAN, [], "a.py", None, "x = 1\n") is None
    d, _ = jev(violates=yes(0.8))  # measured: adding @pytest.mark.skip
    assert d.write_gate(TICKET, PLAN, [], "tests/test_a.py", "a", "b") is not None


@pytest.mark.parametrize(
    ("cause", "expected"),
    [
        (choice("environment"), "environment"),
        (choice("test"), "test"),
        (choice("environment", confidence=0.5), "code"),
    ],
)
def test_diagnose(cause: dict[str, Any], expected: str) -> None:
    d, _ = jev(cause=cause)
    assert d.diagnose(TICKET, "ModuleNotFoundError: fastapi", "diff") == expected


def test_verify_findings_uses_structured_instructions() -> None:
    findings = [
        Finding(severity="major", file="a.py", issue="Resets the counter."),
        Finding(severity="major", file="a.py", issue="Off by one."),
        Finding(severity="major", file="a.py", issue="Unclear."),
    ]
    d, client = jev(finding_0=yes(0.05), finding_1=yes(0.95), finding_2=yes(0.45))
    assert d.verify_findings(TICKET, "diff", findings) == [False, True, True]
    question = client.calls[0][1]["finding_0"]
    assert isinstance(question, BooleanQuestion)
    assert isinstance(question.instructions, dict)
    assert question.instructions["finding"] == "a.py: Resets the counter."
    assert d.verify_findings(TICKET, "diff", []) == []
    assert d.log[-1].outcome == "supported 2/3"
