from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from langchain_core.tracers.context import collect_runs
from langchain_typesafe import TypeSafeClassifier
from pydantic import SecretStr

from jev_agent.config import Settings
from jev_agent.decisions import (
    BooleanQuestion,
    ChoiceQuestion,
    FakeJevClient,
    JevError,
    Question,
    ScoreQuestion,
    TypeSafeJevClient,
)
from jev_agent.decisions.types import BooleanCriteria

QUESTIONS: dict[str, Question] = {
    "done": BooleanQuestion(
        instructions="Is the task complete?",
        criteria=BooleanCriteria(true="All requirements met", false="Work remains"),
    ),
    "kind": ChoiceQuestion(
        instructions="What kind of ticket is this?",
        options={"bug": "Broken behaviour", "feature": "New behaviour"},
    ),
    "risk": ScoreQuestion(instructions="How risky?", levels=["low", "medium", "high"]),
}

FULL_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "done": {"type": "noul", "noul": 0.95},
        "kind": {
            "type": "choice",
            "choice": "bug",
            "probabilities": {"bug": 0.88, "feature": 0.12},
            "confidence": 0.81,
        },
        "risk": {
            "type": "score",
            "score": 1.05,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
            "confidence": 0.92,
        },
    },
    "usage": {"input_tokens": 300, "output_tokens": 20},
}

Handler = Callable[[httpx2.Request], httpx2.Response]


def client_with(
    handler: Handler | list[httpx2.Response], sleeps: list[float] | None = None
) -> tuple[TypeSafeJevClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []
    queue = list(handler) if isinstance(handler, list) else None

    def transport(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if queue is not None:
            return queue.pop(0)
        assert callable(handler)
        return handler(request)

    classifier = TypeSafeClassifier(
        api_key="key-123",
        base_url="https://jev.test",
        client=httpx2.Client(transport=httpx2.MockTransport(transport)),
    )
    log = sleeps if sleeps is not None else []
    return TypeSafeJevClient(classifier, sleep=log.append), seen


def ok(body: dict[str, Any]) -> Handler:
    return lambda _request: httpx2.Response(200, json=body)


def test_maps_questions_to_sdk_and_answers_back() -> None:
    client, seen = client_with(ok(FULL_RESPONSE))
    evaluation = client.evaluate({"ticket": "Login returns 500"}, QUESTIONS, name="triage")

    request = seen[0]
    assert str(request.url) == "https://jev.test/v1/systemone"
    assert request.headers["Authorization"] == "Bearer key-123"
    body = json.loads(request.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == {"ticket": "Login returns 500"}
    assert body["questions"] == {
        "done": {
            "type": "noul",
            "instructions": "Is the task complete?",
            "criteria": {"true": "All requirements met", "false": "Work remains"},
        },
        "kind": {
            "type": "choice",
            "instructions": "What kind of ticket is this?",
            "criteria": {"bug": "Broken behaviour", "feature": "New behaviour"},
        },
        "risk": {
            "type": "score",
            "instructions": "How risky?",
            "criteria": ["low", "medium", "high"],
        },
    }

    assert evaluation.model == "jev-1.13.0"
    done = evaluation.boolean("done")
    assert done.probability == 0.95 and done.yes
    assert done.confidence == pytest.approx(0.9)  # |2p - 1|
    kind = evaluation.choice("kind")
    assert (kind.value, kind.probability, kind.confidence) == ("bug", 0.88, 0.81)
    risk = evaluation.score("risk")
    assert (risk.value, risk.score, risk.probability, risk.confidence) == (
        "medium",
        1.05,
        0.95,
        0.92,
    )
    assert risk.legend == {"0": "low", "1": "medium", "2": "high"}


def test_structured_instructions_pass_through() -> None:
    client, seen = client_with(ok({"model": "m", "answers": {"q": {"type": "noul", "noul": 0.1}}}))
    question = BooleanQuestion(instructions={"finding": "x", "question": "Is `finding` real?"})
    client.evaluate("diff", {"q": question})
    sent = json.loads(seen[0].content)["questions"]["q"]
    assert sent == {
        "type": "noul",
        "instructions": {"finding": "x", "question": "Is `finding` real?"},
    }


def test_calls_are_langchain_runs_named_per_decision() -> None:
    client, _ = client_with(ok(FULL_RESPONSE))
    with collect_runs() as collector:
        client.evaluate("x", QUESTIONS, name="assess_plan")
    (run,) = collector.traced_runs
    assert run.name == "jev:assess_plan"
    assert "jev" in (run.tags or [])
    assert run.extra["metadata"]["ls_provider"] == "typesafe"


def test_retries_rate_limit_and_overload_then_succeeds() -> None:
    sleeps: list[float] = []
    client, seen = client_with(
        [
            httpx2.Response(429, headers={"retry-after": "2"}, json={"error": "slow"}),
            httpx2.Response(529, json={"error": "overloaded"}),
            httpx2.Response(
                200, json={"model": "m", "answers": {"q": {"type": "noul", "noul": 1}}}
            ),
        ],
        sleeps,
    )
    evaluation = client.evaluate("x", {"q": BooleanQuestion(instructions="?")})
    assert evaluation.boolean("q").probability == 1.0
    assert len(seen) == 3
    assert sleeps == [2.0, 1.0]  # retry-after honoured, then exponential backoff


def test_gives_up_after_max_retries() -> None:
    sleeps: list[float] = []
    client, _ = client_with(lambda _r: httpx2.Response(529, json={"error": "overloaded"}), sleeps)
    with pytest.raises(JevError, match="unavailable after retries"):
        client.evaluate("x", {"q": BooleanQuestion(instructions="?")})
    assert len(sleeps) == 3


def test_auth_error_not_retried() -> None:
    sleeps: list[float] = []
    client, seen = client_with(lambda _r: httpx2.Response(401, json={"error": "bad key"}), sleeps)
    with pytest.raises(JevError, match="request failed"):
        client.evaluate("x", {"q": BooleanQuestion(instructions="?")})
    assert sleeps == [] and len(seen) == 1


def test_network_error_raises_jev_error() -> None:
    def down(_request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    client, _ = client_with(down)
    with pytest.raises(JevError, match="request failed"):
        client.evaluate("x", {"q": BooleanQuestion(instructions="?")})


def test_missing_answer_raises_jev_error() -> None:
    client, _ = client_with(ok({"model": "m", "answers": {}}))
    with pytest.raises(JevError, match="missing answers"):
        client.evaluate("x", {"q": BooleanQuestion(instructions="?")})


def test_malformed_response_raises_jev_error() -> None:
    client, _ = client_with(ok({"model": "m", "answers": {"q": {"type": "noul"}}}))
    with pytest.raises(JevError):
        client.evaluate("x", {"q": BooleanQuestion(instructions="?")})


def test_wrong_answer_kind_raises_type_error() -> None:
    fake = FakeJevClient(lambda _s, _q: {"done": {"type": "boolean", "probability": 1.0}})
    evaluation = fake.evaluate("x", {"done": BooleanQuestion(instructions="Done?")}, name="n")
    with pytest.raises(TypeError):
        evaluation.choice("done")
    assert fake.names == ["n"]


def test_from_settings() -> None:
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        TypeSafeJevClient.from_settings(Settings(_env_file=None))
    client = TypeSafeJevClient.from_settings(
        Settings(_env_file=None, typesafe_api_key=SecretStr("ts-key"), jev_model="jev-1.13.0")
    )
    assert client.classifier.model == "jev-1.13.0"
    assert client.classifier.base_url == "https://api.typesafe.ai"
