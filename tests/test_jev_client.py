from __future__ import annotations

import json

import httpx
import pytest
import respx
from pydantic import SecretStr

from jev_agent.config import Settings
from jev_agent.decisions import (
    BooleanQuestion,
    ChoiceQuestion,
    FakeJevClient,
    HttpJevClient,
    JevError,
    Question,
    ScoreQuestion,
)
from jev_agent.decisions.types import BooleanCriteria

BASE = "https://jev.test/v1"

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


def _client(sleeps: list[float] | None = None) -> HttpJevClient:
    log = sleeps if sleeps is not None else []
    return HttpJevClient("key-123", base_url=BASE, sleep=log.append)


# --- Wire format (docs.typesafe.ai/api) ----------------------------------------


@respx.mock
def test_typesafe_request_and_response_mapping() -> None:
    route = respx.post(f"{BASE}/systemone").respond(
        json={
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
    )

    evaluation = _client().evaluate({"ticket": "Login returns 500"}, QUESTIONS)

    request = route.calls.last.request
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


@respx.mock
def test_noul_without_criteria_omits_field() -> None:
    route = respx.post(f"{BASE}/systemone").respond(
        json={"answers": {"q": {"type": "noul", "noul": 0.5}}}
    )
    answer = _client().evaluate("x", {"q": BooleanQuestion(instructions="Done?")}).boolean("q")
    assert "criteria" not in json.loads(route.calls.last.request.content)["questions"]["q"]
    assert answer.confidence == 0.0  # coin flip = no confidence
    assert not answer.is_confident(0.7)


@respx.mock
def test_retries_rate_limit_then_succeeds() -> None:
    route = respx.post(f"{BASE}/systemone")
    route.side_effect = [
        httpx.Response(429, headers={"retry-after": "2"}),
        httpx.Response(529),
        httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 1.0}}}),
    ]
    sleeps: list[float] = []
    evaluation = _client(sleeps=sleeps).evaluate("x", {"q": BooleanQuestion(instructions="?")})
    assert evaluation.boolean("q").probability == 1.0
    assert sleeps == [2.0, 1.0]  # retry-after honoured, then exponential backoff


@respx.mock
def test_gives_up_after_max_retries() -> None:
    respx.post(f"{BASE}/systemone").respond(status_code=429, text="slow down")
    sleeps: list[float] = []
    with pytest.raises(JevError, match="429"):
        _client(sleeps=sleeps).evaluate("x", {"q": BooleanQuestion(instructions="?")})
    assert len(sleeps) == 3


# --- Errors -------------------------------------------------------------------


@respx.mock
def test_auth_error_not_retried() -> None:
    respx.post(f"{BASE}/systemone").respond(status_code=401, text="bad key")
    sleeps: list[float] = []
    with pytest.raises(JevError, match="401"):
        _client(sleeps=sleeps).evaluate("x", {"q": BooleanQuestion(instructions="?")})
    assert sleeps == []


@respx.mock
def test_network_error_raises_jev_error() -> None:
    respx.post(f"{BASE}/systemone").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(JevError, match="request failed"):
        _client().evaluate("x", {"q": BooleanQuestion(instructions="?")})


@respx.mock
def test_missing_answer_raises_jev_error() -> None:
    respx.post(f"{BASE}/systemone").respond(json={"answers": {}})
    with pytest.raises(JevError, match="missing answers"):
        _client().evaluate("x", {"q": BooleanQuestion(instructions="?")})


@pytest.mark.parametrize(
    "answer",
    [{"type": "noul"}, {"type": "mystery", "noul": 1}, {"type": "score", "score": 1.0}],
)
@respx.mock
def test_malformed_response_raises_jev_error(answer: dict[str, object]) -> None:
    respx.post(f"{BASE}/systemone").respond(json={"answers": {"q": answer}})
    with pytest.raises(JevError, match="Unexpected"):
        _client().evaluate("x", {"q": BooleanQuestion(instructions="?")})


def test_wrong_answer_kind_raises_type_error() -> None:
    fake = FakeJevClient(lambda _s, _q: {"done": {"type": "boolean", "probability": 1.0}})
    evaluation = fake.evaluate("x", {"done": BooleanQuestion(instructions="Done?")})
    with pytest.raises(TypeError):
        evaluation.choice("done")
    assert len(fake.calls) == 1


# --- Settings -----------------------------------------------------------------


def test_from_settings_requires_key() -> None:
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        HttpJevClient.from_settings(Settings(_env_file=None))


@respx.mock
def test_from_settings_uses_url_and_pinned_model() -> None:
    route = respx.post("https://api.typesafe.ai/v1/systemone").respond(
        json={"answers": {"q": {"type": "noul", "noul": 1.0}}}
    )
    settings = Settings(
        _env_file=None, typesafe_api_key=SecretStr("ts-key"), jev_model="jev-1.13.0"
    )
    HttpJevClient.from_settings(settings).evaluate("x", {"q": BooleanQuestion(instructions="?")})
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer ts-key"
    assert json.loads(request.content)["model"] == "jev-1.13.0"
