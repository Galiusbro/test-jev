"""Jev clients.

`JevClient` is the interface the workflow depends on. `HttpJevClient` calls the
TypeSafe API (wire format in `wire`).
`FakeJevClient` returns scripted answers for tests and offline runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from jev_agent.config import Settings
from jev_agent.decisions import wire
from jev_agent.decisions.types import Evaluation, Question

State = str | Mapping[str, Any] | list[Any]

# Rate limited / overloaded: retry with exponential backoff (docs.typesafe.ai/api).
_RETRY_STATUSES = {429, 503, 529}


class JevError(RuntimeError):
    """Jev call failed. Callers treat this like a low-confidence answer."""


class JevClient(Protocol):
    def evaluate(self, state: State, questions: Mapping[str, Question]) -> Evaluation: ...


class HttpJevClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.typesafe.ai/v1",
        model: str = "jev-latest",
        timeout_s: float = 30.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        http: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._http = http or httpx.Client(
            base_url=base_url,
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> HttpJevClient:
        if settings.typesafe_api_key is None:
            raise JevError("TYPESAFE_API_KEY is not set (get one at console.typesafe.ai)")
        return cls(
            settings.typesafe_api_key.get_secret_value(),
            base_url=settings.typesafe_base_url,
            model=settings.jev_model,
            timeout_s=settings.jev_timeout_s,
        )

    def evaluate(self, state: State, questions: Mapping[str, Question]) -> Evaluation:
        payload = {
            "model": self._model,
            "state": state,
            "questions": {n: wire.encode_question(q) for n, q in questions.items()},
        }
        response = self._post(payload)
        try:
            evaluation = Evaluation.model_validate(wire.decode(response.json()))
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            raise JevError(f"Unexpected Jev response: {exc}") from exc
        missing = set(questions) - set(evaluation.answers)
        if missing:
            raise JevError(f"Jev response is missing answers for: {sorted(missing)}")
        return evaluation

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post("/systemone", json=payload)
            except httpx.HTTPError as exc:
                raise JevError(f"Jev request failed: {exc}") from exc
            if response.status_code in _RETRY_STATUSES and attempt < self._max_retries:
                self._sleep(self._retry_delay(response, attempt))
                continue
            if response.status_code >= 400:
                raise JevError(f"Jev returned HTTP {response.status_code}: {response.text[:500]}")
            return response
        raise AssertionError("unreachable")

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        try:
            return min(float(response.headers["retry-after"]), 30.0)
        except (KeyError, ValueError):
            return 0.5 * 2.0**attempt

    def close(self) -> None:
        self._http.close()


Responder = Callable[[State, Mapping[str, Question]], Mapping[str, Any]]


class FakeJevClient:
    """Returns answers from a responder function; records every call.

    The responder returns answers in the normalized shape (see `types`).
    """

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls: list[tuple[State, dict[str, Question]]] = []

    def evaluate(self, state: State, questions: Mapping[str, Question]) -> Evaluation:
        self.calls.append((state, dict(questions)))
        return Evaluation.model_validate({"answers": dict(self._responder(state, questions))})
