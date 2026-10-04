"""Jev clients.

`JevClient` is the interface the workflow depends on. `TypeSafeJevClient`
wraps the official LangChain integration, `langchain_typesafe.TypeSafeClassifier`
— a Runnable, so every Jev call shows up in LangSmith traces nested under the
LangGraph node that made it. `FakeJevClient` returns scripted answers for tests
and offline runs.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from langchain_core._api import LangChainBetaWarning
from langchain_core.runnables import RunnableConfig
from langchain_typesafe import (
    Choice,
    ClassifierResponse,
    Noul,
    NoulCriteria,
    Score,
    TypeSafeClassifier,
)
from langchain_typesafe import Question as TypeSafeQuestion
from langchain_typesafe.client import (
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)
from pydantic import ValidationError

from jev_agent.config import Settings
from jev_agent.decisions.types import (
    BooleanQuestion,
    ChoiceQuestion,
    Evaluation,
    Question,
    ScoreQuestion,
)

State = str | Mapping[str, Any] | list[Any]


class JevError(RuntimeError):
    """Jev call failed. Callers treat this like a low-confidence answer."""


class JevClient(Protocol):
    def evaluate(
        self, state: State, questions: Mapping[str, Question], *, name: str | None = None
    ) -> Evaluation: ...


class TypeSafeJevClient:
    """Jev through `TypeSafeClassifier`, with the retries the SDK leaves to callers."""

    def __init__(
        self,
        classifier: TypeSafeClassifier,
        *,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.classifier = classifier
        self._max_retries = max_retries
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> TypeSafeJevClient:
        if settings.typesafe_api_key is None:
            raise JevError("TYPESAFE_API_KEY is not set (get one at console.typesafe.ai)")
        with warnings.catch_warnings():
            # Beta API; the package version is pinned in pyproject.toml.
            warnings.simplefilter("ignore", LangChainBetaWarning)
            classifier = TypeSafeClassifier(
                api_key=settings.typesafe_api_key,
                base_url=settings.typesafe_base_url,
                model=settings.jev_model,
                timeout=settings.jev_timeout_s,
            )
        return cls(classifier)

    def evaluate(
        self, state: State, questions: Mapping[str, Question], *, name: str | None = None
    ) -> Evaluation:
        request = {
            "state": state,
            "questions": {n: _to_typesafe(q) for n, q in questions.items()},
        }
        config: RunnableConfig = {"run_name": f"jev:{name}" if name else "jev", "tags": ["jev"]}
        response = self._invoke(request, config)
        try:
            evaluation = _from_typesafe(response)
        except (KeyError, ValueError, ValidationError) as exc:
            raise JevError(f"Unexpected Jev response: {exc}") from exc
        missing = set(questions) - set(evaluation.answers)
        if missing:
            raise JevError(f"Jev response is missing answers for: {sorted(missing)}")
        return evaluation

    def _invoke(self, request: dict[str, Any], config: RunnableConfig) -> ClassifierResponse:
        for attempt in range(self._max_retries + 1):
            try:
                return self.classifier.invoke(request, config)  # type: ignore[arg-type]
            except (TypeSafeRateLimitError, TypeSafeInternalServerError) as exc:
                # 429 rate limit, 5xx/529 overloaded: back off and retry.
                if attempt == self._max_retries:
                    raise JevError(f"Jev unavailable after retries: {exc}") from exc
                retry_ms = getattr(exc, "retry_after_ms", None)
                delay = retry_ms / 1000 if retry_ms else 0.5 * 2.0**attempt
                self._sleep(min(delay, 30.0))
            except TypeSafeError as exc:
                raise JevError(f"Jev request failed: {exc}") from exc
        raise AssertionError("unreachable")


def _to_typesafe(question: Question) -> TypeSafeQuestion:
    match question:
        case BooleanQuestion():
            criteria = (
                NoulCriteria(true=question.criteria.true, false=question.criteria.false)
                if question.criteria
                else None
            )
            return Noul(instructions=question.instructions, criteria=criteria)
        case ChoiceQuestion():
            return Choice(instructions=question.instructions, criteria=dict(question.options))
        case ScoreQuestion():
            return Score(instructions=question.instructions, criteria=list(question.levels))


def _from_typesafe(response: ClassifierResponse) -> Evaluation:
    """Normalize SDK answers into our typed `Evaluation`."""
    answers: dict[str, dict[str, Any]] = {}
    for name, noul in response.nouls.items():
        answers[name] = {"type": "boolean", "probability": noul.noul}
    for name, choice in response.choices.items():
        answers[name] = {
            "type": "choice",
            "value": choice.choice,
            "probability": choice.probabilities.get(choice.choice, 0.0),
            "confidence": choice.confidence,
            "probabilities": choice.probabilities,
        }
    for name, score in response.scores.items():
        peak = max(score.probabilities, key=lambda level: score.probabilities[level])
        answers[name] = {
            "type": "score",
            "score": score.score,
            "value": str(score.legend.get(peak, peak)),
            "probability": score.probabilities[peak],
            "confidence": score.confidence,
            "probabilities": {str(k): v for k, v in score.probabilities.items()},
            "legend": {str(k): str(v) for k, v in score.legend.items()},
        }
    return Evaluation.model_validate({"answers": answers, "model": response.model})


Responder = Callable[[State, Mapping[str, Question]], Mapping[str, Any]]


class FakeJevClient:
    """Returns answers from a responder function; records every call.

    The responder returns answers in the normalized shape (see `types`).
    """

    def __init__(self, responder: Responder) -> None:
        self._responder = responder
        self.calls: list[tuple[State, dict[str, Question]]] = []
        self.names: list[str | None] = []

    def evaluate(
        self, state: State, questions: Mapping[str, Question], *, name: str | None = None
    ) -> Evaluation:
        self.calls.append((state, dict(questions)))
        self.names.append(name)
        return Evaluation.model_validate({"answers": dict(self._responder(state, questions))})
