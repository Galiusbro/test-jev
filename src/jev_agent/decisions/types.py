"""Typed questions and answers for Jev, independent of the wire format.

Jev answers three kinds of questions about a `state`:

- boolean (TypeSafe calls it "noul"): probability that the answer is yes
- choice: one option out of a named set
- score: a level on an ordered scale (index 0..n-1, fractional score allowed)

`decisions.client` maps these onto `langchain_typesafe`'s Noul / Choice / Score.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

# A string, or an object holding the question plus data it refers to by name
# (docs.typesafe.ai: "Use structure in the questions").
Instructions = str | dict[str, Any]


class BooleanCriteria(BaseModel):
    true: str
    false: str


class BooleanQuestion(BaseModel):
    type: Literal["boolean"] = "boolean"
    instructions: Instructions
    # Optional descriptions of what yes / no mean.
    criteria: BooleanCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: Instructions
    # option key -> description; the key is what comes back in the answer.
    options: dict[str, str] = Field(min_length=2, max_length=255)


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    instructions: Instructions
    # Ordered from lowest (index 0) to highest.
    levels: list[str] = Field(min_length=2, max_length=10)


Question = Annotated[BooleanQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class _AnswerBase(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    def is_confident(self, threshold: float) -> bool:
        """Unknown confidence counts as not confident — callers fall back safely."""
        return self.confidence is not None and self.confidence >= threshold


class BooleanAnswer(_AnswerBase):
    type: Literal["boolean"] = "boolean"

    @model_validator(mode="after")
    def _default_confidence(self) -> Self:
        # TypeSafe doesn't report confidence for yes/no. Use its choice
        # formula for two options: (2 * peak - 1), i.e. distance from a coin flip.
        if self.confidence is None:
            self.confidence = abs(2 * self.probability - 1)
        return self

    @property
    def yes(self) -> bool:
        return self.probability >= 0.5


class ChoiceAnswer(_AnswerBase):
    type: Literal["choice"] = "choice"
    value: str
    probabilities: dict[str, float] = Field(default_factory=dict)


class ScoreAnswer(_AnswerBase):
    type: Literal["score"] = "score"
    value: str  # description of the most likely level
    score: float  # probability-weighted level index
    probabilities: dict[str, float] = Field(default_factory=dict)
    legend: dict[str, str] = Field(default_factory=dict)


Answer = Annotated[BooleanAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Evaluation(BaseModel):
    """Result of one Jev call: one answer per question name."""

    answers: dict[str, Answer]
    model: str | None = None  # versioned model id that answered, when reported

    def boolean(self, name: str) -> BooleanAnswer:
        return self._get(name, BooleanAnswer)

    def choice(self, name: str) -> ChoiceAnswer:
        return self._get(name, ChoiceAnswer)

    def score(self, name: str) -> ScoreAnswer:
        return self._get(name, ScoreAnswer)

    def _get[T: _AnswerBase](self, name: str, kind: type[T]) -> T:
        answer = self.answers[name]
        if not isinstance(answer, kind):
            raise TypeError(f"answer {name!r} is {answer.type}, not {kind.__name__}")
        return answer
