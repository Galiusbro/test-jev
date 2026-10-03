"""TypeSafe wire format (docs.typesafe.ai/api).

`POST /v1/systemone`. Question types are `noul|choice|score` and every option,
level or yes/no description goes in `criteria`. Answers are normalized into the
shape `Evaluation` validates.
"""

from __future__ import annotations

from typing import Any

from jev_agent.decisions.types import BooleanQuestion, ChoiceQuestion, Question, ScoreQuestion

JSON = dict[str, Any]


def encode_question(question: Question) -> JSON:
    match question:
        case BooleanQuestion():
            out: JSON = {"type": "noul", "instructions": question.instructions}
            if question.criteria is not None:
                out["criteria"] = question.criteria.model_dump()
            return out
        case ChoiceQuestion():
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": question.options,
            }
        case ScoreQuestion():
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": question.levels,
            }


def decode(body: JSON) -> JSON:
    return {
        "model": body.get("model"),
        "answers": {name: _answer(a) for name, a in body["answers"].items()},
    }


def _answer(a: JSON) -> JSON:
    kind = a.get("type")
    if kind == "noul":
        return {"type": "boolean", "probability": a["noul"]}
    if kind == "choice":
        probs = a.get("probabilities") or {}
        return {
            "type": "choice",
            "value": a["choice"],
            "probability": probs.get(a["choice"], 0.0),
            "confidence": a.get("confidence"),
            "probabilities": probs,
        }
    if kind == "score":
        probs = a["probabilities"]
        legend = a.get("legend") or {}
        peak = max(probs, key=lambda k: probs[k])
        return {
            "type": "score",
            "score": a["score"],
            "value": legend.get(peak, peak),
            "probability": probs[peak],
            "confidence": a.get("confidence"),
            "probabilities": probs,
            "legend": legend,
        }
    raise ValueError(f"unknown answer type {kind!r}")
