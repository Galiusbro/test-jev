"""Regression proof: new tests must fail without the change.

Passing tests written by the same agent that wrote the code are weak evidence
— they may test nothing. AGENTS.md's definition of done asks for tests that
fail on the base commit, so this check restores the baseline version of every
changed source file, runs the changed tests, and puts the change back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jev_agent.workspace import CommandResult, Workspace

ProofStatus = Literal["proven", "not_proven", "no_tests", "skipped"]
_NO_TESTS_COLLECTED = 5  # pytest exit code


@dataclass(frozen=True)
class RegressionProof:
    status: ProofStatus
    tests: tuple[str, ...]
    detail: str
    result: CommandResult | None = None

    @property
    def acceptable(self) -> bool:
        return self.status in ("proven", "skipped")


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def prove_tests(ws: Workspace, changed: Sequence[str], test_command: str | None) -> RegressionProof:
    tests = tuple(p for p in changed if is_test_file(p))
    code = [p for p in changed if p.endswith(".py") and not is_test_file(p)]
    if test_command is None:
        return RegressionProof("skipped", tests, "project declares no `test` command")
    if not code:
        return RegressionProof("skipped", tests, "no source code changed")
    if not tests:
        return RegressionProof(
            "no_tests", tests, "source code changed but no test file was added or changed"
        )

    current = {p: (ws.root / p).read_text() for p in code}
    try:
        for path in code:
            base = ws.baseline(path)
            if base is None:
                (ws.root / path).unlink()
            else:
                (ws.root / path).write_text(base)
        result = ws.run(f"{test_command} {' '.join(tests)}", max_chars=3000)
    finally:
        for path, text in current.items():
            (ws.root / path).write_text(text)

    if result.exit_code == _NO_TESTS_COLLECTED:
        return RegressionProof("not_proven", tests, "no tests were collected", result)
    if result.ok:
        return RegressionProof(
            "not_proven",
            tests,
            "the changed tests also pass on the original code, so they don't test the change",
            result,
        )
    return RegressionProof("proven", tests, "the changed tests fail on the original code", result)
