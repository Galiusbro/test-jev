"""Policy engine: the target repo's AGENTS.md rules, enforced in code.

The model is not the security boundary. Rules under `## Allowed`,
`## Approval required` and `## Forbidden` become checks that run on every tool
call and on the plan, independent of what any prompt says.

A rule is *enforced* when it names paths (in backticks) or matches a known
trigger; otherwise it is *advisory* — still shown to the agents and the
reviewer, but not machine-checked. Every decision lands in the audit log.
"""

from __future__ import annotations

import ast
import fnmatch
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from jev_agent.agents import Plan

RuleKind = Literal["allowed", "approval", "forbidden"]
Trigger = Literal["public_api_change", "db_migration", "existing_tests"]

_SECTIONS: dict[str, RuleKind] = {
    "allowed": "allowed",
    "approval required": "approval",
    "forbidden": "forbidden",
}
_BACKTICK = re.compile(r"`([^`]+)`")
# Phrases that map a prose rule onto something code can check.
_TRIGGERS: list[tuple[re.Pattern[str], Trigger]] = [
    (re.compile(r"public api", re.I), "public_api_change"),
    (re.compile(r"schema|migration", re.I), "db_migration"),
    (re.compile(r"existing tests", re.I), "existing_tests"),
]


@dataclass(frozen=True)
class Rule:
    kind: RuleKind
    text: str
    paths: tuple[str, ...] = ()
    trigger: Trigger | None = None

    @property
    def enforced(self) -> bool:
        return bool(self.paths or self.trigger)

    def matches(self, path: str) -> bool:
        return any(_path_matches(pattern, path) for pattern in self.paths)


def _path_matches(pattern: str, path: str) -> bool:
    if pattern.endswith("/"):
        return path.startswith(pattern)
    if "/" in pattern:
        return path == pattern or fnmatch.fnmatch(path, pattern)
    name = path.rsplit("/", 1)[-1]  # bare names match anywhere: `.env`, `AGENTS.md`
    return name == pattern or fnmatch.fnmatch(name, pattern) or name.startswith(pattern + ".")


def _looks_like_path(token: str) -> bool:
    return "/" in token or "." in token or "*" in token


def parse_rules(agents_md: str) -> list[Rule]:
    rules: list[Rule] = []
    kind: RuleKind | None = None
    for line in agents_md.splitlines():
        if line.startswith("## "):
            kind = _SECTIONS.get(line[3:].strip().lower())
            continue
        text = line.strip()
        if kind is None or not text.startswith("- "):
            continue
        text = text[2:].strip()
        paths = tuple(t for t in _BACKTICK.findall(text) if _looks_like_path(t))
        trigger = next((t for pattern, t in _TRIGGERS if pattern.search(text)), None)
        rules.append(Rule(kind, text, paths, trigger))
    return rules


Verdict = Literal["allow", "deny", "approved", "rejected"]


@dataclass(frozen=True)
class Decision:
    action: Literal["read", "write", "plan"]
    target: str
    verdict: Verdict
    reason: str


@dataclass(frozen=True)
class ApprovalRequest:
    action: Literal["write", "plan"]
    targets: tuple[str, ...]
    reasons: tuple[str, ...]
    summary: str = ""


Approver = Callable[[ApprovalRequest], bool]


def deny_all(_request: ApprovalRequest) -> bool:
    return False


@dataclass
class PlanVerdict:
    approved: bool
    reasons: list[str]


@dataclass
class Policy:
    rules: list[Rule]
    approver: Approver = deny_all
    # Repo-relative path -> file content at the baseline commit (None if new).
    baseline: Callable[[str], str | None] = lambda _path: None
    # Repo-relative path -> current content (None if missing); for the semantic gate.
    current: Callable[[str], str | None] | None = None
    audit: list[Decision] = field(default_factory=list)
    approved_paths: set[str] = field(default_factory=set)
    # Semantic check for writes that pass the deterministic rules (Jev in M5):
    # (path, content before, content after) -> approval reason, or None.
    semantic_gate: Callable[[str, str | None, str], str | None] | None = None

    @property
    def advisory(self) -> list[str]:
        """Rules code cannot check — candidates for the semantic gate."""
        return [r.text for r in self.rules if not r.enforced and r.kind != "allowed"]

    def _of(self, kind: RuleKind) -> list[Rule]:
        return [r for r in self.rules if r.kind == kind]

    def _first_match(self, kind: RuleKind, path: str) -> Rule | None:
        return next((r for r in self._of(kind) if r.matches(path)), None)

    def _log(
        self, action: Literal["read", "write", "plan"], target: str, verdict: Verdict, reason: str
    ) -> None:
        self.audit.append(Decision(action, target, verdict, reason))

    # --- reads ---------------------------------------------------------------

    def readable(self, path: str) -> bool:
        return self._first_match("forbidden", path) is None

    def check_read(self, path: str) -> str | None:
        """None if the read is allowed, else the error to return to the agent."""
        rule = self._first_match("forbidden", path)
        if rule is None:
            return None
        self._log("read", path, "deny", rule.text)
        return f"ERROR: denied by policy — {rule.text}"

    # --- writes --------------------------------------------------------------

    def check_write(self, path: str, new_content: str) -> str | None:
        """None if the write may proceed, else the error to return to the agent."""
        if rule := self._first_match("forbidden", path):
            self._log("write", path, "deny", rule.text)
            return f"ERROR: denied by policy — {rule.text}"
        if removed := self._removed_tests(path, new_content):
            reason = f"would remove existing tests: {', '.join(sorted(removed))}"
            self._log("write", path, "deny", reason)
            return f"ERROR: denied by policy — {reason}. Existing tests must be kept."
        if self.semantic_gate and (
            flag := self.semantic_gate(path, self._current(path), new_content)
        ):
            return self._ask("write", path, flag)
        if path in self.approved_paths:
            self._log("write", path, "allow", "previously approved")
            return None
        if rule := self._first_match("approval", path):
            return self._ask("write", path, f"requires approval: {rule.text}")
        if rule := self._first_match("allowed", path):
            self._log("write", path, "allow", rule.text)
            return None
        if not self._of("allowed"):
            self._log("write", path, "allow", "no allowed scope defined")
            return None
        return self._ask("write", path, "outside the allowed scope")

    def _current(self, path: str) -> str | None:
        return self.current(path) if self.current else self.baseline(path)

    def _ask(self, action: Literal["write"], path: str, reason: str) -> str | None:
        if self.approver(ApprovalRequest(action, (path,), (reason,))):
            if not reason.startswith("Jev"):  # semantic flags are per edit, not per path
                self.approved_paths.add(path)
            self._log(action, path, "approved", reason)
            return None
        self._log(action, path, "rejected", reason)
        return f"ERROR: denied by policy — {path} {reason}, and approval was not given."

    def _removed_tests(self, path: str, new_content: str) -> set[str]:
        guard = any(r.trigger == "existing_tests" for r in self._of("forbidden"))
        if not guard or not path.endswith(".py") or not path.rsplit("/", 1)[-1].startswith("test"):
            return set()
        before = self.baseline(path)
        if before is None:
            return set()
        return _test_names(before) - _test_names(new_content)

    # --- plan ----------------------------------------------------------------

    def check_plan(self, plan: Plan, extra_reasons: Sequence[str] = ()) -> PlanVerdict:
        files = [*plan.files_to_change, *plan.files_to_create]
        forbidden = [
            f"{path}: {rule.text}"
            for path in files
            if (rule := self._first_match("forbidden", path))
        ]
        if forbidden:
            for reason in forbidden:
                self._log("plan", reason.split(":")[0], "deny", reason)
            return PlanVerdict(False, forbidden)

        reasons: list[str] = []
        for path in files:
            if rule := self._first_match("approval", path):
                reasons.append(f"{path}: {rule.text}")
            elif self._of("allowed") and not self._first_match("allowed", path):
                reasons.append(f"{path}: outside the allowed scope")
        for flag, trigger in (
            (plan.public_api_change, "public_api_change"),
            (plan.db_migration, "db_migration"),
        ):
            rule = next((r for r in self._of("approval") if r.trigger == trigger), None)
            if flag and rule:
                reasons.append(f"plan declares {trigger.replace('_', ' ')}: {rule.text}")

        reasons.extend(extra_reasons)
        if not reasons:
            self._log("plan", "plan", "allow", "within allowed scope")
            return PlanVerdict(True, [])
        request = ApprovalRequest("plan", tuple(files), tuple(reasons), plan.summary)
        if self.approver(request):
            self.approved_paths.update(files)
            self._log("plan", "plan", "approved", "; ".join(reasons))
            return PlanVerdict(True, reasons)
        self._log("plan", "plan", "rejected", "; ".join(reasons))
        return PlanVerdict(False, reasons)


def _test_names(source: str) -> set[str]:
    """Names of test functions (module level and in classes) in a test module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test"
        ):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.update(
                f"{node.name}.{item.name}"
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("test")
            )
    return names
