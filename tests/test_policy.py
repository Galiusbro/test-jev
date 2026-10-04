from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jev_agent.agents import Plan
from jev_agent.policy import ApprovalRequest, Policy, Rule, parse_rules
from jev_agent.tools import make_tools
from jev_agent.workspace import Workspace

DEMO_RULES = parse_rules((Path(__file__).parents[1] / "demo-api" / "AGENTS.md").read_text())

OLD_TESTS = "def test_a():\n    pass\n\n\nclass TestB:\n    def test_c(self):\n        pass\n"


class Approver:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.requests: list[ApprovalRequest] = []

    def __call__(self, request: ApprovalRequest) -> bool:
        self.requests.append(request)
        return self.answer


def policy(answer: bool = False, baseline: dict[str, str] | None = None) -> tuple[Policy, Approver]:
    approver = Approver(answer)
    files = baseline or {}
    return Policy(DEMO_RULES, approver, lambda p: files.get(p)), approver


def plan(**overrides: Any) -> Plan:
    data: dict[str, Any] = {
        "summary": "s",
        "files_to_change": ["src/demo_api/auth.py"],
        "steps": ["x"],
        "tests": ["y"],
        "risk": "low",
        "public_api_change": False,
        "db_migration": False,
    }
    return Plan(**(data | overrides))


def test_demo_rules_parse_into_enforced_and_advisory() -> None:
    by_text = {r.text: r for r in DEMO_RULES}
    assert by_text["Modify application code under `src/demo_api/`"].paths == ("src/demo_api/",)
    assert by_text["Update documentation under `docs/` and `README.md`"].paths == (
        "docs/",
        "README.md",
    )
    public_api = next(r for r in DEMO_RULES if r.text.startswith("Public API"))
    assert (public_api.kind, public_api.trigger) == ("approval", "public_api_change")
    tests_rule = next(r for r in DEMO_RULES if "existing tests" in r.text)
    assert (tests_rule.kind, tests_rule.trigger) == ("forbidden", "existing_tests")
    advisory = [r.text for r in DEMO_RULES if not r.enforced]
    assert "Disabling lint, type checks or tests" in advisory
    assert {r.kind for r in DEMO_RULES} == {"allowed", "approval", "forbidden"}


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("src/demo_api/", "src/demo_api/auth.py", True),
        ("src/demo_api/", "src/other.py", False),
        ("src/demo_api/db.py", "src/demo_api/db.py", True),
        ("tests/*.py", "tests/test_api.py", True),
        (".env", ".env", True),
        (".env", "config/.env", True),
        (".env", ".env.local", True),
        (".env", ".envrc_not", False),
        ("AGENTS.md", "AGENTS.md", True),
    ],
)
def test_path_matching(pattern: str, path: str, expected: bool) -> None:
    assert Rule("forbidden", "r", (pattern,)).matches(path) is expected


def test_forbidden_writes_are_denied_without_asking() -> None:
    p, approver = policy(answer=True)
    assert "denied by policy" in (p.check_write(".env", "X=1") or "")
    assert "denied by policy" in (p.check_write("AGENTS.md", "x") or "")
    assert approver.requests == []
    assert [d.verdict for d in p.audit] == ["deny", "deny"]


def test_allowed_scope_needs_no_approval() -> None:
    p, approver = policy()
    assert p.check_write("src/demo_api/auth.py", "x = 1\n") is None
    assert p.check_write("docs/api.md", "doc") is None
    assert approver.requests == []


def test_approval_paths_ask_once_and_remember() -> None:
    p, approver = policy(answer=True)
    assert p.check_write("src/demo_api/db.py", "x") is None
    assert p.check_write("src/demo_api/db.py", "y") is None
    assert len(approver.requests) == 1
    assert "Database schema" in approver.requests[0].reasons[0]


def test_denied_approval_blocks_write() -> None:
    p, _ = policy(answer=False)
    error = p.check_write("pyproject.toml", "x") or ""
    assert "approval was not given" in error
    assert p.audit[-1].verdict == "rejected"


def test_outside_scope_requires_approval() -> None:
    p, approver = policy(answer=False)
    assert p.check_write("scripts/deploy.sh", "x") is not None
    assert approver.requests[0].reasons == ("outside the allowed scope",)


def test_removing_existing_tests_is_denied_but_adding_is_fine() -> None:
    p, _ = policy(baseline={"tests/test_api.py": OLD_TESTS})
    kept_and_added = OLD_TESTS + "\n\ndef test_new():\n    pass\n"
    assert p.check_write("tests/test_api.py", kept_and_added) is None
    error = p.check_write("tests/test_api.py", "def test_new():\n    pass\n") or ""
    assert "would remove existing tests: TestB.test_c, test_a" in error
    # brand-new test files have no baseline to protect
    assert p.check_write("tests/test_rate_limit.py", "def test_x():\n    pass\n") is None


def test_reads_of_forbidden_paths_are_denied() -> None:
    p, _ = policy()
    assert p.check_read(".env") is not None
    assert p.check_read("src/demo_api/auth.py") is None
    assert not p.readable(".env.production")


def test_plan_within_scope_is_allowed_silently() -> None:
    p, approver = policy()
    assert p.check_plan(plan()).approved
    assert approver.requests == []


def test_plan_public_api_change_needs_approval() -> None:
    p, approver = policy(answer=True)
    verdict = p.check_plan(plan(public_api_change=True))
    assert verdict.approved
    assert "public api change" in approver.requests[0].reasons[0]
    assert "src/demo_api/auth.py" in p.approved_paths  # tool-time writes won't re-ask


def test_plan_rejected_when_approval_denied() -> None:
    p, _ = policy(answer=False)
    verdict = p.check_plan(plan(files_to_change=["src/demo_api/db.py"], db_migration=True))
    assert not verdict.approved
    assert any("Database schema" in r for r in verdict.reasons)
    assert p.audit[-1].verdict == "rejected"


def test_plan_touching_forbidden_file_rejected_without_asking() -> None:
    p, approver = policy(answer=True)
    verdict = p.check_plan(plan(files_to_create=[".env"]))
    assert not verdict.approved and approver.requests == []


def test_no_allowed_section_means_unrestricted_scope() -> None:
    p = Policy(parse_rules("## Forbidden\n\n- Touch `secrets/`\n"))
    assert p.check_write("anything/at/all.py", "x = 1\n") is None
    assert p.check_plan(plan(files_to_change=["x.py"])).approved
    assert p.check_write("secrets/key.pem", "x") is not None


def test_tools_enforce_policy_with_normalized_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src" / "demo_api").mkdir(parents=True)
    (repo / "src" / "demo_api" / "auth.py").write_text("x = 1\n")
    (repo / ".env").write_text("SECRET=1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_api.py").write_text(OLD_TESTS)
    ws = Workspace.create(repo, tmp_path / "run")
    p = Policy(DEMO_RULES, Approver(False), ws.baseline)
    tools = {t.name: t for t in make_tools(ws, writable=True, policy=p)}

    assert ".env" not in tools["list_files"].invoke({})
    assert "SECRET" not in tools["search"].invoke({"pattern": "SECRET"})
    for sneaky in (".env", "./.env", "src/../.env"):
        assert tools["read_file"].invoke({"path": sneaky}).startswith("ERROR: denied by policy")
    assert (
        tools["write_file"]
        .invoke({"path": "src/../AGENTS.md", "content": "x"})
        .startswith("ERROR: denied by policy")
    )
    deleted = tools["write_file"].invoke({"path": "tests/test_api.py", "content": "x = 1\n"})
    assert "would remove existing tests" in deleted
    assert (repo / "tests" / "test_api.py").read_text() == OLD_TESTS
    assert (
        tools["replace_in_file"].invoke(
            {"path": "src/demo_api/auth.py", "old": "x = 1", "new": "x = 2"}
        )
        == "edited src/demo_api/auth.py"
    )


def test_review_grounding() -> None:
    from jev_agent.agents import Finding, Review

    diff = (
        "+    if limiter.is_blocked(ip):\n"
        "+        raise HTTPException(429, 'Too many')\n"
        "     return token\n"
    )

    def f(severity: str, evidence: str) -> Finding:
        return Finding(severity=severity, file="a.py", issue="x", evidence=evidence)

    review = Review(
        summary="s",
        findings=[
            f("major", "if limiter.is_blocked(ip):"),  # quoted, whitespace differs
            f("blocker", "+        raise HTTPException(429, 'Too many')"),  # with diff marker
            f("major", "limiter.reset(ip)"),  # invented
            f("major", ""),  # no evidence
            f("minor", ""),  # minor needs none
        ],
    ).grounded(diff)
    assert [x.severity for x in review.findings] == ["major", "blocker", "minor", "minor", "minor"]
    assert review.findings[2].issue == "[unverified] x"
    assert review.findings[4].issue == "x"
    assert not review.approved
