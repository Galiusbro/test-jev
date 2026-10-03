"""Isolated working copy of the target repository.

The agent never touches the original repo. Each run copies it to
`runs/<run_id>/repo`, commits a git baseline, and all reads, writes, commands
and the final diff happen inside that copy.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "*.pyc"
)


def _clean_env() -> dict[str, str]:
    """Our own virtualenv must not leak into commands run in the target repo."""
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    return env


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    command: str
    exit_code: int
    output: str  # stdout + stderr, tail-truncated

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @classmethod
    def create(cls, source: Path, run_dir: Path) -> Workspace:
        repo = run_dir / "repo"
        if repo.exists():
            raise WorkspaceError(f"{repo} already exists")
        shutil.copytree(source, repo, ignore=_IGNORE)
        ws = cls(repo)
        ws._git("init", "-q", "-b", "base")
        ws._git("add", "-A")
        ws._git("commit", "-q", "-m", "baseline", "--no-gpg-sign")
        return ws

    def resolve(self, relative: str) -> Path:
        """Path inside the workspace; anything escaping it is rejected."""
        path = (self.root / relative).resolve()
        if path != self.root and self.root not in path.parents:
            raise WorkspaceError(f"path escapes the workspace: {relative}")
        if ".git" in path.relative_to(self.root).parts:
            raise WorkspaceError(f"path is inside .git: {relative}")
        return path

    def files(self) -> list[str]:
        """Tracked + new files, as the agent should see the repo."""
        out = self._git("ls-files", "--cached", "--others", "--exclude-standard")
        return sorted(line for line in out.splitlines() if line)

    def diff(self) -> str:
        self._git("add", "-A")
        return self._git("diff", "--cached", "base")

    def changed_files(self) -> list[str]:
        self._git("add", "-A")
        out = self._git("diff", "--cached", "--name-only", "base")
        return [line for line in out.splitlines() if line]

    def run(self, command: str, timeout_s: float = 300.0, max_chars: int = 6000) -> CommandResult:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            return CommandResult(command, 124, f"timed out after {timeout_s:.0f}s")
        output = (proc.stdout + proc.stderr).strip()
        if len(output) > max_chars:
            output = "…(truncated)…\n" + output[-max_chars:]
        return CommandResult(command, proc.returncode, output)

    def _git(self, *args: str) -> str:
        proc = subprocess.run(
            ["git", "-c", "user.name=jev-agent", "-c", "user.email=agent@localhost", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise WorkspaceError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout
