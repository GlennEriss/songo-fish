"""Safe, observable Git bootstrap for disposable Colab source checkouts."""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


class GitBootstrapError(RuntimeError):
    pass


@dataclass(frozen=True)
class BootstrapResult:
    repository: Path
    commit: str
    reused: bool
    dirty_repository_preserved: Path | None


def run_visible(command: Sequence[str], *, runner: Callable = subprocess.run) -> str:
    """Run a critical Git command while always exposing stdout and stderr."""
    result = runner(list(command), text=True, capture_output=True)
    print(f"$ {' '.join(command)}")
    print(f"exit={result.returncode}")
    if result.stdout:
        print("stdout:\n" + result.stdout.rstrip())
    if result.stderr:
        print("stderr:\n" + result.stderr.rstrip())
    if result.returncode:
        raise GitBootstrapError(
            f"Git command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()


def _git(repo: Path, *args: str, runner: Callable = subprocess.run) -> str:
    return run_visible(["git", "-C", str(repo), *args], runner=runner)


def bootstrap_repository(*, url: str, branch: str, preferred: Path,
                         runner_path: str, pinned_commit: str | None = None,
                         runner: Callable = subprocess.run,
                         clock: Callable[[], float] = time.time) -> BootstrapResult:
    remote = run_visible(["git", "ls-remote", "--exit-code", "--heads", url, f"refs/heads/{branch}"], runner=runner)
    fields = remote.split()
    if len(fields) < 2:
        raise GitBootstrapError(f"remote branch has no commit: {branch}")
    remote_commit = fields[0]
    target = pinned_commit or remote_commit
    dirty = None
    repo = preferred
    reusable = (repo / ".git").is_dir()
    if reusable:
        status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all", runner=runner)
        if status:
            dirty = repo
            repo = preferred.with_name(f"{preferred.name}-session-{int(clock())}")
            reusable = False
    fresh_clone = not reusable
    if fresh_clone:
        if repo.exists():
            raise GitBootstrapError(f"clean session destination already exists: {repo}")
        run_visible(["git", "clone", "--no-checkout", "--origin", "origin", url, str(repo)], runner=runner)
    _git(repo, "fetch", "--no-tags", "origin", f"refs/heads/{branch}:refs/remotes/origin/{branch}", runner=runner)
    if pinned_commit:
        present = runner(["git", "-C", str(repo), "cat-file", "-e", f"{target}^{{commit}}"], text=True, capture_output=True)
        if present.returncode:
            _git(repo, "fetch", "--no-tags", "origin", target, runner=runner)
    _git(repo, "cat-file", "-e", f"{target}^{{commit}}", runner=runner)
    _git(repo, "cat-file", "-e", f"{target}:{runner_path}", runner=runner)
    current = _git(repo, "rev-parse", "HEAD", runner=runner) if reusable else None
    if fresh_clone or current != target:
        _git(repo, "checkout", "--detach", target, runner=runner)
    exact = _git(repo, "rev-parse", "HEAD", runner=runner)
    if exact != target:
        raise GitBootstrapError(f"checkout mismatch: expected {target}, got {exact}")
    return BootstrapResult(repo, exact, reusable, dirty)
