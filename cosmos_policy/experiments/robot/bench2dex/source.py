"""Identify Cosmos source without mistaking a containing Bench2Dex repo for it."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def source_hash(repo: Path) -> str:
    package = Path(repo) / "cosmos_policy"
    paths = sorted(package.rglob("*.py"))
    if not paths:
        raise ValueError(f"Cosmos source package missing from {repo}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def exact_git_state(repo: Path) -> tuple[str | None, bool | None]:
    """Return Git identity only when repo itself is the worktree root."""
    repo = Path(repo).resolve()
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=repo, capture_output=True, text=True
    )
    if root.returncode or Path(root.stdout.strip()).resolve() != repo:
        return None, None
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
    if revision.returncode or status.returncode:
        raise RuntimeError("cannot inspect Cosmos Git worktree")
    return revision.stdout.strip(), bool(status.stdout.strip())
