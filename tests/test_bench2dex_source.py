"""Packaged Cosmos source must not inherit the parent Bench2Dex Git identity."""

import subprocess
from pathlib import Path

from cosmos_policy.experiments.robot.bench2dex.source import exact_git_state, source_hash


def test_packaged_source_uses_own_fingerprint_without_parent_commit(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    policy = tmp_path / "policy" / "cosmos_policy"
    package = policy / "cosmos_policy"
    package.mkdir(parents=True)
    (package / "adapter.py").write_text("VALUE = 1\n")
    assert exact_git_state(policy) == (None, None)
    before = source_hash(policy)
    (package / "adapter.py").write_text("VALUE = 2\n")
    assert source_hash(policy) != before


def test_exact_worktree_reports_its_revision(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    package = tmp_path / "cosmos_policy"
    package.mkdir()
    (package / "adapter.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "cosmos_policy/adapter.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "init"],
        check=True,
    )
    revision, dirty = exact_git_state(tmp_path)
    assert len(revision) == 40 and dirty is False
