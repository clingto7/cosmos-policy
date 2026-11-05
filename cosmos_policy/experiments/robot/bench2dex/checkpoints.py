"""Validate local DCP checkpoints before admission, evaluation or retention.

Only checkpoints marked by this workflow can be removed. Partial/newer saves,
symlinks, foreign contracts and the final checkpoint are never removed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
from pathlib import Path

MARKER = "bench2dex-checkpoint.json"
COMPONENTS = ("model", "optim", "scheduler", "trainer")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def validate_checkpoint(path: str | Path) -> dict:
    """Check every DCP metadata reference, then deserialize the saved iteration.

    DCP metadata uses pickle; only call on trusted local training artifacts.
    This checks storage completeness, not GPU model-load equivalence.
    """
    import torch.distributed.checkpoint as dcp

    path = Path(path)
    match = re.fullmatch(r"iter_(\d{9})", path.name)
    if not match or path.is_symlink():
        raise ValueError("expected a real iter_NNNNNNNNN checkpoint directory")
    inventory = {}
    for component in COMPONENTS:
        directory = path / component
        if directory.is_symlink() or (directory / ".metadata").is_symlink():
            raise ValueError("checkpoint components must not be symlinks")
        reader = dcp.FileSystemReader(directory)
        try:
            metadata = reader.read_metadata()
        except (EOFError, pickle.UnpicklingError) as error:
            raise ValueError(f"unreadable checkpoint metadata: {directory}") from error
        if not metadata.state_dict_metadata or not metadata.storage_data:
            raise ValueError(f"empty checkpoint component: {component}")
        files = {".metadata": {"sha256": file_hash(directory / ".metadata")}}
        for storage in metadata.storage_data.values():
            relative = Path(storage.relative_path)
            shard = directory / relative
            if (
                relative.is_absolute()
                or len(relative.parts) != 1
                or shard.is_symlink()
                or not shard.is_file()
                or storage.offset < 0
                or storage.length <= 0
                or shard.stat().st_size < storage.offset + storage.length
            ):
                raise ValueError(f"missing, truncated or unsafe checkpoint shard: {shard}")
            files[str(relative)] = {"bytes": shard.stat().st_size}
        inventory[component] = files
    trainer = {"iteration": 0}
    # Retention runs on rank zero while other ranks continue training. Never
    # enter a distributed load collective from this validation-only read.
    try:
        dcp.load(trainer, storage_reader=dcp.FileSystemReader(path / "trainer"), no_dist=True)
    except dcp.CheckpointException as error:
        raise ValueError(f"unreadable checkpoint trainer state: {path}") from error
    if trainer["iteration"] != int(match[1]):
        raise ValueError("checkpoint directory and saved trainer iteration differ")
    return {"iteration": int(match[1]), "components": inventory}


def require_checkpoint_contract(checkpoint: str | Path, manifest_path: str | Path, *, final=False) -> dict:
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint / MARKER).read_text())
    if marker["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("checkpoint belongs to a different embodiment manifest")
    actual = validate_checkpoint(checkpoint)
    if any(marker[key] != actual[key] for key in ("iteration", "components")):
        raise ValueError("checkpoint changed after validation")
    if final and actual["iteration"] != 50_000:
        raise ValueError("per-embodiment final evaluation requires the fixed 50k checkpoint")
    return marker


def mark_and_prune(root: str | Path, iteration: int, manifest_path: str | Path) -> list[str]:
    root, manifest_path = Path(root), Path(manifest_path)
    name = f"iter_{iteration:09}"
    if (root / "latest_checkpoint.txt").read_text().strip() != name:
        raise ValueError("latest pointer has not committed this checkpoint")
    checkpoint = root / name
    marker = validate_checkpoint(checkpoint)
    marker["manifest_sha256"] = file_hash(manifest_path)
    atomic_json(checkpoint / MARKER, marker)
    candidates = []
    for path in root.iterdir():
        match = re.fullmatch(r"iter_(\d{9})", path.name)
        if not match or path.is_symlink() or not (path / MARKER).is_file() or int(match[1]) > iteration:
            continue
        try:
            require_checkpoint_contract(path, manifest_path)
        except (ValueError, OSError, KeyError):
            # Damaged or foreign artifacts are not retention candidates.
            continue
        else:
            candidates.append(path)
    candidates.sort(key=lambda path: path.name)
    # Revalidate both recovery anchors before deleting anything.
    for path in candidates[-2:]:
        require_checkpoint_contract(path, manifest_path)
    removed = []
    for path in candidates[:-2]:
        if path.name == "iter_000050000":
            continue
        require_checkpoint_contract(path, manifest_path)
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
