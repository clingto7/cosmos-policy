"""Real CPU DCP serialization exercises resume/retention without a GPU model."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_policy.experiments.robot.bench2dex.checkpoints import (
    MARKER,
    mark_and_prune,
    require_checkpoint_contract,
    validate_checkpoint,
)
from cosmos_policy.experiments.robot.bench2dex.train_embodiment import recover_checkpoint


def save_checkpoint(root: Path, iteration: int):
    path = root / f"iter_{iteration:09}"
    for component, state in {
        "model": {"weight": torch.arange(4, dtype=torch.float32)},
        "optim": {"step": torch.tensor(iteration), "momentum": torch.ones(4)},
        "scheduler": {"last_epoch": iteration},
        "trainer": {"iteration": iteration, "grad_scaler": {}},
    }.items():
        dcp.save(state, checkpoint_id=path / component)
    (root / "latest_checkpoint.txt").write_text(path.name)
    return path


def test_only_two_verified_checkpoints_retained_and_partial_saves_untouched(tmp_path):
    manifest = tmp_path / "bundle.json"
    manifest.write_text('{"identity": "example"}')
    root = tmp_path / "checkpoints"
    for iteration in (5000, 10000, 15000):
        checkpoint = save_checkpoint(root, iteration)
        removed = mark_and_prune(root, iteration, manifest)
    assert removed == ["iter_000005000"]
    assert sorted(p.name for p in root.glob("iter_*")) == ["iter_000010000", "iter_000015000"]
    partial = root / "iter_000020000"
    partial.mkdir()
    require_checkpoint_contract(checkpoint, manifest)
    assert recover_checkpoint(root, manifest) == checkpoint
    assert partial.exists()
    # A final checkpoint is never removed, even if a later experiment is accidentally admitted.
    final = save_checkpoint(root, 50000)
    mark_and_prune(root, 50000, manifest)
    require_checkpoint_contract(final, manifest, final=True)
    with pytest.raises(ValueError, match="fixed 50k"):
        require_checkpoint_contract(checkpoint, manifest, final=True)


def test_truncated_shard_blocks_retention_and_falls_back(tmp_path):
    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}")
    root = tmp_path / "checkpoints"
    first = save_checkpoint(root, 5000)
    mark_and_prune(root, 5000, manifest)
    second = save_checkpoint(root, 10000)
    mark_and_prune(root, 10000, manifest)
    newest = save_checkpoint(root, 15000)
    # Truncate a referenced shard; merely checking .metadata and __0_0 existence is insufficient.
    shard = next((newest / "optim").glob("*.distcp"))
    shard.write_bytes(shard.read_bytes()[:10])
    with pytest.raises(ValueError, match="truncated"):
        mark_and_prune(root, 15000, manifest)
    assert first.exists() and second.exists() and not (newest / MARKER).exists()
    assert recover_checkpoint(root, manifest) == second
    assert (root / "latest_checkpoint.txt").read_text() == second.name
    assert newest.exists()


def test_foreign_manifest_symlinks_and_changed_metadata_rejected(tmp_path):
    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}")
    checkpoint = save_checkpoint(tmp_path / "checkpoints", 5000)
    mark_and_prune(checkpoint.parent, 5000, manifest)
    other = tmp_path / "other.json"
    other.write_text('{"different": true}')
    with pytest.raises(ValueError, match="different embodiment"):
        require_checkpoint_contract(checkpoint, other)
    link = tmp_path / "iter_000005000"
    link.symlink_to(checkpoint, target_is_directory=True)
    with pytest.raises(ValueError, match="real iter_"):
        validate_checkpoint(link)
    marker = json.loads((checkpoint / MARKER).read_text())
    marker["components"]["model"][".metadata"]["sha256"] = "changed"
    (checkpoint / MARKER).write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="changed after validation"):
        require_checkpoint_contract(checkpoint, manifest)


def test_completion_pointer_and_saved_iteration_must_agree(tmp_path):
    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}")
    checkpoint = save_checkpoint(tmp_path / "checkpoints", 5000)
    (checkpoint.parent / "latest_checkpoint.txt").write_text("iter_000000000")
    with pytest.raises(ValueError, match="pointer"):
        mark_and_prune(checkpoint.parent, 5000, manifest)
    checkpoint.rename(checkpoint.parent / "iter_000010000")
    with pytest.raises(ValueError, match="saved trainer iteration"):
        validate_checkpoint(checkpoint.parent / "iter_000010000")


def test_corrupted_metadata_falls_back_without_deleting_newer_artifact(tmp_path):
    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}")
    root = tmp_path / "checkpoints"
    first = save_checkpoint(root, 5000)
    mark_and_prune(root, 5000, manifest)
    latest = save_checkpoint(root, 10000)
    mark_and_prune(root, 10000, manifest)
    (latest / "model" / ".metadata").write_bytes(b"truncated")
    assert recover_checkpoint(root, manifest) == first
    assert latest.exists()


def _distributed_save_and_validate(rank, root, init_file, manifest):
    from datetime import timedelta

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        save_checkpoint(Path(root), 5000)
        mesh = init_device_mesh("cpu", (2,))
        weight = distribute_tensor(torch.arange(8, dtype=torch.float32), mesh, [Shard(0)])
        dcp.save({"weight": weight}, checkpoint_id=Path(root) / "iter_000005000" / "model")
        dist.barrier()
        if rank == 0:
            mark_and_prune(root, 5000, manifest)
        # Rank-zero validation must not enter collectives while rank one waits here.
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_retention_runs_on_rank_zero_with_two_cpu_ranks(tmp_path):
    import torch.multiprocessing as mp

    manifest = tmp_path / "bundle.json"
    manifest.write_text("{}")
    root = tmp_path / "checkpoints"
    mp.spawn(
        _distributed_save_and_validate, args=(str(root), str(tmp_path / "init"), str(manifest)), nprocs=2, join=True
    )
    require_checkpoint_contract(root / "iter_000005000", manifest)
    second_shard = root / "iter_000005000" / "model" / "__1_0.distcp"
    assert second_shard.is_file()
    second_shard.unlink()
    with pytest.raises(ValueError, match="missing, truncated"):
        require_checkpoint_contract(root / "iter_000005000", manifest)
