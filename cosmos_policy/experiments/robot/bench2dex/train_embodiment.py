"""Plan or execute a budgeted embodiment run with frozen inputs and safe resume."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from cosmos_policy.datasets.bench2dex_embodiment import load_embedding, read_bundle
from cosmos_policy.experiments.robot.bench2dex.budget import Budget
from cosmos_policy.experiments.robot.bench2dex.checkpoints import (
    MARKER,
    atomic_json,
    file_hash,
    require_checkpoint_contract,
)

PRIVATE_ROOT = Path("/inspire/hdd2/project/roboticsystem2/ky26063")


def check_output_path(path: Path):
    resolved = path.resolve()
    if str(resolved).startswith("/inspire/") and not resolved.is_relative_to(PRIVATE_ROOT):
        raise ValueError(f"training output must resolve under {PRIVATE_ROOT}")


def source_hash(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "cosmos_policy").rglob("*.py")):
        digest.update(str(path.relative_to(repo)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_versions():
    versions = {"python": sys.version}
    for name in ("torch", "numpy", "h5py", "omegaconf", "megatron-core", "transformer-engine"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def plan_run(args) -> tuple[list[str], dict, Path]:
    bundle, profile, _ = read_bundle(args.manifest, str(args.robot_map))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        raise ValueError("run name must contain only letters, digits, underscores, dots or hyphens")
    if not args.base_model.is_file() or args.base_model.suffix != ".pt" or not args.tokenizer.is_file():
        raise ValueError("existing Video2World .pt base weights and tokenizer are required")
    for task in bundle["tasks"]:
        load_embedding(task, args.manifest.parent)
    check_output_path(args.output)
    run_dir = args.output.resolve() / "cosmos_policy" / "bench2dex_embodiment" / args.run_name
    check_output_path(run_dir)
    max_iter = getattr(args, "smoke_steps", 4) if args.smoke else getattr(args, "max_steps", 32000)
    save_iter = 2 if args.smoke else 5000
    budget = Budget(len(bundle["tasks"]), max_iter)
    expected = getattr(args, "expected_tasks", None)
    if expected is not None and expected != budget.task_count:
        raise ValueError("script task count differs from the complete embodiment bundle")
    experiment = f"cosmos_predict2_2b_480p_bench2dex_{profile.robot_key}"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={budget.world_size}",
        "-m",
        "cosmos_policy.scripts.train",
        "--config=cosmos_policy/config/config.py",
        "--",
        f"experiment={experiment}",
        f"job.name={args.run_name}",
        "job.project=cosmos_policy",
        "job.group=bench2dex_embodiment",
        f"trainer.max_iter={max_iter}",
        "trainer.grad_accum_iter=2",
        f"dataloader_train.batch_size={budget.microbatch}",
        f"checkpoint.save_iter={save_iter}",
        "checkpoint.dcp_async_mode_enabled=false",
    ]
    identity = {
        "schema": "bench2dex-embodiment-run-v2",
        "mode": "smoke" if args.smoke else "train",
        "manifest_sha256": file_hash(args.manifest),
        "robot_map_sha256": file_hash(args.robot_map),
        "base_model_sha256": file_hash(args.base_model),
        "tokenizer_sha256": file_hash(args.tokenizer),
        "source_sha256": source_hash(args.repo),
        "environment": environment_versions(),
        "robot_key": profile.robot_key,
        "task_ids": [t["task_id"] for t in bundle["tasks"]],
        **budget.to_dict(),
        "scheduler": budget.scheduler(),
        "max_iter": max_iter,
        "save_iter": save_iter,
        "sampler_seed": 0,
        "command": command,
    }
    return command, identity, run_dir


def recover_checkpoint(root: Path, manifest: Path):
    """Fall back to the newest previously validated checkpoint after interruption."""
    if not root.exists():
        return None
    valid = []
    for path in sorted(root.glob("iter_*"), reverse=True):
        if path.is_symlink() or not (path / MARKER).is_file():
            continue
        try:
            require_checkpoint_contract(path, manifest)
        except (ValueError, OSError, KeyError, RuntimeError):
            continue
        valid.append(path)
        break
    if not valid:
        if (root / "latest_checkpoint.txt").exists() or any(root.glob("iter_*")):
            raise ValueError("no validated recovery checkpoint; inspect partial saves before restarting")
        return None
    temporary = root / "latest_checkpoint.txt.tmp"
    temporary.write_text(valid[0].name)
    temporary.replace(root / "latest_checkpoint.txt")
    return valid[0]


def check_training_devices(smoke: bool, world_size: int):
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise ValueError(f"expose exactly {world_size} CUDA GPUs using CUDA_VISIBLE_DEVICES")
    if not smoke and any("H100" not in torch.cuda.get_device_name(rank) for rank in range(world_size)):
        raise ValueError("the fixed-budget production recipe requires H100 GPUs")


def execute_run(args, command, identity, run_dir):
    check_training_devices(args.smoke, identity["world_size"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "launch.lock").open("a") as lock:
        # The child inherits this lock, so killing the launcher cannot admit a duplicate run.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = run_dir / "run-identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("existing run has different data, weights, code or configuration; use a new run name")
        else:
            if (run_dir / "checkpoints").exists():
                raise ValueError("cannot adopt checkpoints from an unidentified run")
            atomic_json(identity_path, identity)
        checkpoint = recover_checkpoint(run_dir / "checkpoints", args.manifest)
        complete = checkpoint and int(checkpoint.name.removeprefix("iter_")) >= identity["max_iter"]
        if args.verify_restore and not complete:
            raise ValueError("--verify-restore requires a completed run; it does not advance training")
        if complete and not args.verify_restore:
            print(f"Run already complete: {checkpoint}")
            return
        bundle, profile, _ = read_bundle(args.manifest, str(args.robot_map))
        for task in bundle["tasks"]:
            for entry in task["audit"]["entries"]:
                source = Path(entry["source"]).resolve()
                check_output_path(source)
                if file_hash(source) != entry["sha256"]:
                    raise ValueError(f"demonstration changed since audit: {source}")
        from cosmos_policy.experiments.robot.bench2dex.monitoring import dataset_summary

        summary = dataset_summary(bundle, profile, Budget(len(bundle["tasks"]), identity["max_iter"]))
        summary_path = run_dir / "dataset-summary.json"
        if summary_path.exists() and json.loads(summary_path.read_text()) != json.loads(json.dumps(summary)):
            raise ValueError("dataset monitoring summary changed since launch")
        atomic_json(summary_path, summary)
        atomic_json(run_dir / "manifest.snapshot.json", bundle)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=args.repo, capture_output=True, text=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=args.repo, capture_output=True, text=True)
        atomic_json(run_dir / "deployment.json", {
            "commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "source_sha256": identity["source_sha256"],
        })
        environment = os.environ.copy()
        for key in list(environment):
            if key.startswith("BENCH2DEX_"):
                del environment[key]
        environment.update(
            BENCH2DEX_EMBODIMENT_MANIFEST=str(args.manifest),
            BENCH2DEX_ROBOT_MAP=str(args.robot_map),
            BENCH2DEX_MAX_STEPS=str(identity["max_iter"]),
            COSMOS_POLICY_PRETRAINED_CHECKPOINT=str(args.base_model),
            COSMOS_POLICY_TOKENIZER_CHECKPOINT=str(args.tokenizer),
            IMAGINAIRE_OUTPUT_ROOT=str(args.output),
            COSMOS_INTERNAL="1",
            PYTHONUNBUFFERED="1",
        )
        with (run_dir / "launcher.log").open("a") as log:
            result = subprocess.run(
                command, cwd=args.repo, env=environment, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),)
            )
        if result.returncode:
            raise RuntimeError(f"training exited with {result.returncode}; see {run_dir / 'launcher.log'}")
        require_checkpoint_contract(run_dir / "checkpoints" / f"iter_{identity['max_iter']:09}", args.manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "robot-map", "base-model", "tokenizer", "output"):
        parser.add_argument("--" + name, type=lambda p: Path(p).expanduser().resolve(), required=True)
    parser.add_argument("--repo", type=lambda p: Path(p).resolve(), default=Path(__file__).resolve().parents[4])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, default=32000)
    parser.add_argument(
        "--expected-tasks", type=int, choices=(1, 2, 3), help="Guard the selected script against a wrong bundle"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Isolated 4-update/2-save GPU proof; use a separate run name"
    )
    parser.add_argument(
        "--smoke-steps", type=int, choices=(4, 8), default=4,
        help="Use 8 with --smoke to exercise the low-LR stage as well as save/restore",
    )
    parser.add_argument("--execute", action="store_true", help="Without this flag, print the plan and write nothing")
    parser.add_argument(
        "--verify-restore",
        action="store_true",
        help="Reload a completed run's full state on the configured GPUs without additional optimizer updates",
    )
    args = parser.parse_args()
    command, identity, run_dir = plan_run(args)
    print(json.dumps({"run_dir": str(run_dir), **identity}, indent=2))
    if args.execute:
        execute_run(args, command, identity, run_dir)


if __name__ == "__main__":
    main()
