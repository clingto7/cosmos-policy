"""Audited multi-task bundles and reproducible task-uniform sampling.

No CUDA/model imports: bundle preparation and the data contract run on CPU.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from cosmos_policy.datasets.bench2dex_audit import TaskStatistics, compute_statistics, load_profile, sha256_file
from cosmos_policy.datasets.bench2dex_dataset import CAMERAS, CHUNK_SIZE, Bench2DexDataset
from cosmos_policy.datasets.bench2dex_training import resolve_bench2dex_audit_config

SCHEMA = "bench2dex-embodiment-v1"
UPDATES = 50_000
GLOBAL_BATCH = 64


def checked_statistics(data: dict, profile) -> TaskStatistics:
    stats = TaskStatistics.from_dict(data)
    if stats.robot_key != profile.robot_key or stats.active_joint_names != profile.active_joint_names:
        raise ValueError("statistics do not match embodiment joint order")
    for name in ("proprio", "actions"):
        low, high = getattr(stats, name + "_min"), getattr(stats, name + "_max")
        if (
            low.shape != (len(profile.active_joint_names),)
            or high.shape != low.shape
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or np.any(low > high)
        ):
            raise ValueError("invalid statistics bounds")
    return stats


def merge_statistics(statistics: list[TaskStatistics]) -> TaskStatistics:
    """Min/max over a union equals the extrema of each audited task's extrema."""
    first = statistics[0]
    if any(s.robot_key != first.robot_key or s.active_joint_names != first.active_joint_names for s in statistics):
        raise ValueError("cannot merge different embodiments or joint orders")
    return TaskStatistics(
        first.robot_key,
        first.active_joint_names,
        *(
            operation(np.stack([getattr(s, key) for s in statistics]), axis=0)
            for key, operation in (
                ("proprio_min", np.min),
                ("proprio_max", np.max),
                ("actions_min", np.min),
                ("actions_max", np.max),
            )
        ),
    )


def load_embedding(task: dict, root: Path) -> np.ndarray:
    path = root / task["embedding_path"]
    if sha256_file(path) != task["embedding_sha256"]:
        raise ValueError(f"embedding changed for task {task['task_id']}")
    with np.load(path, allow_pickle=False) as saved:
        embedding = np.asarray(saved["embedding"], dtype=np.float32)
        if (
            str(saved["prompt"]) != task["audit"]["task_description"]
            or embedding.ndim != 2
            or min(embedding.shape) == 0
            or not np.isfinite(embedding).all()
        ):
            raise ValueError("embedding must match the audited full task description")
    return embedding


def validate_bundle(bundle: dict, robot_map_path: str) -> tuple[object, TaskStatistics]:
    if bundle.get("schema") != SCHEMA or not bundle.get("tasks"):
        raise ValueError("expected a nonempty Bench2Dex embodiment bundle")
    if bundle.get("robot_map_sha256") != sha256_file(robot_map_path):
        raise ValueError("robot map changed since bundle preparation")
    profile = load_profile(robot_map_path, bundle["robot_key"])
    if tuple(bundle["active_joint_names"]) != profile.active_joint_names:
        raise ValueError("bundle active-joint order differs from robot map")
    ids, prompts, sources, statistics = set(), set(), set(), []
    for task in bundle["tasks"]:
        audit = task["audit"]
        config = resolve_bench2dex_audit_config(audit, robot_map_path, expected_task_id=task["task_id"])
        if config["robot_key"] != profile.robot_key:
            raise ValueError("bundle mixes embodiments")
        prompt = audit["task_description"]
        if task["task_id"] in ids or not prompt.strip() or prompt in prompts:
            raise ValueError("task IDs and standard prompts must be nonempty and unique")
        ids.add(task["task_id"])
        prompts.add(prompt)
        entries = audit["entries"]
        if len(entries) != 100:
            raise ValueError("each task requires 100 audited episodes")
        for entry in entries:
            path = str(Path(entry["source"]).resolve())
            if (
                path in sources
                or entry.get("exclusion_reason") is not None
                or entry.get("identity") not in ("same_path", "same_sha256", "renamed_sha256")
                or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))
            ):
                raise ValueError("duplicate, invalid, or unmatched audited episode")
            sources.add(path)
        statistics.append(checked_statistics(audit["statistics"], profile))
    shared = checked_statistics(bundle["statistics"], profile)
    if shared.to_dict() != merge_statistics(statistics).to_dict():
        raise ValueError("shared statistics differ from the union of task statistics")
    return profile, shared


def read_bundle(path: str | Path, robot_map_path: str) -> tuple[dict, object, TaskStatistics]:
    bundle = json.loads(Path(path).read_text())
    profile, statistics = validate_bundle(bundle, robot_map_path)
    return bundle, profile, statistics


def resolve_embodiment_config(path: str, robot_map_path: str) -> dict:
    bundle, profile, _ = read_bundle(path, robot_map_path)
    return {
        "robot_key": profile.robot_key,
        "task_id": bundle["tasks"][0]["task_id"],
        "active_joint_names": profile.active_joint_names,
        "action_dim": len(profile.active_joint_names),
        "camera_names": CAMERAS,
        "action_horizon": CHUNK_SIZE,
        "control_hz": 20,
    }


def select_task(bundle: dict, task_id: str) -> dict:
    for task in bundle["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"task {task_id!r} does not belong to embodiment {bundle['robot_key']}")


class EmbodimentDataset:
    """A virtual stream: each draw independently chooses a task, then a window.

    Draw identity is independent of rank, worker count, prefetch and resume.
    Task probabilities are equal in expectation, not forced within each batch.
    """

    def __init__(self, datasets, task_ids, *, seed=0, total_samples=UPDATES * GLOBAL_BATCH):
        if not datasets or any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("all tasks need valid training windows")
        if len(datasets) != len(task_ids) or len(set(task_ids)) != len(task_ids):
            raise ValueError("datasets require distinct task IDs")
        self.datasets, self.task_ids = list(datasets), list(task_ids)
        self.seed, self.total_samples = int(seed), int(total_samples)
        if self.seed < 0 or self.total_samples <= 0:
            raise ValueError("invalid sampling seed or budget")

    def __len__(self):
        return self.total_samples

    def locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, int(index)])))
        task = int(rng.integers(len(self.datasets)))
        return task, int(rng.integers(len(self.datasets[task])))

    def __getitem__(self, index):
        task, window = self.locate(index)
        sample = dict(self.datasets[task][window])
        sample["task_id"] = self.task_ids[task]
        sample["__key__"] = f"task{self.task_ids[task]}/{sample['__key__']}"
        return sample


class EmbodimentSampler:
    """Shard draw IDs across ranks; resume at completed_updates * global_batch."""

    def __init__(self, dataset, num_replicas, rank, global_batch=GLOBAL_BATCH):
        if num_replicas != 2 or not 0 <= rank < num_replicas:
            raise ValueError("per-embodiment training requires two data-parallel ranks")
        if global_batch != GLOBAL_BATCH or len(dataset) % global_batch:
            raise ValueError("sampler requires complete global batches of 64")
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.global_batch, self.start = global_batch, 0

    def set_start_iteration(self, iteration):
        self.start = int(iteration) * self.global_batch
        if not 0 <= self.start <= len(self.dataset):
            raise ValueError("resume iteration exceeds the sampling budget")

    def set_epoch(self, epoch):
        if epoch != 0:
            raise ValueError("embodiment draw stream exhausted before training budget")

    def __iter__(self):
        return iter(range(self.start + self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return (len(self.dataset) - self.start) // self.num_replicas


def load_embodiment_dataset(
    manifest_path: str, robot_map_path: str, chunk_size=CHUNK_SIZE, seed=0, total_samples=UPDATES * GLOBAL_BATCH
):
    if chunk_size != CHUNK_SIZE:
        raise ValueError("Bench2Dex requires 16-step actions")
    bundle, profile, stats = read_bundle(manifest_path, robot_map_path)
    datasets, shapes = [], set()
    for task in bundle["tasks"]:
        embedding = load_embedding(task, Path(manifest_path).parent)
        shapes.add(embedding.shape)
        if len(shapes) > 1:
            raise ValueError("task embeddings must have the same padded shape for batching")
        datasets.append(
            Bench2DexDataset(
                [entry["source"] for entry in task["audit"]["entries"]],
                profile,
                task["audit"]["task_description"],
                text_embedding=embedding,
                statistics=stats,
            )
        )
    return EmbodimentDataset(datasets, [t["task_id"] for t in bundle["tasks"]], seed=seed, total_samples=total_samples)


def prepare_bundles(
    catalog_path: str | Path, robot_map_path: str, output: str | Path, grouping_path: str | Path
) -> list[Path]:
    """Build one bundle per robot from a JSON list of audit/embedding paths.

    Recheck source hashes and statistics before admitting expensive training.
    Input catalog paths are relative to the catalog, never the process cwd.
    """
    catalog_path, output = Path(catalog_path).resolve(), Path(output).resolve()
    catalog = json.loads(catalog_path.read_text())
    if not isinstance(catalog, list) or not catalog:
        raise ValueError("catalog must be a nonempty JSON list")
    grouping_path = Path(grouping_path).resolve()
    expected = defaultdict(set)
    for record in json.loads(grouping_path.read_text())["per_task"]:
        expected[record["robot_key"]].add(record["task"].split("_", 1)[0])
    groups, ids, shapes = defaultdict(list), set(), {}
    for record in catalog:
        manifest_path = (catalog_path.parent / record["manifest"]).resolve()
        embedding_path = (catalog_path.parent / record["embedding"]).resolve()
        audit = json.loads(manifest_path.read_text())
        config = resolve_bench2dex_audit_config(audit, robot_map_path)
        task_id, robot_key = str(config["task_id"]), str(config["robot_key"])
        if task_id in ids or not re.fullmatch(r"[A-Za-z0-9_]+", robot_key):
            raise ValueError("duplicate task ID or unsafe robot key")
        ids.add(task_id)
        audit["task_id"] = task_id
        profile = load_profile(robot_map_path, robot_key)
        task = {
            "task_id": task_id,
            "audit": audit,
            "audit_path": str(manifest_path),
            "audit_sha256": sha256_file(manifest_path),
            "embedding_path": str(embedding_path),
            "embedding_sha256": sha256_file(embedding_path),
        }
        embedding = load_embedding(task, catalog_path.parent)
        if robot_key in shapes and shapes[robot_key] != embedding.shape:
            raise ValueError("task embeddings must have the same padded shape for batching")
        shapes[robot_key] = embedding.shape
        for entry in audit["entries"]:
            if not Path(entry["source"]).is_absolute() or sha256_file(entry["source"]) != entry["sha256"]:
                raise ValueError(f"demonstration changed since audit: {entry['source']}")
        dataset = Bench2DexDataset([e["source"] for e in audit["entries"]], profile, audit["task_description"])
        actual = compute_statistics(dataset.episodes, profile)
        if actual.to_dict() != checked_statistics(audit["statistics"], profile).to_dict():
            raise ValueError("training data statistics differ from audit")
        groups[robot_key].append(task)
    bundles = []
    for robot_key, tasks in sorted(groups.items()):
        tasks.sort(key=lambda task: task["task_id"])
        if {task["task_id"] for task in tasks} != expected.get(robot_key):
            raise ValueError(f"catalog must include the existing complete task group for {robot_key}")
        profile = load_profile(robot_map_path, robot_key)
        bundle = {
            "schema": SCHEMA,
            "robot_map_sha256": sha256_file(robot_map_path),
            "robot_key": robot_key,
            "grouping_path": str(grouping_path),
            "grouping_sha256": sha256_file(grouping_path),
            "active_joint_names": list(profile.active_joint_names),
            "tasks": tasks,
            "statistics": merge_statistics(
                [checked_statistics(t["audit"]["statistics"], profile) for t in tasks]
            ).to_dict(),
        }
        validate_bundle(bundle, robot_map_path)
        bundles.append((output / f"{robot_key}.json", bundle))
    # Validate the entire catalog before creating any output.
    if any(path.exists() for path, _ in bundles):
        raise FileExistsError("bundle output already exists; use a new versioned directory")
    output.mkdir(parents=True, exist_ok=True)
    for path, bundle in bundles:
        with path.open("x") as stream:
            stream.write(json.dumps(bundle, indent=2) + "\n")
    return [path for path, _ in bundles]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="JSON list of {manifest, embedding} audit paths")
    parser.add_argument("--robot-map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grouping", required=True, help="Bench2Dex policy/GR00T_XE/xe_norm_stats.taskstats.json")
    args = parser.parse_args()
    for path in prepare_bundles(args.catalog, args.robot_map, args.output, args.grouping):
        print(path)


if __name__ == "__main__":
    main()
