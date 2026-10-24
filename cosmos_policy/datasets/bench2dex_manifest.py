"""Build and dry-run the per-task Bench2Dex RGB training manifest.

The input is a YAML list with one record per benchmark task. This command only
audits data and validates the per-task recipe; it never starts training.

Each record needs task_id, task_name, robot_key, task_dir, scene, robot_map,
code_revision, base_weights, and separate statistics_root, checkpoint_root,
and log_root paths. Supply either yk_dir or yk_sha256 plus yk_location.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

TASK_COUNT = 26
EXPECTED_EPISODES = 100
CAMERA_COUNT = 4
CONTROL_HZ = 20
CAMERAS = ("cam_wrist_left", "cam_wrist_right", "cam_stereo_left", "cam_stereo_right")
ACTION_CHUNK = 16
PRIVATE_ROOT = "/inspire/hdd2/project/roboticsystem2/ky26063"


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task {record.get('task_id', '<unknown>')}: {key} must be a nonempty string")
    return value


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, list) or len(raw) != TASK_COUNT:
        raise ValueError(f"task catalog must contain exactly {TASK_COUNT} task records")
    tasks: list[dict[str, Any]] = []
    ids: set[str] = set()
    for record in raw:
        if not isinstance(record, dict):
            raise ValueError("each task catalog item must be a mapping")
        task_id = _required_text(record, "task_id")
        if task_id in ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        ids.add(task_id)
        for key in (
            "task_name", "robot_key", "task_dir", "scene", "robot_map", "code_revision",
            "data_revision", "scene_revision", "robot_map_revision", "base_weights",
        ):
            _required_text(record, key)
        if (record.get("yk_dir") is None) == (record.get("yk_sha256") is None):
            raise ValueError(f"task {task_id}: provide exactly one of yk_dir or yk_sha256")
        if record.get("yk_sha256") is not None:
            _required_text(record, "yk_location")
        for key in ("code_revision", "base_weights"):
            _required_text(record, key)
        for key in ("statistics_root", "checkpoint_root", "log_root"):
            _required_text(record, key)
        output_roots = [str(Path(record[key]).expanduser().resolve()) for key in ("statistics_root", "checkpoint_root", "log_root")]
        if len(set(output_roots)) != len(output_roots):
            raise ValueError(f"task {task_id}: statistics, checkpoint, and log roots must be separate")
        if any(path != PRIVATE_ROOT and not path.startswith(f"{PRIVATE_ROOT}/") for path in output_roots):
            raise ValueError(f"task {task_id}: output roots must resolve under {PRIVATE_ROOT}")
        tasks.append(record)
    return sorted(tasks, key=lambda item: item["task_id"])


def _task_output_name(task_id: str, robot_key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id):
        raise ValueError(f"unsafe task_id for output naming: {task_id!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", robot_key):
        raise ValueError(f"unsafe robot_key for output naming: {robot_key!r}")
    return f"task-{task_id}__{robot_key}"


def _audit_one(record: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    from cosmos_policy.datasets.bench2dex_audit import (
        audit_task,
        load_sha256sum,
        load_task_description,
        sha256_file,
    )
    from cosmos_policy.datasets.bench2dex_dataset import Bench2DexDataset, RobotProfile

    task_id = record["task_id"]
    robot_map = yaml.safe_load(Path(record["robot_map"]).read_text(encoding="utf-8"))
    robot_entry = robot_map["robots"][record["robot_key"]]
    if "active_joint_names" in robot_entry:
        active_joint_names = tuple(robot_entry["active_joint_names"])
    else:
        full_joint_names = tuple(robot_entry["full_joint_names"])
        active_indices = tuple(int(index) for index in robot_entry["active_indices"])
        if len(set(active_indices)) != len(active_indices) or any(index < 0 or index >= len(full_joint_names) for index in active_indices):
            raise ValueError(f"invalid active_indices for robot {record['robot_key']}")
        if "active_dof" in robot_entry and int(robot_entry["active_dof"]) != len(active_indices):
            raise ValueError(f"active_dof mismatch for robot {record['robot_key']}")
        active_joint_names = tuple(full_joint_names[index] for index in active_indices)
    profile = RobotProfile(record["robot_key"], tuple(robot_entry["full_joint_names"]), active_joint_names)
    description = load_task_description(record["scene"])
    reference_hashes = load_sha256sum(record["yk_sha256"]) if record.get("yk_sha256") else None
    manifest, statistics = audit_task(
        record["task_dir"],
        record.get("yk_dir"),
        profile,
        description,
        expected_count=expected_count,
        yk_hashes=reference_hashes,
        yk_location=record.get("yk_location"),
    )
    if record.get("yk_sha256"):
        checksum_path = Path(record["yk_sha256"])
        manifest["yk_checksum_file"] = str(checksum_path.resolve())
        manifest["yk_checksum_file_sha256"] = sha256_file(checksum_path)
    total_bytes = sum(Path(entry["source"]).stat().st_size for entry in manifest["entries"])
    output_name = _task_output_name(task_id, profile.robot_key)
    sample_check: dict[str, Any] = {"passed": False}
    if manifest["source_valid"]:
        dataset = Bench2DexDataset(
            [entry["source"] for entry in manifest["entries"]],
            profile,
            manifest["task_description"],
            statistics=statistics,
        )
        sample = dataset[0]
        if sample["command"] != manifest["task_description"]:
            raise ValueError("dataset sample does not use the full scene description")
        if tuple(sample["camera_order"]) != CAMERAS or sample["actions"].shape != (ACTION_CHUNK, len(profile.active_joint_names)):
            raise ValueError("dataset sample violates four-camera or active-joint action contract")
        sample_check = {
            "passed": True,
            "camera_order": list(sample["camera_order"]),
            "action_shape": list(sample["actions"].shape),
            "prompt_matches_scene": True,
        }
    roots = {
        key: str(Path(record[key]).expanduser().resolve())
        for key in ("statistics_root", "checkpoint_root", "log_root")
    }
    scoped_roots = {key: str(Path(value) / output_name) for key, value in roots.items()}
    dry_run = {
        "configuration_valid": True,
        "provenance_complete": bool(record.get("code_revision") and record.get("base_weights")),
        "output_name": output_name,
        "output_is_task_scoped": True,
        "sample_contract": sample_check,
        "resource_estimate": {
            "episode_count": manifest["source_count"],
            "input_bytes_to_scan": total_bytes,
            "active_joint_count": len(profile.active_joint_names),
            "camera_count": CAMERA_COUNT,
            "action_chunk": ACTION_CHUNK,
            "control_hz": CONTROL_HZ,
            "training_jobs_started": 0,
        },
        "ready": bool(manifest["ready"] and record.get("code_revision") and record.get("base_weights")),
        "reasons": [],
    }
    if not manifest["ready"]:
        dry_run["reasons"].append("dataset audit is not ready; see data_audit")
    if not record.get("code_revision") or not record.get("base_weights"):
        dry_run["reasons"].append("code_revision and base_weights are required")
    manifest.update(
        task_id=task_id,
        task_name=record["task_name"],
        source_revision=record["code_revision"],
        data_revision=record["data_revision"],
        scene_revision=record["scene_revision"],
        scene_sha256=hashlib.sha256(Path(record["scene"]).read_bytes()).hexdigest(),
        robot_map_revision=record["robot_map_revision"],
        robot_map_sha256=hashlib.sha256(Path(record["robot_map"]).read_bytes()).hexdigest(),
        base_weights=record["base_weights"],
        active_joint_count=len(profile.active_joint_names),
        action_contract={
            "target": "absolute_active_joint_positions",
            "chunk_size": ACTION_CHUNK,
            "execute_entire_chunk": True,
            "control_hz": CONTROL_HZ,
        },
        input_contract={
            "cameras": ["cam_stereo_left", "cam_stereo_right", "cam_wrist_left", "cam_wrist_right"],
            "state": "current_active_joint_positions",
            "task_description": "full scene.description",
            "excluded": ["depth", "tactile", "object_state", "box", "occupancy"],
        },
        output_name=output_name,
        output_roots=scoped_roots,
        dry_run=dry_run,
    )
    return manifest


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_manifest(
    tasks: list[dict[str, Any]],
    *,
    selected_task_id: str | None,
    expected_count: int = EXPECTED_EPISODES,
    output_path: Path | None = None,
) -> dict[str, Any]:
    selected = {task["task_id"] for task in tasks} if selected_task_id is None else {selected_task_id}
    if selected_task_id is not None and selected_task_id not in {task["task_id"] for task in tasks}:
        raise ValueError(f"task_id not in catalog: {selected_task_id}")
    results: list[dict[str, Any]] = [
        {
            "task_id": record["task_id"],
            "task_name": record["task_name"],
            "status": "pending" if record["task_id"] in selected else "not_run",
        }
        for record in tasks
    ]

    def persist_progress() -> dict[str, Any]:
        document = {
            "schema_version": 1,
            "protocol": "bench2dex-26task-rgb-baseline",
            "expected_task_count": TASK_COUNT,
            "expected_episodes_per_task": expected_count,
            "selected_task_id": selected_task_id,
            "training_jobs_started": 0,
            "tasks": results.copy(),
            "summary": {
                "ready": sum(task["status"] == "ready" for task in results),
                "failed": sum(task["status"] == "failed" for task in results),
                "not_run": sum(task["status"] == "not_run" for task in results),
                "pending": sum(task["status"] == "pending" for task in results),
            },
        }
        if output_path is not None:
            _write_json_atomic(output_path, document)
        return document

    persist_progress()
    for index, record in enumerate(tasks):
        if record["task_id"] not in selected:
            continue
        try:
            result = _audit_one(record, expected_count=expected_count)
            result["status"] = "ready" if result["dry_run"]["ready"] else "failed"
        except Exception as error:  # Preserve a machine-readable failure for each task.
            result = {
                "task_id": record["task_id"],
                "task_name": record["task_name"],
                "robot_key": record["robot_key"],
                "status": "failed",
                "failure_reason": f"{type(error).__name__}: {error}",
                "dry_run": {"ready": False, "training_jobs_started": 0},
            }
        results[index] = result
        print(json.dumps({"task_id": record["task_id"], "status": result["status"]}), flush=True)
        document = persist_progress()
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path, help="YAML catalog containing exactly 26 task records")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task-id", help="audit one task only; the other catalog entries are marked not_run")
    parser.add_argument("--expected-episodes", type=int, default=EXPECTED_EPISODES)
    parser.add_argument("--allow-incomplete", action="store_true", help="return success even when tasks fail or are not run")
    args = parser.parse_args()
    if args.expected_episodes <= 0:
        parser.error("--expected-episodes must be positive")
    try:
        tasks = load_tasks(args.tasks)
        output_path = args.output.expanduser().resolve()
        if output_path != Path(PRIVATE_ROOT) and not str(output_path).startswith(f"{PRIVATE_ROOT}/"):
            parser.error(f"--output must resolve under {PRIVATE_ROOT}")
        result = build_manifest(
            tasks,
            selected_task_id=args.task_id,
            expected_count=args.expected_episodes,
            output_path=output_path,
        )
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps(result["summary"], sort_keys=True))
    complete = result["summary"]["not_run"] == 0 or args.task_id is not None
    return 0 if args.allow_incomplete or result["summary"]["failed"] == 0 and complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
