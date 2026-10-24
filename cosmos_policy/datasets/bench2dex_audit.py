# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Audit a task's matched Bench2Dex demonstrations and training statistics.

Example::

    python -m cosmos_policy.datasets.bench2dex_audit \
      --task-dir /data/03_wine_glass_plate_balance/replay-generalization \
      --yk-dir /yk/03_wine_glass_plate_balance/replay-generalization \
      --scene /Bench2Dex/scenes/03_wine_glass_plate_balance.yaml \
      --robot-map /Bench2Dex/robots/active_dof_maps.yml \
      --manifest-output /work/task03-manifest.json

Use ``--yk-sha256`` and ``--yk-location`` instead of ``--yk-dir`` when the
reference directory is on a separate host. Capture hashes there with
``sha256sum episode_*.hdf5`` and keep the unmodified output as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import yaml

from cosmos_policy.datasets.bench2dex_dataset import (
    CAMERAS,
    Bench2DexDataset,
    EpisodeInfo,
    RobotProfile,
    inspect_episode,
)


def load_profile(robot_map: str | Path, robot_key: str) -> RobotProfile:
    with Path(robot_map).open(encoding="utf-8") as stream:
        robots = yaml.safe_load(stream)["robots"]
    if robot_key not in robots:
        raise ValueError(f"robot_key={robot_key!r} not found in {robot_map}")
    entry = robots[robot_key]
    return RobotProfile(robot_key, tuple(entry["full_joint_names"]), tuple(entry["active_joint_names"]))


def load_task_description(scene: str | Path) -> str:
    with Path(scene).open(encoding="utf-8") as stream:
        scene_data = yaml.safe_load(stream)
    description = scene_data.get("description") if isinstance(scene_data, dict) else None
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"scene has no full description: {scene}")
    return description


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskStatistics:
    robot_key: str
    active_joint_names: tuple[str, ...]
    proprio_min: np.ndarray
    proprio_max: np.ndarray
    actions_min: np.ndarray
    actions_max: np.ndarray

    def _transform(self, values: np.ndarray, minimum: np.ndarray, maximum: np.ndarray, inverse: bool) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1] != len(self.active_joint_names):
            raise ValueError("statistics dimension does not match active joints")
        center = (minimum + maximum) / 2
        half_range = (maximum - minimum) / 2
        safe_range = np.where(half_range > 0, half_range, 1)
        if inverse:
            return (center + values * half_range).astype(np.float32)
        return ((values - center) / safe_range).astype(np.float32)

    def normalize_actions(self, values: np.ndarray) -> np.ndarray:
        return self._transform(values, self.actions_min, self.actions_max, False)

    def denormalize_actions(self, values: np.ndarray) -> np.ndarray:
        return self._transform(values, self.actions_min, self.actions_max, True)

    def normalize_proprio(self, values: np.ndarray) -> np.ndarray:
        return self._transform(values, self.proprio_min, self.proprio_max, False)

    def denormalize_proprio(self, values: np.ndarray) -> np.ndarray:
        return self._transform(values, self.proprio_min, self.proprio_max, True)

    def to_dict(self) -> dict:
        return {
            "robot_key": self.robot_key,
            "active_joint_names": list(self.active_joint_names),
            "proprio_min": self.proprio_min.tolist(),
            "proprio_max": self.proprio_max.tolist(),
            "actions_min": self.actions_min.tolist(),
            "actions_max": self.actions_max.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskStatistics:
        keys = ("proprio_min", "proprio_max", "actions_min", "actions_max")
        arrays = (np.asarray(data[key], dtype=np.float32) for key in keys)
        return cls(data["robot_key"], tuple(data["active_joint_names"]), *arrays)


def compute_statistics(episodes: Sequence[EpisodeInfo], profile: RobotProfile) -> TaskStatistics:
    """Use only selected task demonstration frames; no final/pilot episodes."""
    dim = len(profile.active_joint_names)
    proprio_min = np.full(dim, np.inf, dtype=np.float32)
    proprio_max = np.full(dim, -np.inf, dtype=np.float32)
    actions_min = np.full(dim, np.inf, dtype=np.float32)
    actions_max = np.full(dim, -np.inf, dtype=np.float32)
    for info in episodes:
        with h5py.File(info.path, "r") as file:
            for start, end in info.action_runs:
                proprio = np.asarray(file["robot/qpos"][start:end], dtype=np.float32)[:, info.joint_indices]
                actions = np.asarray(file["action/commanded"][start:end], dtype=np.float32)[:, info.action_indices]
                proprio_min = np.minimum(proprio_min, proprio.min(axis=0))
                proprio_max = np.maximum(proprio_max, proprio.max(axis=0))
                actions_min = np.minimum(actions_min, actions.min(axis=0))
                actions_max = np.maximum(actions_max, actions.max(axis=0))
    if not np.isfinite(actions_min).all() or not np.isfinite(proprio_min).all():
        raise ValueError("cannot compute statistics from empty or nonfinite demonstrations")
    return TaskStatistics(
        profile.robot_key, profile.active_joint_names, proprio_min, proprio_max, actions_min, actions_max
    )


def _episodes(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"episode directory not found: {directory}")
    return sorted(directory.glob("episode_*.hdf5"))


def load_sha256sum(path: str | Path) -> dict[str, str]:
    """Read a GNU sha256sum list captured from the yk training directory."""
    hashes: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (episode_[^/]+\.hdf5)", line)
        if match is None or match[2] in hashes:
            raise ValueError(f"invalid or duplicate sha256sum entry at line {number}")
        hashes[match[2]] = match[1]
    if not hashes:
        raise ValueError("empty yk sha256sum file")
    return hashes


def audit_task(
    task_dir: str | Path,
    yk_dir: str | Path | None,
    profile: RobotProfile,
    task_description: str,
    *,
    expected_count: int = 100,
    yk_hashes: dict[str, str] | None = None,
    yk_location: str | None = None,
    task_id: str | None = None,
) -> tuple[dict, TaskStatistics | None]:
    """Audit data against yk files or a checksum list captured on its host."""
    if yk_dir is not None and yk_hashes is not None:
        raise ValueError("provide either yk_dir or yk_hashes")
    source_paths = _episodes(Path(task_dir))
    inferred_task = re.match(r"^(\d{2})_", Path(task_dir).parent.name)
    resolved_task_id = task_id or (inferred_task.group(1) if inferred_task else None)
    if not resolved_task_id or not re.fullmatch(r"\d{2}", resolved_task_id):
        raise ValueError("task_id must be supplied or inferable from <NN>_<task>/replay-generalization")
    reference_paths = _episodes(Path(yk_dir)) if yk_dir is not None else []
    digest_cache: dict[Path, str] = {}

    def digest(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = sha256_file(path)
        return digest_cache[resolved]

    reference_by_hash: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in reference_paths:
        reference_by_hash[digest(path)].append((path.name, str(path.resolve())))
    if yk_hashes is not None:
        for name, file_hash in yk_hashes.items():
            if not re.fullmatch(r"episode_[^/]+\.hdf5", name) or not re.fullmatch(r"[0-9a-f]{64}", file_hash):
                raise ValueError(f"invalid yk checksum record: {name}")
            location = f"{yk_location.rstrip('/')}/{name}" if yk_location else name
            reference_by_hash[file_hash].append((name, location))
    has_reference = yk_dir is not None or yk_hashes is not None
    reference_count = len(reference_paths) if yk_dir is not None else len(yk_hashes or {})

    entries: list[dict] = []
    valid_infos: list[EpisodeInfo] = []
    for path in source_paths:
        file_hash = digest(path)
        references = reference_by_hash[file_hash]
        exact = next((index for index, (name, _) in enumerate(references) if name == path.name), None)
        reference = references.pop(exact if exact is not None else 0) if references else None
        if reference:
            reference_name, reference_source = reference
            if str(path.resolve()) == reference_source:
                identity = "same_path"
            else:
                identity = "same_sha256" if path.name == reference_name else "renamed_sha256"
        else:
            identity = "unmatched" if has_reference else "not_checked"
        entry = {
            "source": str(path.resolve()),
            "sha256": file_hash,
            "yk_source": reference[1] if reference else None,
            "identity": identity,
        }
        try:
            with h5py.File(path, "r") as file:
                key = file.get("meta/robot_key")
                if key is not None:
                    value = key[()]
                    entry["robot_key"] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                entry["camera_complete"] = all(f"cameras/{camera}/rgb" in file for camera in CAMERAS)
        except OSError:
            entry["camera_complete"] = False
        entry.setdefault("robot_key", None)
        try:
            info = inspect_episode(path, profile)
            entry.update(
                robot_key=info.robot_key,
                camera_complete=True,
                usable_frames=info.usable_frames,
                excluded_homing_frames=info.excluded_frames,
                unusable_task_frames=info.unusable_task_frames,
            )
            with h5py.File(path, "r") as file:
                instruction = file.get("meta/instruction")
                if instruction is not None:
                    value = instruction[()]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    # yk's train.sh uses scene.description as the prompt.
                    # Keep historical HDF5 text for provenance, not filtering.
                    entry["embedded_instruction"] = str(value)
                    entry["prompt_matches_embedded"] = str(value) == task_description
                else:
                    entry["embedded_instruction"] = None
                    entry["prompt_matches_embedded"] = None
            entry["exclusion_reason"] = None
            valid_infos.append(info)
        except (OSError, KeyError, TypeError, ValueError) as error:
            entry.setdefault("usable_frames", 0)
            entry.setdefault("excluded_homing_frames", 0)
            entry.setdefault("unusable_task_frames", 0)
            entry["exclusion_reason"] = str(error)
        entry.setdefault("embedded_instruction", None)
        entry.setdefault("prompt_matches_embedded", None)
        entries.append(entry)

    unmatched_yk = [source for records in reference_by_hash.values() for _, source in records]
    source_valid = len(source_paths) == expected_count and all(entry["exclusion_reason"] is None for entry in entries)
    ready = (
        has_reference
        and source_valid
        and reference_count == expected_count
        and not unmatched_yk
        and all(entry["identity"] in ("same_path", "same_sha256", "renamed_sha256") for entry in entries)
    )
    statistics = compute_statistics(valid_infos, profile) if source_valid else None
    reference_kind = "local_directory" if yk_dir is not None else "remote_sha256sum" if yk_hashes is not None else None
    reference_location = (
        str(Path(yk_dir).resolve()) if yk_dir is not None else yk_location if yk_hashes is not None else None
    )
    manifest = {
        "task_id": resolved_task_id,
        "robot_key": profile.robot_key,
        "active_joint_names": list(profile.active_joint_names),
        "active_joint_count": len(profile.active_joint_names),
        "camera_names": list(CAMERAS),
        "action_horizon": 16,
        "control_hz": 20,
        "task_description": task_description,
        "prompt_source": "scene.description",
        "expected_count": expected_count,
        "source_count": len(source_paths),
        "yk_count": reference_count,
        "yk_reference_kind": reference_kind,
        "yk_location": reference_location,
        "matched_count": sum(entry["identity"] in ("same_path", "same_sha256", "renamed_sha256") for entry in entries),
        "source_valid": source_valid,
        "ready": ready,
        "entries": entries,
        "unmatched_yk": unmatched_yk,
        "statistics": statistics.to_dict() if statistics else None,
    }
    return manifest, statistics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, type=Path)
    yk = parser.add_mutually_exclusive_group()
    yk.add_argument("--yk-dir", type=Path)
    yk.add_argument("--yk-sha256", type=Path)
    parser.add_argument("--yk-location")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--robot-map", required=True, type=Path)
    parser.add_argument("--robot-key", default="multi_xarm7_with_ability")
    parser.add_argument("--task-id", help="Two-digit task ID; otherwise infer it from the task directory")
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    profile = load_profile(args.robot_map, args.robot_key)
    description = load_task_description(args.scene)
    if args.yk_sha256 and not args.yk_location:
        parser.error("--yk-location is required with --yk-sha256")
    yk_hashes = load_sha256sum(args.yk_sha256) if args.yk_sha256 else None
    manifest, statistics = audit_task(
        args.task_dir,
        args.yk_dir,
        profile,
        description,
        yk_hashes=yk_hashes,
        yk_location=args.yk_location,
        task_id=args.task_id,
    )
    if args.yk_sha256:
        manifest["yk_checksum_file"] = str(args.yk_sha256.resolve())
        manifest["yk_checksum_file_sha256"] = sha256_file(args.yk_sha256)
    if statistics is not None:
        # Confirm that the same full prompt and four-camera contract reach the
        # training sample, with no extra demonstration holdout.
        paths = [entry["source"] for entry in manifest["entries"]]
        dataset = Bench2DexDataset(paths, profile, description, statistics=statistics)
        sample = dataset[0]
        if sample["command"] != description or sample["video"].shape[1] != 49:
            raise AssertionError("dataset sample differs from audited task contract")
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_keys = ("source_count", "yk_count", "matched_count", "source_valid", "ready")
    print(json.dumps({key: manifest[key] for key in summary_keys}))
    return 0 if manifest["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
