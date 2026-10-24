# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Load an audited Bench2Dex task as a Cosmos training dataset."""

from __future__ import annotations

import json
import argparse
import re
from pathlib import Path

import numpy as np

from cosmos_policy.datasets.bench2dex_audit import TaskStatistics, load_profile
from cosmos_policy.datasets.bench2dex_dataset import CAMERAS, CHUNK_SIZE, Bench2DexDataset


def resolve_bench2dex_task_config(
    manifest_path: str,
    robot_map_path: str,
    *,
    expected_task_id: str | None = None,
    expected_active_joints: int | None = None,
) -> dict[str, object]:
    """Resolve model dimensions from the selected audit manifest and robot map."""
    if not manifest_path:
        if expected_task_id not in (None, "03") or expected_active_joints not in (None, 26):
            raise ValueError("non-Task03 configs require a task audit manifest")
        return {
            "task_id": "03",
            "robot_key": "multi_xarm7_with_ability",
            "active_joint_names": (),
            "action_dim": 26,
            "camera_names": CAMERAS,
            "action_horizon": CHUNK_SIZE,
            "control_hz": 20,
        }
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("ready") or manifest.get("source_count") != 100 or manifest.get("matched_count") != 100:
        raise ValueError("Bench2Dex model config requires an audit of 100 matched demonstrations")
    task_id = manifest.get("task_id")
    if task_id is None and manifest.get("entries"):
        task_dir = Path(manifest["entries"][0]["source"]).parent.parent.name
        task_match = re.match(r"^(\d{2})_", task_dir)
        task_id = task_match.group(1) if task_match else None
    if not isinstance(task_id, str) or not re.fullmatch(r"\d{2}", task_id):
        raise ValueError("audit manifest must identify a two-digit task_id")
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError(f"audit task_id={task_id!r} does not match configured task_id={expected_task_id!r}")
    if manifest.get("robot_key") is None:
        raise ValueError("audit manifest is missing robot_key")
    profile = load_profile(robot_map_path, manifest["robot_key"])
    active_names = tuple(manifest.get("active_joint_names", profile.active_joint_names))
    if active_names != profile.active_joint_names:
        raise ValueError("audit active-joint order differs from the robot profile")
    action_dim = len(active_names)
    if not action_dim or manifest.get("active_joint_count", action_dim) != action_dim:
        raise ValueError("audit active-joint count is inconsistent")
    if expected_active_joints is not None and expected_active_joints != action_dim:
        raise ValueError("configured action dimension differs from the audited active-joint count")
    cameras = tuple(manifest.get("camera_names", CAMERAS))
    if cameras != CAMERAS:
        raise ValueError(f"audit camera order must match the RGB baseline: {CAMERAS}")
    horizon = int(manifest.get("action_horizon", CHUNK_SIZE))
    control_hz = int(manifest.get("control_hz", 20))
    if horizon != CHUNK_SIZE or control_hz != 20:
        raise ValueError("audit action horizon and control rate must be 16 targets at 20 Hz")
    return {
        "task_id": task_id,
        "robot_key": profile.robot_key,
        "active_joint_names": profile.active_joint_names,
        "action_dim": action_dim,
        "camera_names": cameras,
        "action_horizon": horizon,
        "control_hz": control_hz,
    }


def load_bench2dex_training_dataset(
    manifest_path: str,
    robot_map_path: str,
    embedding_path: str,
    expected_active_joints: int | None = None,
    expected_task_id: str | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> Bench2DexDataset:
    """Require the matched audit and the embedding of its full scene prompt."""
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"Bench2Dex action chunk must be {CHUNK_SIZE}")
    if not all((manifest_path, robot_map_path, embedding_path)):
        raise ValueError("Bench2Dex manifest, robot map, and task embedding paths are required")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("ready") or manifest.get("source_count") != 100 or manifest.get("matched_count") != 100:
        raise ValueError("training requires a ready audit of 100 matched demonstrations")
    profile = load_profile(robot_map_path, manifest["robot_key"])
    if expected_active_joints is not None and len(profile.active_joint_names) != expected_active_joints:
        raise ValueError(f"expected {expected_active_joints} active joints for {profile.robot_key}")
    task_id = manifest.get("task_id")
    if task_id is None and manifest.get("entries"):
        task_dir_name = Path(manifest["entries"][0]["source"]).parent.parent.name
        task_match = re.match(r"^(\d{2})_", task_dir_name)
        task_id = task_match.group(1) if task_match else None
    if expected_task_id is not None and task_id != expected_task_id:
        raise ValueError(f"audit task_id={task_id!r} does not match configured task_id={expected_task_id!r}")
    resolve_bench2dex_task_config(
        manifest_path,
        robot_map_path,
        expected_task_id=expected_task_id,
        expected_active_joints=expected_active_joints,
    )
    statistics = TaskStatistics.from_dict(manifest["statistics"])
    if statistics.robot_key != profile.robot_key or statistics.active_joint_names != profile.active_joint_names:
        raise ValueError("audit statistics do not match robot profile")
    with np.load(embedding_path, allow_pickle=False) as data:
        prompt = str(data["prompt"])
        embedding = data["embedding"]
    if prompt != manifest["task_description"] or embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("T5 embedding must match the audited full task description")
    paths = [entry["source"] for entry in manifest["entries"]]
    if len(paths) != 100 or len(set(paths)) != 100:
        raise ValueError("audit must contain 100 distinct episodes")
    return Bench2DexDataset(paths, profile, prompt, text_embedding=embedding, statistics=statistics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode the audited full task prompt for Cosmos training")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-model", default="google-t5/t5-11b")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not manifest.get("ready"):
        raise ValueError("prompt embedding requires a ready audit")
    from cosmos_policy._src.predict2.inference.get_t5_emb import CosmosT5TextEncoder, get_text_embedding

    prompt = manifest["task_description"]
    encoder = CosmosT5TextEncoder(model_name=args.text_model, device=args.device)
    embedding = get_text_embedding(prompt, encoder=encoder, device=args.device).squeeze(0).float().cpu().numpy()
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("text encoder returned an invalid embedding")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, prompt=prompt, embedding=embedding)


if __name__ == "__main__":
    main()
