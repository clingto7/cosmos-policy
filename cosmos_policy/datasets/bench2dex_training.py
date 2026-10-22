# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Load an audited Bench2Dex task as a Cosmos training dataset."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np

from cosmos_policy.datasets.bench2dex_audit import TaskStatistics, load_profile
from cosmos_policy.datasets.bench2dex_dataset import CHUNK_SIZE, Bench2DexDataset


def load_bench2dex_training_dataset(
    manifest_path: str,
    robot_map_path: str,
    embedding_path: str,
    expected_active_joints: int = 26,
    chunk_size: int = CHUNK_SIZE,
) -> Bench2DexDataset:
    """Require the matched audit and the embedding of its full scene prompt."""
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"task03 action chunk must be {CHUNK_SIZE}")
    if not all((manifest_path, robot_map_path, embedding_path)):
        raise ValueError("BENCH2DEX_TASK03_MANIFEST, BENCH2DEX_ROBOT_MAP, and BENCH2DEX_TASK03_EMBEDDING are required")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("ready") or manifest.get("source_count") != 100 or manifest.get("matched_count") != 100:
        raise ValueError("training requires a ready audit of 100 matched demonstrations")
    profile = load_profile(robot_map_path, manifest["robot_key"])
    if len(profile.active_joint_names) != expected_active_joints:
        raise ValueError(f"expected {expected_active_joints} active joints for {profile.robot_key}")
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
