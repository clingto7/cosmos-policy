#!/usr/bin/env python3
"""Offline generation on audited expert observations; never advances a simulator."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def full_windows(runs, count=6):
    starts = [i for start, end in runs for i in range(start, end - 15)]
    if not starts:
        return []
    return sorted({starts[int(i)] for i in np.linspace(0, len(starts) - 1, count)})


def melody_progress(qpos, names, valid):
    result = {}
    for side, melody in [("left", (18, 18, 23, 23, 24, 24, 23)),
                         ("right", (33, 33, 38, 38, 39, 39, 38))]:
        index, armed, events = 0, True, []
        for t in np.flatnonzero(valid):
            if index == len(melody):
                break
            value = qpos[t, names.index(f"joint_{melody[index]}_to_1")]
            if not armed and value < 0.002:
                armed = True
            if armed and value > 0.004:
                index += 1
                armed = False
                events.append(int(t))
        result[side] = {"notes": index, "event_frames": events}
    return result


def choose_sources(records, count):
    # Balance metadata strata while selecting each commanded trajectory only once.
    buckets = {}
    for matching in (False, True):
        candidates = [r for r in records if r["prompt_matches"] == matching]
        candidates.sort(key=lambda r: r["command_sha256"])
        buckets[matching] = candidates
    selected, seen = [], set()
    while len(selected) < count:
        changed = False
        for bucket in buckets.values():
            while bucket and bucket[0]["command_sha256"] in seen:
                bucket.pop(0)
            if bucket and len(selected) < count:
                record = bucket.pop(0)
                selected.append(record)
                seen.add(record["command_sha256"])
                changed = True
        if not changed:
            break
    return selected


def raw_image(file, camera, frame):
    import cv2
    ds = file[f"cameras/{camera}/rgb"]
    if ds.ndim == 1:
        decoded = cv2.imdecode(np.asarray(ds[frame], np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("invalid JPEG")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return np.asarray(ds[frame], np.uint8)


def metrics(predicted, target, hold, names):
    error = np.abs(predicted - target)
    baseline = np.abs(hold - target)
    groups = {
        "left_arm": [i for i, n in enumerate(names) if n.startswith("xarm_l_")],
        "right_arm": [i for i, n in enumerate(names) if n.startswith("multi_xarm_r_")],
        "left_hand": [i for i, n in enumerate(names) if "xarm" not in n and not n.startswith("multi_")],
        "right_hand": [i for i, n in enumerate(names) if "xarm" not in n and n.startswith("multi_")],
    }
    assert sorted(i for ids in groups.values() for i in ids) == list(range(len(names)))
    return {
        "mae_rad": float(error.mean()), "hold_mae_rad": float(baseline.mean()),
        "rmse_rad": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "per_horizon_mae_rad": error.mean(axis=(0, 2)).tolist(),
        "per_joint_mae_rad": dict(zip(names, error.mean(axis=(0, 1)).tolist())),
        "groups": {g: {"mae_rad": float(error[..., ids].mean()),
                       "hold_mae_rad": float(baseline[..., ids].mean())} for g, ids in groups.items()},
    }


def run(args):
    from cosmos_policy.datasets.bench2dex_dataset import Bench2DexDataset, CAMERAS, inspect_episode
    from cosmos_policy.experiments.robot.bench2dex.runtime import (
        Bench2DexPolicyAdapter, _decode_observation, _load_inputs, _make_predictor,
    )

    baseline = json.loads(args.baseline.read_text())
    argv = baseline["argv"]
    value = lambda flag: argv[argv.index(flag) + 1]
    task = value("--task-id")
    runtime = SimpleNamespace(
        checkpoint=value("--checkpoint"), manifest=value("--manifest"),
        robot_map=value("--robot-map"), embedding=value("--embedding"), task_id=task,
        embodiment_manifest=value("--embodiment-manifest"),
        config_file="cosmos_policy/config/config.py", trained_with_image_aug=False,
        image_resize_mode="cv2-area", denoising_steps=5,
    )
    profile, statistics, description, embedding, _ = _load_inputs(runtime)
    runtime.task_description = description
    runtime.experiment = f"cosmos_predict2_2b_480p_bench2dex_{profile.robot_key}"
    os.environ.update(
        COSMOS_INTERNAL="1", BENCH2DEX_EMBODIMENT_MANIFEST=runtime.embodiment_manifest,
        BENCH2DEX_TASK_ID=task, BENCH2DEX_ACTIVE_JOINTS=str(len(profile.active_joint_names)),
        BENCH2DEX_MANIFEST=runtime.manifest, BENCH2DEX_ROBOT_MAP=runtime.robot_map,
        BENCH2DEX_EMBEDDING=runtime.embedding,
        COSMOS_POLICY_PRETRAINED_CHECKPOINT=value("--base-model"),
        COSMOS_POLICY_TOKENIZER_CHECKPOINT=value("--tokenizer"),
    )
    assert str(args.output.resolve()).startswith("/inspire/hdd2/")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"task_id": task, "checkpoint": runtime.checkpoint, "prompt": description,
                "preprocessing": "no-crop-cv2-area", "denoising_steps": 5, "chunk_size": 16,
                "episodes_requested": args.episodes, "seeds": args.seeds,
                "physics": "not used: offline expert observations", "scope": "training-data diagnostic, not held-out evaluation",
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "active_joint_names": list(profile.active_joint_names),
                "statistics": statistics.to_dict()}
    write_json(args.output / "protocol.json", protocol)
    audit = json.loads(Path(runtime.manifest).read_text())
    records, infos = [], {}
    for entry in audit["entries"]:
        path = Path(entry["source"])
        info = inspect_episode(path, profile)
        infos[str(path)] = info
        with h5py.File(path, "r") as file:
            commands = np.asarray(file["action/commanded"][:], np.float32)[:, info.action_indices]
            record = {"source": str(path), "file_sha256_from_training_audit": entry["sha256"],
                      "command_sha256": hashlib.sha256(commands.tobytes()).hexdigest(),
                      "prompt_matches": entry["embedded_instruction"] == description,
                      "embedded_instruction": entry["embedded_instruction"],
                      "frames": info.frame_count, "cutoff": info.cutoff}
            if task == "79":
                piano = file["objects/obj_325_piano_1"]
                names = [x.decode() if isinstance(x, bytes) else str(x) for x in piano["joint_names"][:]]
                record["melody"] = melody_progress(piano["qpos"][:], names, info.observation_valid)
            records.append(record)
    selected = choose_sources(records, args.episodes)
    write_json(args.output / "source-audit.json", {
        "source_count": len(records), "unique_command_trajectories": len({r["command_sha256"] for r in records}),
        "selected": selected, "all_sources": records,
    })
    dataset = Bench2DexDataset([r["source"] for r in selected], profile, description,
                              text_embedding=embedding, statistics=statistics)
    lookup = {(ep, start): i for i, (ep, start, _) in enumerate(dataset._index)}
    windows = []
    for ep, record in enumerate(selected):
        info = infos[record["source"]]
        chosen = {frame: "uniform" for frame in full_windows(info.action_runs)}
        for side in ("left", "right"):
            events = record.get("melody", {}).get(side, {}).get("event_frames", [])
            if events:
                frame = max(0, events[0] - 4)
                if any(a <= frame and frame + 16 <= b for a, b in info.action_runs):
                    chosen[frame] = side + "_first_press_window"
        for frame, stage in sorted(chosen.items()):
            windows.append({"episode": ep, "frame": frame, "stage": stage,
                            "source": record["source"], "prompt_matches": record["prompt_matches"]})
    write_json(args.output / "windows.json", windows)
    print(f"DATA_READY task={task} sources={len(records)} unique={len({r['command_sha256'] for r in records})} windows={len(windows)}", flush=True)
    predictor = _make_predictor(runtime, profile, statistics, embedding)
    from cosmos_policy.experiments.robot.cosmos_utils import prepare_images_for_model
    captured = {}

    def capture(observation, instruction, seed):
        captured["normalized"] = predictor(observation, instruction, seed)
        return captured["normalized"]

    adapter = Bench2DexPolicyAdapter(capture, profile, statistics, description,
                                    runtime_joint_names=profile.full_joint_names)
    active_columns = [profile.full_joint_names.index(n) for n in profile.active_joint_names]
    predictions, targets, holds, rows = [], [], [], []
    image_cfg = SimpleNamespace(use_jpeg_compression=False, trained_with_image_aug=False, image_resize_mode="cv2-area")
    for window_index, window in enumerate(windows):
        info = infos[window["source"]]
        t = window["frame"]
        sample = dataset[lookup[window["episode"], t]]
        assert bool(sample["action_padding_mask"].sum() == 0)
        with h5py.File(window["source"], "r") as file:
            images = [raw_image(file, camera, t) for camera in CAMERAS]
            qpos = np.asarray(file["robot/qpos"][t], np.float32)[info.joint_indices]
            target = np.asarray(file["action/commanded"][t:t + 16], np.float32)[:, info.action_indices]
        payload = {"observation": {c: {"rgb": image} for c, image in zip(CAMERAS, images)},
                   "joint_action": {"vector": qpos}, "language": description}
        observation, _ = _decode_observation(payload, description, profile, statistics)
        np.testing.assert_array_equal(observation["proprio"], sample["proprio"])
        np.testing.assert_array_equal(statistics.normalize_actions(target), sample["actions"])
        np.testing.assert_allclose(statistics.denormalize_actions(sample["actions"]), target, atol=1e-6, rtol=0)
        processed = prepare_images_for_model(images, image_cfg)
        for i in range(4):
            np.testing.assert_array_equal(processed[i], sample["video"][:, 5 + 4 * i].transpose(1, 2, 0))
        for seed in args.seeds:
            adapter.reset(seed)
            begin = time.monotonic()
            expanded = adapter.get_action_chunk(payload)
            predicted = expanded[:, active_columns]
            hold = np.repeat(qpos[None], 16, axis=0)
            np.testing.assert_array_equal(predicted, statistics.denormalize_actions(captured["normalized"]))
            normalized_error = np.abs(captured["normalized"] - sample["actions"])
            row = {**window, "window_index": window_index, "seed": seed,
                   "seconds": time.monotonic() - begin,
                   "mae_rad": float(np.abs(predicted - target).mean()),
                   "hold_mae_rad": float(np.abs(hold - target).mean()),
                   "normalized_mae": float(normalized_error.mean()),
                   "input_contract_passed": True}
            np.savez_compressed(args.output / f"prediction-{window_index:04d}-seed{seed}.npz",
                                predicted=predicted, target=target, hold=hold,
                                normalized_predicted=captured["normalized"], normalized_target=sample["actions"])
            with (args.output / "predictions.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            rows.append(row); predictions.append(predicted); targets.append(target); holds.append(hold)
            summary = metrics(np.stack(predictions), np.stack(targets), np.stack(holds), profile.active_joint_names)
            summary.update(completed=len(rows), expected=len(windows) * len(args.seeds),
                           normalized_mae=float(np.mean([r["normalized_mae"] for r in rows])),
                           status="running", input_contract_passed=True)
            write_json(args.output / "summary.json", summary)
            print(f"PREDICTION {len(rows)}/{summary['expected']} task={task} mae_rad={row['mae_rad']:.6f} hold={row['hold_mae_rad']:.6f}", flush=True)
    summary["status"] = "complete"
    summary["by_metadata_prompt_match"] = {}
    for matching in (False, True):
        indices = [i for i, row in enumerate(rows) if row["prompt_matches"] == matching]
        if indices:
            summary["by_metadata_prompt_match"][str(matching)] = metrics(
                np.stack(predictions)[indices], np.stack(targets)[indices], np.stack(holds)[indices], profile.active_joint_names)
    write_json(args.output / "summary.json", summary)
    print("COMPLETE", task, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 107])
    run(parser.parse_args())
