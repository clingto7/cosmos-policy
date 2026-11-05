#!/usr/bin/env python3
"""Run one selected Cosmos checkpoint on Bench2Dex's four-channel protocol.

Run on a GPU instance with the Cosmos and Bench2Dex Python environments. Each
simulator invocation writes to a distinct attempt directory, so interruption
never overwrites an accepted episode or its rollout recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
from pathlib import Path

CHANNELS = ("none", "cov_only", "inv_only", "inv_cov")
FINAL_SEED = 100_000_000
FULL_SEED = 200_000_000
PILOT_SEED = 300_000_000
PILOT_FULL_SEED = 400_000_000


def fail(message: str) -> None:
    raise SystemExit(message)


def budget(scene: Path) -> tuple[int, int, float]:
    source = scene.read_text()
    match = re.search(r"^\s*expert_time_step\s*:\s*(\d+(?:\.\d+)?)", source, re.M)
    if match:
        expert_steps = float(match.group(1))
    else:
        match = re.search(r"^\s*expert_time_s\s*:\s*(\d+(?:\.\d+)?)", source, re.M)
        if not match:
            fail(f"scene has no expert time for the reference budget: {scene}")
        expert_steps = float(match.group(1)) * 60
    policy_steps = math.ceil(expert_steps * 1.5 / 3)
    return policy_steps, policy_steps * 3, expert_steps


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def accepted_attempts(channel_dir: Path, channel: str, task_id: str,
                      seed_pair: tuple[int, int], target: int = 50) -> dict[int, tuple[dict, Path]]:
    accepted: dict[int, tuple[dict, Path]] = {}
    base_seed = (seed_pair[1] if channel == "inv_cov" else seed_pair[0]) + int(task_id) * 100_000
    for attempt in sorted((channel_dir / "attempts").glob("attempt-*")):
        for row in rows(attempt / "results" / "per_episode.jsonl"):
            index = row.get("episode_index")
            if not isinstance(index, int) or not 1 <= index <= target:
                fail(f"invalid episode index in {attempt}: {index!r}")
            expected_seed = base_seed + index - 1
            if row.get("episode_seed") != expected_seed or row.get("base_seed") != base_seed:
                fail(f"seed mismatch for episode {index} in {attempt}")
            if row.get("error") is not None:
                continue
            if not isinstance(row.get("success"), bool):
                fail(f"missing success flag for episode {index} in {attempt}")
            if not isinstance(row.get("latched_stage_completion_rate"), (int, float)):
                fail(f"missing LSCR for episode {index} in {attempt}")
            recordings = list((attempt / "episodes").rglob(f"episode_{index:06d}.hdf5"))
            if len(recordings) != 1 or not recordings[0].stat().st_size:
                continue  # incomplete recording is an infrastructure failure; retry this seed
            if index in accepted:
                fail(f"duplicate valid episode {index} in {channel_dir}")
            accepted[index] = row, attempt
    return accepted


def first_gap(accepted: dict[int, tuple[dict, Path]], target: int = 50) -> tuple[int, int] | None:
    for start in range(1, target + 1):
        if start not in accepted:
            count = 0
            while start + count <= target and start + count not in accepted and count < 10:
                count += 1
            return start, count
    return None


def write_channel(channel_dir: Path, accepted: dict[int, tuple[dict, Path]], expert_steps: float) -> dict:
    ordered = [accepted[index][0] for index in sorted(accepted)]
    (channel_dir / "per_episode.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
    )
    successes = [row for row in ordered if row.get("success") is True]
    success_steps = [row.get("steps_to_stable_success") for row in successes]
    success_steps = [float(value) for value in success_steps if isinstance(value, (int, float)) and value > 0]
    lscr_values = [row.get("latched_stage_completion_rate") for row in ordered]
    if any(not isinstance(value, (int, float)) for value in lscr_values):
        fail(f"missing LSCR in {channel_dir}")
    if successes and len(success_steps) != len(successes):
        fail(f"missing stable-success steps in {channel_dir}")
    result = {
        "episodes": len(ordered),
        "successes": len(successes),
        "sr": len(successes) / len(ordered) if ordered else None,
        "lscr": sum(lscr_values) / len(ordered) if ordered else None,
        "eff": expert_steps / (sum(success_steps) / len(success_steps)) if success_steps else None,
        "successful_step_sum": sum(success_steps),
        "errors": 0,
    }
    (channel_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    attempts = []
    for attempt in sorted((channel_dir / "attempts").glob("attempt-*")):
        attempt_rows = rows(attempt / "results" / "per_episode.jsonl")
        attempts.append({"path": str(attempt),
                         "exit_code": (attempt / "exit-code.txt").read_text().strip() if (attempt / "exit-code.txt").exists() else None,
                         "error_episodes": [row.get("episode_index") for row in attempt_rows if row.get("error") is not None],
                         "missing_recording_episodes": [row.get("episode_index") for row in attempt_rows
                             if row.get("error") is None and not list((attempt / "episodes").rglob(f"episode_{row['episode_index']:06d}.hdf5"))]})
    (channel_dir / "attempts.json").write_text(json.dumps(attempts, indent=2) + "\n")
    return result


def export_videos(args: argparse.Namespace, channel_dir: Path, accepted: dict[int, tuple[dict, Path]]) -> None:
    video_dir = channel_dir / "videos"
    video_dir.mkdir(exist_ok=True)
    exporter = args.bench_repo / "tools/export/export_video.py"
    for index in range(1, args.episodes_per_channel + 1):
        _, attempt = accepted[index]
        recordings = list((attempt / "episodes").rglob(f"episode_{index:06d}.hdf5"))
        if len(recordings) != 1:
            fail(f"expected one recording for {channel_dir.name} episode {index}; found {len(recordings)}")
        video = video_dir / f"episode_{index:06d}_grid_realtime.mp4"
        if video.is_file() and video.stat().st_size:
            continue
        with (channel_dir / "video-export.log").open("a") as log:
            subprocess.run([str(args.sim_python), str(exporter), str(recordings[0]),
                            "--output", str(video), "--fps", "7", "--cell-width", "256",
                            "--cell-height", "192"], cwd=args.bench_repo, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    if len([video for video in video_dir.glob("*_grid_realtime.mp4") if video.stat().st_size]) != args.episodes_per_channel:
        fail(f"expected {args.episodes_per_channel} videos in {video_dir}")


def wait_for_policy(process: subprocess.Popen, port: int, timeout: int = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            fail(f"policy server exited with code {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(2)
    fail("policy server readiness timeout")


def run(args: argparse.Namespace) -> None:
    if args.episodes_per_channel < 1 or (args.seed_namespace == "final" and args.episodes_per_channel != 50):
        fail("pilot episodes must be positive; final evaluation requires 50 per channel")
    seed_pair = (PILOT_SEED, PILOT_FULL_SEED) if args.seed_namespace == "pilot" else (FINAL_SEED, FULL_SEED)
    if not re.fullmatch(r"\d{2}", args.task_id) or not args.scene.name.startswith(args.task_id + "_"):
        fail("task ID must match the two-digit scene prefix")
    manifest = json.loads(args.manifest.read_text())
    if not manifest.get("ready") or manifest.get("task_id") != args.task_id:
        fail("task manifest is not ready or task ID differs")
    embodiment_manifest = getattr(args, "embodiment_manifest", None)
    if embodiment_manifest:
        # Validate using the policy environment, which owns numpy/torch/DCP.
        subprocess.run([
            str(args.policy_python), "-c",
            "import sys; from cosmos_policy.experiments.robot.bench2dex.checkpoints import require_checkpoint_contract; "
            "require_checkpoint_contract(sys.argv[1], sys.argv[2], final=True)",
            str(args.checkpoint), str(embodiment_manifest),
        ], cwd=args.policy_repo, check=True)
    if not args.checkpoint.joinpath("model", ".metadata").is_file():
        fail("checkpoint model metadata missing")
    if not args.anchor_dir.is_dir():
        fail("matched anchor directory missing")
    anchor_files = {path.name for path in args.anchor_dir.glob("episode_*.hdf5")}
    expected_anchors = {f"episode_{index:06d}.hdf5" for index in range(25)} | {
        f"episode_{index:06d}_1.hdf5" for index in range(25, 50)
    }
    if not expected_anchors <= anchor_files:
        fail(f"reference 50-anchor set is incomplete: {args.anchor_dir}")
    if not args.embedding.is_file():
        fail("task prompt embedding missing")
    if not args.base_model.is_file() or not args.tokenizer.is_file():
        fail("base model or tokenizer missing")
    if not args.scene.is_file():
        fail("task scene missing")
    if not args.robot_map.is_file():
        fail("active-DOF robot map missing")
    if not args.policy_python.is_file() or not args.sim_python.is_file():
        fail("policy or simulator Python missing")
    if not (args.bench_repo / "run_policy.py").is_file():
        fail("Bench2Dex runner missing")
    if not (args.bench_repo / "tools/export/export_video.py").is_file():
        fail("Bench2Dex video exporter missing")
    policy_steps, physics_steps, expert_steps = budget(args.scene)
    args.output.mkdir(parents=True, exist_ok=True)
    if str(args.output.resolve()).startswith("/inspire/") and not str(args.output.resolve()).startswith("/inspire/hdd2/"):
        fail("evaluation output must resolve under private HDD2")
    protocol = {
        "task_id": args.task_id,
        "task_name": args.scene.stem.split("_", 1)[-1].replace("_", " ").title(),
        "robot_key": manifest["robot_key"],
        "checkpoint": str(args.checkpoint.resolve()),
        "base_model": str(args.base_model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "embedding_sha256": file_sha256(args.embedding),
        "evaluation_runner_sha256": file_sha256(Path(__file__)),
        "cosmos_revision": git_head(args.policy_repo),
        "bench2dex_revision": git_head(args.bench_repo),
        "scene": str(args.scene.resolve()),
        "anchor_dir": str(args.anchor_dir.resolve()),
        "channels": list(CHANNELS),
        "physics_device": "cpu",
        "episodes_per_channel": args.episodes_per_channel,
        "policy_steps": policy_steps,
        "physics_steps": physics_steps,
        "expert_steps": expert_steps,
        "seed_namespace": args.seed_namespace,
        "seed_pair": list(seed_pair),
    }
    protocol_file = args.output / "protocol.json"
    if embodiment_manifest:
        protocol["embodiment_manifest_sha256"] = file_sha256(embodiment_manifest)
        protocol["training_scope"] = "embodiment"
    if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
        fail("existing output was created with a different protocol")
    protocol_file.write_text(json.dumps(protocol, indent=2) + "\n")
    for channel in CHANNELS:
        (args.output / channel / "attempts").mkdir(parents=True, exist_ok=True)

    policy_log = (args.output / "policy-server.log").open("a")
    try:
        with socket.create_connection(("127.0.0.1", args.port), timeout=1):
            fail(f"policy port {args.port} already in use")
    except OSError:
        pass
    command = [str(args.policy_python), "-m", "cosmos_policy.experiments.robot.bench2dex.runtime",
        "--checkpoint", str(args.checkpoint), "--manifest", str(args.manifest),
        "--robot-map", str(args.robot_map), "--embedding", str(args.embedding),
        "--task-id", args.task_id, "--checkpoint-id", args.checkpoint_id,
        "--host", "127.0.0.1", "--port", str(args.port), "--seed", "7"]
    if embodiment_manifest:
        command.extend(["--embodiment-manifest", str(embodiment_manifest)])
    environment = os.environ.copy()
    environment.update(COSMOS_INTERNAL="1", CUDA_HOME=str(args.policy_python.parent.parent), PYTHONUNBUFFERED="1",
        COSMOS_POLICY_PRETRAINED_CHECKPOINT=str(args.base_model),
        COSMOS_POLICY_TOKENIZER_CHECKPOINT=str(args.tokenizer))
    libraries = subprocess.check_output([str(args.policy_python), "-c",
        "import nvidia,pathlib; p=pathlib.Path(nvidia.__file__).parent; print(':'.join(str(x/'lib') for x in p.iterdir() if (x/'lib').is_dir()))"],
        text=True).strip()
    environment["LD_LIBRARY_PATH"] = libraries + ":" + environment.get("LD_LIBRARY_PATH", "")
    policy = subprocess.Popen(command, cwd=args.policy_repo, env=environment, stdout=policy_log, stderr=subprocess.STDOUT)
    try:
        wait_for_policy(policy, args.port)
        for channel in CHANNELS:
            channel_dir = args.output / channel
            attempts = 0
            while True:
                if policy.poll() is not None:
                    fail(f"policy server exited with code {policy.returncode}")
                accepted = accepted_attempts(channel_dir, channel, args.task_id, seed_pair, args.episodes_per_channel)
                gap = first_gap(accepted, args.episodes_per_channel)
                write_channel(channel_dir, accepted, expert_steps)
                if gap is None:
                    export_videos(args, channel_dir, accepted)
                    break
                start, count = gap
                attempts += 1
                if attempts > args.max_attempts:
                    fail(f"too many failed simulator attempts in {channel}")
                attempt = channel_dir / "attempts" / f"attempt-{int(time.time())}-{start:03d}"
                attempt.mkdir()
                command = [str(args.sim_python), "run_policy.py", "--task", str(args.scene),
                    "--policy-type", "REMOTE", "--policy-name", "GR00T_XE",
                    "--policy-display-name", args.checkpoint_id, "--remote-host", "127.0.0.1",
                    "--remote-port", str(args.port), "--robot-key", manifest["robot_key"],
                    "--active-dof", "--enable-rgb", "--headless", "--device", "cpu", "--generalization-profile", channel,
                    "--generalization-split", "seen" if channel == "none" else "unseen",
                    "--seed", str(seed_pair[0]), "--inv-cov-seed", str(seed_pair[1]),
                    "--task-seed-id", str(int(args.task_id)), "--start-episode", str(start),
                    "--num-episodes", str(count), "--episode-steps", str(policy_steps),
                    "--warmup-steps", "60", "--record-all", "--record-dir", str(attempt / "episodes"),
                    "--output-dir", str(attempt / "results")]
                if channel != "inv_cov":
                    command.extend(("--anchor-dir", str(args.anchor_dir)))
                (attempt / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                with (attempt / "simulation.log").open("w") as log:
                    returncode = subprocess.call(command, cwd=args.bench_repo, stdout=log, stderr=subprocess.STDOUT)
                (attempt / "exit-code.txt").write_text(str(returncode) + "\n")
                # Successful task failures count toward SR; policy/simulator errors do not.
                print(f"{channel}: attempt {attempt.name} exit={returncode}; accepted={len(accepted_attempts(channel_dir, channel, args.task_id, seed_pair, args.episodes_per_channel))}/{args.episodes_per_channel}", flush=True)

        summaries = {channel: json.loads((args.output / channel / "summary.json").read_text()) for channel in CHANNELS}
        if any(value["episodes"] != args.episodes_per_channel for value in summaries.values()):
            fail("four-channel result is incomplete")
        result = {
            "task_id": args.task_id,
            "task_name": protocol["task_name"],
            "checkpoint_id": args.checkpoint_id,
            "seed_namespace": args.seed_namespace,
            "channels": summaries,
            "per_task_sr": sum(value["sr"] for value in summaries.values()) / 4,
            "per_task_lscr": sum(value["lscr"] for value in summaries.values()) / 4,
            "per_task_eff": None,
        }
        success_count = sum(value["successes"] for value in summaries.values())
        successful_step_sum = sum(value["successful_step_sum"] for value in summaries.values())
        if success_count:
            result["per_task_eff"] = expert_steps / (successful_step_sum / success_count)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        policy.terminate()
        try:
            policy.wait(timeout=20)
        except subprocess.TimeoutExpired:
            policy.kill()
        policy_log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task-id", "checkpoint-id"):
        parser.add_argument("--" + name, required=True)
    for name in ("manifest", "checkpoint", "embedding", "base-model", "tokenizer", "scene", "anchor-dir", "robot-map",
                 "policy-repo", "bench-repo", "policy-python", "sim-python", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--port", type=int, default=19006)
    parser.add_argument("--embodiment-manifest", type=Path, help="Shared embodiment training bundle (fixed 32k)")
    parser.add_argument("--seed-namespace", choices=("pilot", "final"), default="final")
    parser.add_argument("--episodes-per-channel", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
