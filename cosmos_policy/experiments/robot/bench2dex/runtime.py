# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Bench2Dex's length-prefixed RPC adapter for Cosmos task 03 inference."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import socketserver
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from cosmos_policy.datasets.bench2dex_audit import TaskStatistics
from cosmos_policy.datasets.bench2dex_dataset import CAMERAS, CHUNK_SIZE, RobotProfile

logger = logging.getLogger("bench2dex_runtime")
MAX_MESSAGE_BYTES = 64 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30.0


def _decode_observation(payload: dict[str, Any], expected_prompt: str, profile: RobotProfile, statistics: TaskStatistics):
    """Map Bench2Dex's raw remote observation to the task03 policy inputs."""
    camera_data = payload.get("observation")
    if not isinstance(camera_data, dict):
        raise ValueError("remote observation must contain the named camera mapping")
    images: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        entry = camera_data.get(camera)
        image = entry.get("rgb") if isinstance(entry, dict) else None
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"missing or invalid RGB image for {camera}")
        if image.dtype != np.uint8 or not image.size:
            raise ValueError(f"{camera} must be a nonempty uint8 RGB image")
        images[camera] = np.ascontiguousarray(image)
    if len({image.shape for image in images.values()}) != 1:
        raise ValueError("all four Bench2Dex RGB views must have the same shape")

    joint_action = payload.get("joint_action")
    qpos = joint_action.get("vector") if isinstance(joint_action, dict) else None
    if qpos is None:
        raise ValueError("joint_action.vector is required")
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape != (len(profile.active_joint_names),) or not np.isfinite(qpos).all():
        raise ValueError(f"joint_action.vector must contain {len(profile.active_joint_names)} finite active-joint positions")

    instruction = payload.get("language") or payload.get("instruction")
    if instruction != expected_prompt:
        raise ValueError("runtime task description does not match the audited full task description")

    return {
        "left_wrist_image": images["cam_wrist_left"],
        "right_wrist_image": images["cam_wrist_right"],
        "primary_image": images["cam_stereo_left"],
        "secondary_image": images["cam_stereo_right"],
        "proprio": statistics.normalize_proprio(qpos),
    }, instruction


class Bench2DexPolicyAdapter:
    """Validate observations, query Cosmos once, and expand 16 absolute targets."""

    def __init__(
        self,
        predict: Callable[[dict[str, Any], str, int], np.ndarray],
        profile: RobotProfile,
        statistics: TaskStatistics,
        task_description: str,
        *,
        runtime_joint_names: tuple[str, ...] | list[str],
        seed: int = 0,
    ) -> None:
        if statistics.robot_key != profile.robot_key or statistics.active_joint_names != profile.active_joint_names:
            raise ValueError("statistics do not match the task03 robot profile")
        if not task_description.strip():
            raise ValueError("the full task description is required")
        if CHUNK_SIZE != 16 or len(profile.active_joint_names) != 26:
            raise ValueError("task03 runtime requires 16 actions across 26 active joints")
        if len(set(runtime_joint_names)) != len(runtime_joint_names) or set(runtime_joint_names) != set(
            profile.full_joint_names
        ):
            raise ValueError("runtime joint names do not match the robot profile")
        self.predict = predict
        self.profile = profile
        self.statistics = statistics
        self.task_description = task_description
        self.runtime_joint_names = tuple(runtime_joint_names)
        self.base_seed = int(seed)
        self.query_index = 0

    def reset(self, seed: int | None = None) -> None:
        self.base_seed = self.base_seed if seed is None else int(seed)
        self.query_index = 0

    def update_after_action(self, observation: dict[str, Any]) -> None:
        # The runtime passes this after each open-loop action. The next query
        # carries the freshest observation, so no model state needs updating.
        del observation

    def get_action_chunk(self, payload: dict[str, Any]) -> np.ndarray:
        observation, instruction = _decode_observation(payload, self.task_description, self.profile, self.statistics)
        normalized = np.asarray(
            self.predict(observation, instruction, self.base_seed + self.query_index), dtype=np.float32
        )
        expected_shape = (CHUNK_SIZE, len(self.profile.active_joint_names))
        if normalized.shape != expected_shape or not np.isfinite(normalized).all():
            raise ValueError(f"Cosmos must return {expected_shape} finite normalized active-joint targets")
        actions = self.statistics.denormalize_actions(normalized)
        commands = self.profile.expand(actions, self.runtime_joint_names)
        self.query_index += 1
        return np.ascontiguousarray(commands, dtype=np.float32)


def profile_with_urdf_mimics(
    profile: RobotProfile,
    urdf_path: str | Path,
    expected_mimic_count: int,
) -> RobotProfile:
    """Attach topologically ordered URDF mimic rules to the audited profile."""
    path = Path(urdf_path)
    if expected_mimic_count == 0:
        return profile
    if not path.is_file():
        raise FileNotFoundError(f"cannot expand {expected_mimic_count} mimic joints: URDF not found at {path}")
    name_to_index = {name: index for index, name in enumerate(profile.full_joint_names)}
    raw_rules: dict[str, tuple[str, float, float]] = {}
    for joint in ET.parse(path).getroot().findall(".//joint"):
        mimic = joint.find("mimic")
        name = joint.get("name", "")
        source = mimic.get("joint", "") if mimic is not None else ""
        if mimic is not None and name in name_to_index and source in name_to_index:
            raw_rules[name] = (
                source,
                float(mimic.get("multiplier", "1.0")),
                float(mimic.get("offset", "0.0")),
            )
    resolved = set(profile.active_joint_names)
    rules = []
    while raw_rules:
        ready = [name for name, (source, _, _) in raw_rules.items() if source in resolved]
        if not ready:
            raise ValueError(f"URDF mimic rules contain unresolved sources: {sorted(raw_rules)}")
        for name in ready:
            source, multiplier, offset = raw_rules.pop(name)
            rules.append((name, source, multiplier, offset))
            resolved.add(name)
    if len(rules) != expected_mimic_count:
        raise ValueError(f"expected {expected_mimic_count} mimic rules in {path}, found {len(rules)}")
    return RobotProfile(profile.robot_key, profile.full_joint_names, profile.active_joint_names, tuple(rules))


def handle_rpc_command(adapter: Bench2DexPolicyAdapter, message: dict[str, Any]) -> dict[str, Any]:
    command = message.get("command")
    if command == "reset":
        adapter.reset(message.get("seed"))
        return {"result": True}
    if command == "update_after_action":
        adapter.update_after_action(message.get("observation", {}))
        return {"result": True}
    if command == "get_action_chunk":
        observation = message.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("get_action_chunk requires an observation object")
        return {"result": adapter.get_action_chunk(observation)}
    if command == "close":
        return {"result": True, "close": True}
    raise ValueError(f"unsupported Bench2Dex policy command: {command!r}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _object_hook(item: dict[str, Any]) -> Any:
    if item.get("__numpy_array__"):
        raw = base64.b64decode(item["data"], validate=True)
        array = np.frombuffer(raw, dtype=np.dtype(item["dtype"]))
        if array.size != int(np.prod(item["shape"], dtype=np.int64)):
            raise ValueError("invalid numpy array payload size")
        return array.reshape(item["shape"]).copy()
    return item


def _receive_exact(sock: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(min(size - len(chunks), 1024 * 1024))
        if not chunk:
            if not chunks:
                return None
            raise ConnectionError("truncated Bench2Dex RPC frame")
        chunks.extend(chunk)
    return bytes(chunks)


class _PolicyRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.settimeout(self.server.request_timeout_seconds)  # type: ignore[attr-defined]
        adapter: Bench2DexPolicyAdapter = self.server.adapter  # type: ignore[attr-defined]
        while True:
            header = _receive_exact(self.request, 4)
            if header is None:
                return
            message_size = int.from_bytes(header, "big")
            if message_size <= 0 or message_size > MAX_MESSAGE_BYTES:
                raise ValueError(f"invalid Bench2Dex RPC message size: {message_size}")
            payload = _receive_exact(self.request, message_size)
            if payload is None:
                raise ConnectionError("missing Bench2Dex RPC message body")
            should_close = False
            try:
                message = json.loads(payload.decode("utf-8"), object_hook=_object_hook)
                if not isinstance(message, dict):
                    raise ValueError("Bench2Dex RPC message must be an object")
                response = handle_rpc_command(adapter, message)
                should_close = bool(response.get("close"))
            except Exception as exc:
                logger.exception("Bench2Dex RPC request failed")
                response = {"error": str(exc)}
            encoded = json.dumps(response, cls=_NumpyEncoder).encode("utf-8")
            self.request.sendall(len(encoded).to_bytes(4, "big") + encoded)
            if should_close:
                return


class Bench2DexTCPServer(socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        adapter: Bench2DexPolicyAdapter,
        request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        if request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        self.adapter = adapter
        self.request_timeout_seconds = request_timeout_seconds
        super().__init__(address, _PolicyRequestHandler)


def _make_predictor(args: argparse.Namespace, profile: RobotProfile, statistics: TaskStatistics, embedding: np.ndarray):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Bench2Dex task03 Cosmos inference requires CUDA")

    from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model

    cfg = SimpleNamespace(
        ckpt_path=args.checkpoint,
        config=args.experiment,
        config_file=args.config_file,
    )
    model, cosmos_config = get_model(cfg)
    expected_chunk = cosmos_config.dataloader_train.dataset.chunk_size
    if expected_chunk != CHUNK_SIZE:
        raise ValueError(f"checkpoint expects action chunk {expected_chunk}, task03 requires {CHUNK_SIZE}")
    action_dim = len(profile.active_joint_names)
    if action_dim != 26 or len(statistics.active_joint_names) != action_dim:
        raise ValueError("task03 checkpoint statistics must describe 26 active joints")

    inference_cfg = SimpleNamespace(
        suite="bench2dex",
        use_jpeg_compression=False,
        trained_with_image_aug=args.trained_with_image_aug,
        use_proprio=True,
        normalize_proprio=False,
        use_wrist_image=True,
        num_wrist_images=2,
        use_third_person_image=True,
        num_third_person_images=2,
        use_variance_scale=False,
        chunk_size=CHUNK_SIZE,
        unnormalize_actions=False,
    )

    def predict(observation: dict[str, Any], instruction: str, seed: int) -> np.ndarray:
        if instruction != args.task_description:
            raise ValueError("runtime instruction differs from the model's audited prompt")
        result = get_action(
            inference_cfg,
            model,
            {},
            observation,
            embedding,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=args.denoising_steps,
            generate_future_state_and_value_in_parallel=False,
        )
        return np.asarray(result["actions"], dtype=np.float32)

    return predict


def _load_inputs(args: argparse.Namespace) -> tuple[RobotProfile, TaskStatistics, str, np.ndarray]:
    import yaml

    from cosmos_policy.datasets.bench2dex_audit import load_profile

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not manifest.get("ready") or manifest.get("matched_count") != 100:
        raise ValueError("task03 inference requires the ready audit of 100 matched demonstrations")
    profile = load_profile(args.robot_map, manifest["robot_key"])
    map_path = Path(args.robot_map).expanduser().resolve()
    robot_entry = yaml.safe_load(map_path.read_text(encoding="utf-8"))["robots"][profile.robot_key]
    mimic_count = int(robot_entry.get("mimic_dof", 0))
    if mimic_count:
        dataset_root = os.environ.get("DEX2BENCH_DATASET_ROOT")
        base = Path(dataset_root).expanduser() if dataset_root else map_path.parents[2] / "dex2bench_dataset"
        urdf_rel = robot_entry.get("urdf_path", "").replace("Bench2Dex/", "", 1)
        profile = profile_with_urdf_mimics(profile, base / urdf_rel, mimic_count)
    statistics = TaskStatistics.from_dict(manifest["statistics"])
    if profile.active_joint_names != statistics.active_joint_names or len(profile.active_joint_names) != 26:
        raise ValueError("manifest statistics do not match the 26-joint robot profile")
    with np.load(args.embedding, allow_pickle=False) as saved:
        prompt = str(saved["prompt"])
        embedding = np.asarray(saved["embedding"], dtype=np.float32)
    description = manifest["task_description"]
    if prompt != description or embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("text embedding must match the audited full task description")
    return profile, statistics, description, embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a task03 Cosmos policy to Bench2Dex REMOTE evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--robot-map", required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--experiment", default="cosmos_predict2_2b_480p_bench2dex_task03")
    parser.add_argument("--config-file", default="cosmos_policy/config/config.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--denoising-steps", type=int, default=5)
    parser.add_argument(
        "--trained-with-image-aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the deterministic center crop used during task03 training (default: enabled).",
    )
    parser.add_argument("--checkpoint-id", default="", help="Stable label to record in simulator result metadata.")
    args = parser.parse_args()

    if args.denoising_steps <= 0:
        parser.error("--denoising-steps must be positive")
    profile, statistics, description, embedding = _load_inputs(args)
    args.task_description = description
    args.checkpoint_id = args.checkpoint_id or str(Path(args.checkpoint).expanduser().resolve())
    predictor = _make_predictor(args, profile, statistics, embedding)
    adapter = Bench2DexPolicyAdapter(
        predictor,
        profile,
        statistics,
        description,
        runtime_joint_names=profile.full_joint_names,
        seed=args.seed,
    )
    logger.info(
        "Serving Bench2Dex task03: robot=%s, active_joints=%d, full_joints=%d, chunk=%d, checkpoint_id=%s, checkpoint=%s",
        profile.robot_key,
        len(profile.active_joint_names),
        len(profile.full_joint_names),
        CHUNK_SIZE,
        args.checkpoint_id,
        Path(args.checkpoint).expanduser().resolve(),
    )
    with Bench2DexTCPServer((args.host, args.port), adapter) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Stopping Bench2Dex task03 policy server")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
