# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Bench2Dex HDF5 episodes, named-joint actions, and Cosmos policy samples.

The first policy uses one robot key per dataset.  A profile is supplied by the
Bench2Dex robot registry rather than inferred from HDF5 column positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np

CAMERAS = (
    "cam_wrist_left",
    "cam_wrist_right",
    "cam_stereo_left",
    "cam_stereo_right",
)
CHUNK_SIZE = 16
SEGMENT_REPEAT = 4


def _string(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _names(dataset: h5py.Dataset, path: str) -> tuple[str, ...]:
    names = tuple(_string(item) for item in dataset[()])
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError(f"{path}: names must be nonempty and unique")
    return names


def _required(file: h5py.File, path: str) -> h5py.Dataset:
    item = file.get(path)
    if not isinstance(item, h5py.Dataset):
        raise ValueError(f"missing dataset /{path}")
    return item


@dataclass(frozen=True)
class RobotProfile:
    robot_key: str
    full_joint_names: tuple[str, ...]
    active_joint_names: tuple[str, ...]
    # (mimic name, source name, multiplier, offset), in dependency order.
    mimic_rules: tuple[tuple[str, str, float, float], ...] = ()

    def __post_init__(self) -> None:
        full = self.full_joint_names
        active = self.active_joint_names
        if not self.robot_key or len(set(full)) != len(full) or len(set(active)) != len(active):
            raise ValueError("robot profile needs a key and unique joint names")
        if not active or not set(active).issubset(full):
            raise ValueError("active joints must be a nonempty subset of full joints")
        resolved = set(active)
        for mimic, source, multiplier, offset in self.mimic_rules:
            if mimic not in full or mimic in resolved or source not in resolved:
                raise ValueError(f"invalid or unordered mimic rule: {mimic} <- {source}")
            if not np.isfinite(multiplier) or not np.isfinite(offset):
                raise ValueError(f"nonfinite mimic rule: {mimic}")
            resolved.add(mimic)

    def with_bench2dex_mimic_rules(self, info: object) -> RobotProfile:
        """Add runtime mimic rules while retaining the training joint order."""
        full = tuple(info.full_joint_names)
        if (
            info.robot_key != self.robot_key
            or set(full) != set(self.full_joint_names)
            or set(info.active_joint_names) != set(self.active_joint_names)
        ):
            raise ValueError("Bench2Dex ActiveDofInfo does not match robot profile")
        return RobotProfile(
            robot_key=self.robot_key,
            full_joint_names=self.full_joint_names,
            active_joint_names=self.active_joint_names,
            mimic_rules=tuple(
                (full[mimic], full[source], float(mult), float(offset))
                for mimic, source, mult, offset in info.mimic_rules
            ),
        )

    def indices(self, names: Sequence[str], field: str) -> np.ndarray:
        if len(set(names)) != len(names):
            raise ValueError(f"{field}: duplicate joint names")
        if set(names) != set(self.full_joint_names):
            raise ValueError(f"{field}: expected full joint set for {self.robot_key}")
        lookup = {name: index for index, name in enumerate(names)}
        return np.asarray([lookup[name] for name in self.active_joint_names], dtype=np.intp)

    def expand(self, actions: np.ndarray, runtime_joint_names: Sequence[str]) -> np.ndarray:
        """Convert active actions to runtime order and apply mimic relationships."""
        values = np.asarray(actions, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.active_joint_names):
            raise ValueError("policy action shape does not match active joints")
        if len(set(runtime_joint_names)) != len(runtime_joint_names) or set(runtime_joint_names) != set(
            self.full_joint_names
        ):
            raise ValueError("runtime joint names do not match robot profile")
        by_name = dict(zip(self.active_joint_names, values.T))
        for mimic, source, multiplier, offset in self.mimic_rules:
            by_name[mimic] = by_name[source] * multiplier + offset
        if set(by_name) != set(self.full_joint_names):
            missing = sorted(set(self.full_joint_names) - set(by_name))
            raise ValueError(f"cannot expand unmapped runtime joints: {missing}")
        commands = np.stack([by_name[name] for name in runtime_joint_names], axis=-1)
        if not np.isfinite(commands).all():
            raise ValueError("expanded runtime commands must be finite")
        return commands


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    robot_key: str
    joint_indices: np.ndarray
    action_indices: np.ndarray
    action_runs: tuple[tuple[int, int], ...]  # inclusive start, exclusive end
    observation_valid: np.ndarray
    cutoff: int
    frame_count: int

    @property
    def usable_frames(self) -> int:
        return sum(end - start for start, end in self.action_runs)

    @property
    def excluded_frames(self) -> int:
        return self.frame_count - self.cutoff

    @property
    def unusable_task_frames(self) -> int:
        return self.cutoff - self.usable_frames


def inspect_episode(path: str | Path, profile: RobotProfile) -> EpisodeInfo:
    """Fail closed on schema, names, timing, and homing; keep contiguous action runs."""
    path = Path(path)
    with h5py.File(path, "r") as file:
        robot_key = _string(_required(file, "meta/robot_key")[()])
        if robot_key != profile.robot_key:
            raise ValueError(f"robot_key={robot_key!r}, expected {profile.robot_key!r}")
        if _string(_required(file, "action/action_type")[()]) != "absolute":
            raise ValueError("action/action_type must be absolute")
        if _string(_required(file, "action/control_mode")[()]) != "joint_position":
            raise ValueError("action/control_mode must be joint_position")
        # Older collector files do not record units.  Joint-position control
        # defines radians; reject an explicit contradictory unit if present.
        unit = file.get("meta/action_unit")
        if unit is not None and _string(unit[()]) not in ("rad", "radian", "radians"):
            raise ValueError("meta/action_unit must be radians")
        fps = file.get("meta/effective_fps")
        if fps is not None and not np.isclose(float(fps[()]), 20.0, atol=0.1):
            raise ValueError("meta/effective_fps must be 20 Hz")

        joint_names = _names(_required(file, "robot/joint_names"), "robot/joint_names")
        action_names = _names(_required(file, "action/action_names"), "action/action_names")
        joint_indices = profile.indices(joint_names, "robot/joint_names")
        action_indices = profile.indices(action_names, "action/action_names")
        qpos = _required(file, "robot/qpos")
        commanded = _required(file, "action/commanded")
        if qpos.ndim != 2 or commanded.ndim != 2 or qpos.shape[1] != len(joint_names) or commanded.shape[1] != len(
            action_names
        ):
            raise ValueError("joint/action array widths do not match their names")
        if not np.issubdtype(qpos.dtype, np.number) or not np.issubdtype(commanded.dtype, np.number):
            raise ValueError("robot/qpos and action/commanded must be numeric")
        frame_count = qpos.shape[0]
        if frame_count < 2 or commanded.shape[0] != frame_count:
            raise ValueError("joint and action arrays need the same episode length >= 2")
        steps = _required(file, "time/sim_step")
        if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer):
            raise ValueError("time/sim_step must be a 1D integer array")
        sim_step = np.asarray(steps[:], dtype=np.int64)
        frame_valid = np.asarray(_required(file, "frame_valid")[:], dtype=bool)
        action_valid = np.asarray(_required(file, "action/action_valid")[:], dtype=bool)
        if any(values.ndim != 1 or len(values) != frame_count for values in (sim_step, frame_valid, action_valid)):
            raise ValueError("time and validity arrays must match episode length")
        if not np.all(np.diff(sim_step) > 0):
            raise ValueError("time/sim_step must increase strictly")
        homing = int(_required(file, "meta/homing_start_sim_step")[()])
        if homing <= int(sim_step[0]):
            raise ValueError("homing_start_sim_step must follow the first observation")
        cutoff = int(np.searchsorted(sim_step, homing, side="left"))
        if not 1 <= cutoff <= frame_count:
            raise ValueError("homing boundary leaves no task frames")

        image_shapes: set[tuple[int, ...]] = set()
        image_encodings: set[str] = set()
        for camera in CAMERAS:
            image = _required(file, f"cameras/{camera}/rgb")
            if image.shape[0] != frame_count:
                raise ValueError(f"cameras/{camera}/rgb length must match episode")
            if image.ndim == 4 and image.shape[-1] == 3 and image.dtype == np.uint8:
                image_shapes.add(tuple(image.shape[1:]))
                image_encodings.add("raw")
            elif image.ndim == 1 and h5py.check_dtype(vlen=image.dtype) == np.dtype("uint8"):
                image_encodings.add("jpeg")
            else:
                raise ValueError(f"cameras/{camera}/rgb needs raw RGB or variable-length JPEG bytes")
        if len(image_encodings) != 1 or len(image_shapes) > 1:
            raise ValueError("four cameras must have the same image shape")
        declared_encoding = file.get("meta/rgb_encoding")
        if declared_encoding is not None and _string(declared_encoding[()]) not in image_encodings:
            raise ValueError("meta/rgb_encoding differs from camera data")

        # h5py requires fancy column indices to increase; the canonical
        # active-joint order can differ from either HDF5 column order.
        observed = frame_valid & np.isfinite(np.asarray(qpos[:])[:, joint_indices]).all(axis=1)
        valid_action = observed & action_valid & np.isfinite(np.asarray(commanded[:])[:, action_indices]).all(axis=1)
        valid_action[cutoff:] = False
        observed[cutoff:] = False
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for index in range(cutoff + 1):
            is_valid = index < cutoff and bool(valid_action[index])
            if is_valid and start is None:
                start = index
            elif not is_valid and start is not None:
                runs.append((start, index))
                start = None
        if not runs:
            raise ValueError("episode has no valid task actions before homing")
        return EpisodeInfo(path, robot_key, joint_indices, action_indices, tuple(runs), observed, cutoff, frame_count)


def _image(file: h5py.File, camera: str, index: int, image_size: int) -> np.ndarray:
    dataset = file[f"cameras/{camera}/rgb"]
    if dataset.ndim == 1:
        import cv2

        encoded = np.asarray(dataset[index], dtype=np.uint8)
        frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError(f"cameras/{camera}/rgb[{index}] is not a valid JPEG")
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        frame = np.asarray(dataset[index], dtype=np.uint8)
    if frame.shape[:2] != (image_size, image_size):
        import cv2

        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return frame


def _video(current: list[np.ndarray], future: list[np.ndarray]) -> tuple[np.ndarray, dict[str, int]]:
    blank = np.zeros_like(current[0])
    segments = [blank, blank, *current, blank, blank, *future, blank]
    indices = {
        "current_proprio_latent_idx": 1,
        "current_wrist_image_latent_idx": 2,
        "current_wrist_image2_latent_idx": 3,
        "current_image_latent_idx": 4,
        "current_image2_latent_idx": 5,
        "action_latent_idx": 6,
        "future_proprio_latent_idx": 7,
        "future_wrist_image_latent_idx": 8,
        "future_wrist_image2_latent_idx": 9,
        "future_image_latent_idx": 10,
        "future_image2_latent_idx": 11,
        "value_latent_idx": 12,
    }
    assert len(segments) == 13
    frames = np.concatenate([segments[0][None], np.repeat(np.stack(segments[1:]), SEGMENT_REPEAT, axis=0)])
    return np.transpose(frames, (3, 0, 1, 2)).copy(), indices


class Bench2DexDataset:
    """Indexed task-specific HDF5 samples. HDF5 handles never survive a call."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        profile: RobotProfile,
        task_description: str,
        *,
        text_embedding: np.ndarray | None = None,
        image_size: int = 224,
        statistics: object | None = None,
    ) -> None:
        if not task_description.strip():
            raise ValueError("full task description is required")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.profile = profile
        self.task_description = task_description
        self.text_embedding = text_embedding
        self.image_size = image_size
        self.statistics = statistics
        if statistics is not None and (
            statistics.robot_key != profile.robot_key
            or tuple(statistics.active_joint_names) != profile.active_joint_names
        ):
            raise ValueError("statistics do not match robot key and active joint order")
        self.episodes = [inspect_episode(path, profile) for path in paths]
        if not self.episodes:
            raise ValueError("dataset needs at least one episode")
        self._index = [
            (episode_number, start, end)
            for episode_number, episode in enumerate(self.episodes)
            for run_start, end in episode.action_runs
            for start in range(run_start, end)
        ]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict:
        episode_number, start, end = self._index[index]
        info = self.episodes[episode_number]
        candidate_future = min(start + CHUNK_SIZE, end)
        future_frame = (
            candidate_future
            if candidate_future < info.cutoff and info.observation_valid[candidate_future]
            else end - 1
        )
        with h5py.File(info.path, "r") as file:
            proprio = np.asarray(file["robot/qpos"][start], dtype=np.float32)[info.joint_indices]
            future_proprio = np.asarray(file["robot/qpos"][future_frame], dtype=np.float32)[info.joint_indices]
            # h5py fancy indexing requires increasing unique indices. Read the
            # contiguous run and apply repeat-last padding in memory.
            available = min(CHUNK_SIZE, end - start)
            raw_actions = np.asarray(file["action/commanded"][start : start + available], dtype=np.float32)
            actions = raw_actions[:, info.action_indices]
            if available < CHUNK_SIZE:
                actions = np.concatenate([actions, np.repeat(actions[-1:], CHUNK_SIZE - available, axis=0)])
            current = [_image(file, camera, start, self.image_size) for camera in CAMERAS]
            future = [_image(file, camera, future_frame, self.image_size) for camera in CAMERAS]
        if self.statistics is not None:
            proprio = self.statistics.normalize_proprio(proprio)
            future_proprio = self.statistics.normalize_proprio(future_proprio)
            actions = self.statistics.normalize_actions(actions)
        video, indices = _video(current, future)
        sample = {
            "video": video,
            "command": self.task_description,
            "actions": actions.astype(np.float32),
            "proprio": proprio.astype(np.float32),
            "future_proprio": future_proprio.astype(np.float32),
            "fps": 16,  # Cosmos video conditioning convention; control remains 20 Hz.
            "padding_mask": np.zeros((1, self.image_size, self.image_size), dtype=np.float32),
            "image_size": np.full(4, self.image_size, dtype=np.float32),
            "value_function_return": -100.0,
            "value_function_sample_mask": 0,
            "world_model_sample_mask": 1,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "__key__": f"{info.path.name}:{start}",
            "action_padding_mask": np.arange(CHUNK_SIZE) >= available,
            "source_frame": start,
            "future_frame": future_frame,
            "camera_order": CAMERAS,
            **indices,
        }
        if self.text_embedding is not None:
            sample["t5_text_embeddings"] = self.text_embedding
            sample["t5_text_mask"] = np.ones(self.text_embedding.shape[0], dtype=np.int64)
        return sample


def execute_stub_policy(
    sample: dict,
    policy: Callable[[dict], np.ndarray],
    profile: RobotProfile,
    runtime_joint_names: Sequence[str],
    statistics: object | None = None,
) -> np.ndarray:
    """End-to-end contract probe: policy output to physical runtime commands."""
    actions = np.asarray(policy(sample), dtype=np.float32)
    if actions.shape != (CHUNK_SIZE, len(profile.active_joint_names)) or not np.isfinite(actions).all():
        raise ValueError("policy must return 16 finite active-joint targets")
    if statistics is not None:
        if statistics.robot_key != profile.robot_key or tuple(statistics.active_joint_names) != profile.active_joint_names:
            raise ValueError("statistics do not match robot key and active joint order")
        actions = statistics.denormalize_actions(actions)
    return profile.expand(actions, runtime_joint_names)
