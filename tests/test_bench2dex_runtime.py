"""Runtime adapter contract for the task 03 closed-loop evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from cosmos_policy.datasets.bench2dex_audit import TaskStatistics
from cosmos_policy.datasets.bench2dex_dataset import CAMERAS, CHUNK_SIZE, RobotProfile
from cosmos_policy.experiments.robot.bench2dex.runtime import (
    Bench2DexPolicyAdapter,
    profile_with_urdf_mimics,
)

ACTIVE = tuple(f"joint_{index}" for index in range(26))
MIMIC = tuple(f"mimic_{index}" for index in range(8))
FULL = ACTIVE + MIMIC
PROMPT = "Place three wine glasses upright on the plate."


def make_profile() -> RobotProfile:
    return RobotProfile(
        "task03_robot",
        FULL,
        ACTIVE,
        tuple((name, ACTIVE[index], -1.0, 0.1) for index, name in enumerate(MIMIC)),
    )


def make_statistics() -> TaskStatistics:
    return TaskStatistics(
        "task03_robot",
        ACTIVE,
        np.full(26, -2.0, dtype=np.float32),
        np.full(26, 2.0, dtype=np.float32),
        np.full(26, -2.0, dtype=np.float32),
        np.full(26, 2.0, dtype=np.float32),
    )


def make_observation() -> dict:
    return {
        "observation": {name: {"rgb": np.full((8, 8, 3), i, dtype=np.uint8)} for i, name in enumerate(CAMERAS)},
        "joint_action": {"vector": np.zeros(26, dtype=np.float32)},
        "language": PROMPT,
    }


class Bench2DexRuntimeTest(unittest.TestCase):
    def test_registered_inactive_joint_is_zero_but_unknown_joint_fails(self) -> None:
        active = np.asarray([[0.25], [0.5]], dtype=np.float32)
        profile = RobotProfile("robot", ("active", "inactive"), ("active",), inactive_joint_names=("inactive",))
        np.testing.assert_array_equal(profile.expand(active, ("inactive", "active"))[:, 0], [0, 0])
        with self.assertRaisesRegex(ValueError, "unmapped runtime joints"):
            RobotProfile("robot", ("active", "unknown"), ("active",)).expand(
                active, ("active", "unknown")
            )

    def test_inactive_joint_can_source_a_urdf_mimic(self) -> None:
        profile = RobotProfile(
            "robot", ("active", "inactive", "mimic"), ("active",), inactive_joint_names=("inactive",)
        )
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "robot.urdf"
            urdf.write_text("<robot><joint name='mimic'><mimic joint='inactive' multiplier='2'/></joint></robot>")
            resolved = profile_with_urdf_mimics(profile, urdf, 1)
        np.testing.assert_array_equal(
            resolved.expand(np.asarray([[1]], dtype=np.float32), resolved.full_joint_names), [[1, 0, 0]]
        )

    def test_inference_rgb_matches_training_frame_from_same_jpeg(self) -> None:
        from cosmos_policy.datasets.bench2dex_dataset import _image
        from cosmos_policy.experiments.robot.bench2dex.runtime import prepare_bench2dex_model_observation

        y, x = np.indices((480, 640))
        rgb = np.stack(((x * 3) % 256, (y * 5) % 256, (x + y) % 256), axis=-1).astype(np.uint8)
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        observation = {key: rgb.copy() for key in (
            "left_wrist_image", "right_wrist_image", "primary_image", "secondary_image"
        )}
        observation["proprio"] = np.zeros(26, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.hdf5"
            with h5py.File(path, "w") as file:
                camera = file.create_group("cameras/cam_stereo_left")
                data = camera.create_dataset("rgb", (1,), dtype=h5py.vlen_dtype(np.uint8))
                data[0] = np.asarray(jpeg, dtype=np.uint8)
            with h5py.File(path) as file:
                training_frame = _image(file, "cam_stereo_left", 0, 224)

        # The simulator provides decoded RGB from its camera, so compare against
        # the decoded JPEG used by training, before either path resizes it.
        decoded = cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for key in observation:
            if key != "proprio":
                observation[key] = decoded.copy()
        prepared = prepare_bench2dex_model_observation(observation, 224)
        for key in ("left_wrist_image", "right_wrist_image", "primary_image", "secondary_image"):
            np.testing.assert_array_equal(prepared[key], training_frame)
        np.testing.assert_array_equal(prepared["proprio"], observation["proprio"])

    def test_adapter_maps_four_named_views_and_expands_absolute_targets(self) -> None:
        seen: dict = {}

        def predict(observation: dict, instruction: str, seed: int) -> np.ndarray:
            seen["observation"] = observation
            seen["instruction"] = instruction
            seen["seed"] = seed
            return np.full((CHUNK_SIZE, 26), 0.5, dtype=np.float32)

        adapter = Bench2DexPolicyAdapter(
            predict,
            make_profile(),
            make_statistics(),
            PROMPT,
            runtime_joint_names=FULL[::-1],
            seed=7,
        )
        commands = adapter.get_action_chunk(make_observation())

        self.assertEqual(commands.shape, (16, 34))
        self.assertEqual(seen["instruction"], PROMPT)
        self.assertEqual(seen["seed"], 7)
        self.assertEqual(
            [
                int(seen["observation"][key][0, 0, 0])
                for key in ("left_wrist_image", "right_wrist_image", "primary_image", "secondary_image")
            ],
            [0, 1, 2, 3],
        )
        np.testing.assert_allclose(commands[:, FULL[::-1].index("joint_0")], 1.0)
        np.testing.assert_allclose(commands[:, FULL[::-1].index("mimic_0")], -0.9)

    def test_urdf_mimics_are_resolved_in_dependency_order(self) -> None:
        profile = RobotProfile("test", ("active", "mimic_a", "mimic_b"), ("active",))
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "robot.urdf"
            urdf.write_text(
                '<robot name="test">'
                '<joint name="mimic_b"><mimic joint="mimic_a" multiplier="2" offset="0.5"/></joint>'
                '<joint name="mimic_a"><mimic joint="active" multiplier="-1" offset="0.25"/></joint>'
                '</robot>',
                encoding="utf-8",
            )
            resolved = profile_with_urdf_mimics(profile, urdf, 2)
        commands = resolved.expand(np.asarray([[1.0]], dtype=np.float32), resolved.full_joint_names)
        np.testing.assert_allclose(commands, [[1.0, -0.75, -1.0]])

    def test_adapter_rejects_wrong_prompt_or_malformed_views(self) -> None:
        adapter = Bench2DexPolicyAdapter(
            lambda observation, instruction, seed: np.zeros((16, 26), dtype=np.float32),
            make_profile(),
            make_statistics(),
            PROMPT,
            runtime_joint_names=FULL,
        )
        bad_prompt = make_observation()
        bad_prompt["language"] = "different prompt"
        with self.assertRaisesRegex(ValueError, "task description"):
            adapter.get_action_chunk(bad_prompt)

        missing_camera = make_observation()
        del missing_camera["observation"][CAMERAS[0]]
        with self.assertRaisesRegex(ValueError, "cam_wrist_left"):
            adapter.get_action_chunk(missing_camera)

    def test_runtime_protocol_dispatches_reset_update_action_and_close(self) -> None:
        seeds = []
        adapter = Bench2DexPolicyAdapter(
            lambda observation, instruction, seed: seeds.append(seed) or np.zeros((16, 26), dtype=np.float32),
            make_profile(),
            make_statistics(),
            PROMPT,
            runtime_joint_names=FULL,
        )
        from cosmos_policy.experiments.robot.bench2dex.runtime import handle_rpc_command

        self.assertEqual(handle_rpc_command(adapter, {"command": "reset", "seed": 19}), {"result": True})
        self.assertEqual(
            handle_rpc_command(adapter, {"command": "update_after_action", "observation": make_observation()}),
            {"result": True},
        )
        action = handle_rpc_command(
            adapter,
            {"command": "get_action_chunk", "observation": make_observation()},
        )["result"]
        self.assertEqual(action.shape, (16, 34))
        self.assertEqual(seeds, [19])
        self.assertEqual(handle_rpc_command(adapter, {"command": "close"}), {"result": True, "close": True})


if __name__ == "__main__":
    unittest.main()
