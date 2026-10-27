"""One HDF5-to-command seam for the first Bench2Dex task pilot."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import cv2

from cosmos_policy.datasets.bench2dex_audit import audit_task, load_sha256sum, sha256_file
from cosmos_policy.datasets.bench2dex_dataset import (
    CAMERAS,
    Bench2DexDataset,
    RobotProfile,
    execute_stub_policy,
    inspect_episode,
)
from cosmos_policy.datasets.bench2dex_training import load_bench2dex_training_dataset


PROFILE = RobotProfile(
    "test_robot",
    ("mimic", "joint_b", "joint_a"),
    ("joint_a", "joint_b"),
    (("mimic", "joint_b", -1.0, 0.25),),
)
DESCRIPTION = "Move two joints while keeping the mimic joint aligned."


def write_episode(path: Path, *, homing: int = 18) -> None:
    frames = 20
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as file:
        meta = file.create_group("meta")
        meta.create_dataset("robot_key", data="test_robot", dtype=strings)
        meta.create_dataset("instruction", data=DESCRIPTION, dtype=strings)
        meta.create_dataset("homing_start_sim_step", data=homing)
        meta.create_dataset("effective_fps", data=20.0)
        time = file.create_group("time")
        time.create_dataset("sim_step", data=np.arange(frames, dtype=np.int64))
        file.create_dataset("frame_valid", data=np.ones(frames, dtype=bool))

        robot = file.create_group("robot")
        robot.create_dataset("joint_names", data=np.asarray(["joint_b", "mimic", "joint_a"], dtype=strings))
        qpos = np.stack([20 + np.arange(frames), np.zeros(frames), 10 + np.arange(frames)], axis=1)
        robot.create_dataset("qpos", data=qpos.astype(np.float32))

        action = file.create_group("action")
        action.create_dataset("action_names", data=np.asarray(["joint_a", "joint_b", "mimic"], dtype=strings))
        commands = np.stack([np.arange(frames), 100 + np.arange(frames), np.zeros(frames)], axis=1)
        action.create_dataset("commanded", data=commands.astype(np.float32))
        action_valid = np.ones(frames, dtype=bool)
        action_valid[0] = False
        action_valid[-1] = False
        action.create_dataset("action_valid", data=action_valid)
        action.create_dataset("action_type", data="absolute", dtype=strings)
        action.create_dataset("control_mode", data="joint_position", dtype=strings)

        cameras = file.create_group("cameras")
        for number, camera in enumerate(CAMERAS):
            rgb = np.empty((frames, 4, 4, 3), dtype=np.uint8)
            for frame in range(frames):
                rgb[frame] = number * 40 + frame
            cameras.create_group(camera).create_dataset("rgb", data=rgb)


class Bench2DexContractTest(unittest.TestCase):
    def test_matched_audit_and_full_prompt_embedding_enter_training(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "source"
            source.mkdir()
            template = directory / "template.hdf5"
            write_episode(template)
            for index in range(100):
                (source / f"episode_{index:06d}.hdf5").hardlink_to(template)
            manifest, _ = audit_task(source, source, PROFILE, DESCRIPTION)
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            robot_map = directory / "robots.yml"
            robot_map.write_text(
                "robots:\n  test_robot:\n    full_joint_names: [mimic, joint_b, joint_a]\n"
                "    active_joint_names: [joint_a, joint_b]\n",
                encoding="utf-8",
            )
            embedding_path = directory / "embedding.npz"
            np.savez(embedding_path, prompt=DESCRIPTION, embedding=np.ones((512, 1024), dtype=np.float32))
            dataset = load_bench2dex_training_dataset(
                str(manifest_path), str(robot_map), str(embedding_path), expected_active_joints=2
            )
            sample = dataset[0]
            self.assertEqual(sample["camera_order"], CAMERAS)
            self.assertEqual(sample["t5_text_embeddings"].shape, (512, 1024))
            self.assertEqual(sample["value_function_sample_mask"], 0)
            np.savez(embedding_path, prompt="Different text", embedding=np.ones((512, 1024), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "audited full task description"):
                load_bench2dex_training_dataset(
                    str(manifest_path), str(robot_map), str(embedding_path), expected_active_joints=2
                )

    def test_runtime_mimic_rules_keep_training_active_joint_order(self) -> None:
        bare = RobotProfile("test_robot", PROFILE.full_joint_names, PROFILE.active_joint_names)
        runtime_info = SimpleNamespace(
            robot_key="test_robot",
            full_joint_names=["joint_b", "mimic", "joint_a"],
            active_joint_names=["joint_b", "joint_a"],
            mimic_rules=[(1, 0, -1.0, 0.25)],
        )
        resolved = bare.with_bench2dex_mimic_rules(runtime_info)
        self.assertEqual(resolved.active_joint_names, ("joint_a", "joint_b"))
        np.testing.assert_allclose(
            resolved.expand(np.asarray([[1, 101]], dtype=np.float32), ("joint_b", "mimic", "joint_a")),
            [[101, 0.25 - 101, 1]],
        )

    def test_episode_to_cosmos_sample_to_physical_runtime_commands(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode_000000.hdf5"
            write_episode(path)
            info = inspect_episode(path, PROFILE)
            self.assertEqual(info.usable_frames, 17)
            self.assertEqual(info.excluded_frames, 2)
            self.assertEqual(info.unusable_task_frames, 1)
            manifest, stats = audit_task(root, root, PROFILE, DESCRIPTION, expected_count=1)
            self.assertTrue(manifest["ready"])
            self.assertIsNotNone(stats)
            dataset = Bench2DexDataset([path], PROFILE, DESCRIPTION, image_size=4, statistics=stats)
            first = dataset[0]
            self.assertEqual(first["command"], DESCRIPTION)
            self.assertEqual(first["video"].shape, (3, 49, 4, 4))
            self.assertEqual(first["actions"].shape, (16, 2))
            self.assertEqual(first["future_frame"], 17)
            self.assertEqual(first["rollout_data_mask"], 0)
            self.assertEqual(first["world_model_sample_mask"], 0)
            self.assertEqual(first["video"][0, 5, 0, 0], 1)  # left wrist, current
            self.assertEqual(first["video"][0, 9, 0, 0], 41)  # right wrist, current
            self.assertEqual(first["video"][0, 13, 0, 0], 81)  # stereo left, current
            self.assertEqual(first["video"][0, 17, 0, 0], 121)  # stereo right, current
            self.assertEqual(first["video"][0, 29, 0, 0], 17)  # left wrist, future
            self.assertEqual(first["value_function_sample_mask"], 0)
            self.assertEqual(first["proprio"].shape, (2,))
            np.testing.assert_allclose(stats.denormalize_proprio(first["proprio"]), [11, 21])

            commands = execute_stub_policy(
                first,
                lambda sample: sample["actions"],
                PROFILE,
                ("mimic", "joint_a", "joint_b"),
                statistics=stats,
            )
            self.assertEqual(commands.shape, (16, 3))
            np.testing.assert_allclose(commands[0], [0.25 - 101, 1, 101], atol=1e-5)
            np.testing.assert_allclose(commands[-1], [0.25 - 116, 16, 116], atol=1e-5)

            final = dataset[-1]
            self.assertEqual(final["source_frame"], 17)
            self.assertEqual(final["future_frame"], 17)
            self.assertEqual(int(final["action_padding_mask"].sum()), 15)
            np.testing.assert_allclose(final["actions"], np.repeat(final["actions"][:1], 16, axis=0))

    def test_bad_episode_is_rejected_with_reason(self) -> None:
        cases = (
            ("action/action_type", "relative", "action/action_type"),
            ("meta/robot_key", "wrong_robot", "robot_key"),
            ("meta/homing_start_sim_step", 0, "homing_start_sim_step"),
        )
        with tempfile.TemporaryDirectory() as root:
            for index, (key, value, reason) in enumerate(cases):
                path = Path(root) / f"episode_{index:06d}.hdf5"
                write_episode(path)
                with h5py.File(path, "r+") as file:
                    file[key][()] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, reason):
                    inspect_episode(path, PROFILE)
            path = Path(root) / "episode_000003.hdf5"
            write_episode(path)
            with h5py.File(path, "r+") as file:
                file["robot/joint_names"][:] = ["joint_a", "joint_a", "mimic"]
            with self.assertRaisesRegex(ValueError, "unique"):
                inspect_episode(path, PROFILE)
            path = Path(root) / "episode_000004.hdf5"
            write_episode(path)
            with h5py.File(path, "r+") as file:
                del file["cameras/cam_stereo_right/rgb"]
            with self.assertRaisesRegex(ValueError, "cam_stereo_right"):
                inspect_episode(path, PROFILE)
            path = Path(root) / "episode_000005.hdf5"
            write_episode(path)
            with h5py.File(path, "r+") as file:
                file["meta"].create_dataset("action_unit", data="degrees", dtype=h5py.string_dtype())
            with self.assertRaisesRegex(ValueError, "radians"):
                inspect_episode(path, PROFILE)

    def test_invalid_frame_splits_action_runs_without_crossing_gap(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode_000000.hdf5"
            write_episode(path)
            with h5py.File(path, "r+") as file:
                file["frame_valid"][10] = False
            info = inspect_episode(path, PROFILE)
            self.assertEqual(info.action_runs, ((1, 10), (11, 18)))
            dataset = Bench2DexDataset([path], PROFILE, DESCRIPTION, image_size=4)
            last_before_gap = dataset[8]
            self.assertEqual(last_before_gap["source_frame"], 9)
            self.assertEqual(last_before_gap["future_frame"], 9)
            self.assertEqual(int(last_before_gap["action_padding_mask"].sum()), 15)

    def test_jpeg_encoded_cameras_decode_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode_000000.hdf5"
            write_episode(path)
            with h5py.File(path, "r+") as file:
                file["meta"].create_dataset("rgb_encoding", data="jpeg", dtype=h5py.string_dtype())
                for camera in CAMERAS:
                    group = file[f"cameras/{camera}"]
                    raw = group["rgb"][:]
                    del group["rgb"]
                    jpeg = group.create_dataset("rgb", (len(raw),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                    for index, frame in enumerate(raw):
                        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        self.assertTrue(ok)
                        jpeg[index] = encoded
            sample = Bench2DexDataset([path], PROFILE, DESCRIPTION, image_size=4)[0]
            self.assertEqual(sample["video"].shape, (3, 49, 4, 4))
            self.assertLess(abs(int(sample["video"][0, 5, 0, 0]) - 1), 3)

            with h5py.File(path, "r+") as file:
                file["meta/rgb_encoding"][()] = "raw"
            with self.assertRaisesRegex(ValueError, "rgb_encoding"):
                inspect_episode(path, PROFILE)

    def test_scene_prompt_overrides_embedded_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "episode_000000.hdf5"
            write_episode(path)
            with h5py.File(path, "r+") as file:
                file["meta/instruction"][()] = "A different task."
            manifest, statistics = audit_task(root, root, PROFILE, DESCRIPTION, expected_count=1)
            self.assertTrue(manifest["ready"])
            self.assertIsNotNone(statistics)
            self.assertEqual(manifest["entries"][0]["usable_frames"], 17)
            self.assertEqual(manifest["entries"][0]["embedded_instruction"], "A different task.")
            self.assertFalse(manifest["entries"][0]["prompt_matches_embedded"])
            self.assertEqual(Bench2DexDataset([path], PROFILE, DESCRIPTION, image_size=4)[0]["command"], DESCRIPTION)

    def test_100_file_audit_fails_on_source_drift(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "source"
            reference = Path(root) / "reference"
            source.mkdir()
            reference.mkdir()
            template = Path(root) / "template.hdf5"
            write_episode(template)
            for index in range(100):
                name = f"episode_{index:06d}.hdf5"
                (source / name).hardlink_to(template)
                (reference / name).hardlink_to(template)
            manifest, stats = audit_task(source, reference, PROFILE, DESCRIPTION)
            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["source_count"], 100)
            self.assertEqual(manifest["matched_count"], 100)
            self.assertEqual(len(manifest["entries"]), 100)
            self.assertEqual(stats.active_joint_names, ("joint_a", "joint_b"))
            np.testing.assert_allclose(stats.denormalize_actions(stats.normalize_actions([1, 101])), [1, 101])

            candidate, candidate_stats = audit_task(source, None, PROFILE, DESCRIPTION)
            self.assertTrue(candidate["source_valid"])
            self.assertFalse(candidate["ready"])
            self.assertEqual(candidate["matched_count"], 0)
            self.assertEqual(candidate["entries"][0]["identity"], "not_checked")
            self.assertIsNotNone(candidate_stats)

            checksum_file = Path(root) / "yk.sha256"
            checksum_file.write_text(
                "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(reference.glob("episode_*.hdf5")))
            )
            remote, remote_stats = audit_task(
                source,
                None,
                PROFILE,
                DESCRIPTION,
                yk_hashes=load_sha256sum(checksum_file),
                yk_location="host:/yk/task03",
            )
            self.assertTrue(remote["ready"])
            self.assertEqual(remote["yk_reference_kind"], "remote_sha256sum")
            self.assertEqual(remote["entries"][0]["yk_source"], "host:/yk/task03/episode_000000.hdf5")
            self.assertIsNotNone(remote_stats)

            drifted = reference / "episode_000099.hdf5"
            drifted.unlink()
            write_episode(drifted, homing=17)
            manifest, stats = audit_task(source, reference, PROFILE, DESCRIPTION)
            self.assertFalse(manifest["ready"])
            self.assertEqual(manifest["matched_count"], 99)
            self.assertTrue(manifest["source_valid"])
            self.assertIsNotNone(stats)


if __name__ == "__main__":
    unittest.main()
