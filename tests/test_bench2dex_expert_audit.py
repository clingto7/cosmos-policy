import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("expert_audit", Path(__file__).parents[1] / "bin/bench2dex_expert_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ExpertAuditTest(unittest.TestCase):
    def test_windows_never_cross_invalid_gaps_or_pad(self):
        starts = audit.full_windows([(1, 17), (20, 36), (40, 55)], count=20)
        self.assertEqual(starts, [1, 20])

    def test_repeated_note_requires_release(self):
        names = [f"joint_{i}_to_1" for i in (18, 23, 24, 33, 38, 39)]
        positions = np.zeros((5, 6))
        positions[:, 0] = [0.005, 0.005, 0.001, 0.005, 0.005]
        result = audit.melody_progress(positions, names, np.ones(5, bool))
        self.assertEqual(result["left"]["event_frames"], [0, 3])
        self.assertEqual(result["left"]["notes"], 2)
        self.assertEqual(result["right"]["notes"], 0)

    def test_same_trajectory_not_selected_twice_across_metadata_strata(self):
        rows = [{"command_sha256": h, "prompt_matches": p} for h, p in [("a", False), ("a", True), ("b", True)]]
        result = audit.choose_sources(rows, 10)
        self.assertEqual(len(result), 2)

    def test_metrics_preserve_horizon_and_joint_axes(self):
        target = np.zeros((2, 16, 4))
        predicted = target.copy()
        predicted[:, -1, 2] = 1
        result = audit.metrics(predicted, target, np.ones_like(target),
                               ["xarm_l_joint1", "multi_xarm_r_joint1", "index_q1", "multi_index_q1"])
        self.assertEqual(result["per_horizon_mae_rad"][-1], 0.25)
        self.assertEqual(result["per_horizon_mae_rad"][0], 0)
        self.assertEqual(result["groups"]["left_hand"]["mae_rad"], 1 / 16)
        self.assertEqual(result["hold_mae_rad"], 1)


if __name__ == "__main__":
    unittest.main()
