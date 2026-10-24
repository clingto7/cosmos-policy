# Cosmos Policy on Bench2Dex

This context defines the robot data and evaluation language used to adapt Cosmos Policy to Bench2Dex.

## Language

**Robot key**:
The Bench2Dex identifier for a robot embodiment, which determines its joint set.

**Active joint**:
A robot joint independently targeted by the policy, excluding mimic joints.

**Action chunk**:
A sequence of future active-joint targets produced from one observation.

**Evaluation channel**:
One of the Bench2Dex simulation conditions `none`, `cov`, `inv`, or `inv_cov`.

**Pilot evaluation**:
Closed-loop Bench2Dex episodes on fixed seeds used to compare training checkpoints.

**Final evaluation**:
Closed-loop Bench2Dex episodes on the yk comparison seeds, run after selecting a checkpoint.

## Task 03 data contract

- Training uses all 100 RGB HDF5 episodes in `03_wine_glass_plate_balance/replay-generalization`. Their file names and SHA256 hashes matched yk's `teleopdata` directory on 2026-10-22; the public Bench2Dex/teleopdata revision was `dbfb05dfc5b636e5c83925bc28c3041e3d5492a8`.
- The prompt is the full `scene.description` from `scenes/03_wine_glass_plate_balance.yaml`, as in yk's `policy/GR00T_n15/train.sh`. The embedded HDF5 instructions differ in all 100 files and are recorded for provenance, not used as the training prompt.
- Four RGB views are stored as per-frame JPEG bytes. The policy targets 26 active joints at 20 Hz; eight mimic joints are expanded for the 34-joint runtime command. Homing frames and invalid actions are excluded from training statistics.

## Task 06 second-robot candidate (preliminary)

- The public source is pinned to `Bench2Dex/teleopdata` revision `dbfb05dfc5b636e5c83925bc28c3041e3d5492a8`. Its `06_fruit_bowl_loading/replay-generalization` directory contains 100 HDF5 files (16,906,017,800 bytes). All 100 SHA256 values match the instance's `teleopdata/dataset/06_fruit_bowl_loading/replay-generalization` files and the pinned Hugging Face LFS hashes. The yk `policy/GR00T_n15/train.sh` resolves task data from `../teleopdata/dataset/${TASK_NUM}_*`.
- The sampled HDF5 identifies `multi_ur5_rh56dfx_with_flange`; the Bench2Dex active-DOF registry lists 24 active and 36 full joints (12 mimic). A raw-byte string scan found that robot key, the four RGB camera names, absolute joint-position mode, and all 24 registry active-joint names in each of the 100 files. This scan is preliminary and does not validate HDF5 dataset shapes or frame-level validity.
- A private HDD2 copy is staged at `/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task06/dataset/replay-generalization`, with `hf-source-manifest.json` and `SHA256SUMS`. Keep the SSD `teleopdata` tree as the source and use the HDD2 copy for this pilot.
- Task 06 remains a provisional candidate until the structured HDF5 audit checks named `qpos`/action arrays, camera dataset shapes, homing and invalid-frame boundaries, and computes its task-specific statistics. The current `cosmos-copy4090` system Python has no `h5py`; do not install packages there as part of the smoke inspection.
