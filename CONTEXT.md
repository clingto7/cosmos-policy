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
