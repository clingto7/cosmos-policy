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
