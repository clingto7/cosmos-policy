# Bench2Dex task 03 closed-loop evaluation

`runtime.py` serves the audited task 03 Cosmos checkpoint through Bench2Dex's
`REMOTE` length-prefixed RPC protocol. The adapter validates the four named RGB
views, 26 active-joint positions, and exact audited task description. It
normalizes proprioception, requests a 16-step action chunk, denormalizes the
absolute targets, expands URDF mimic joints, and returns all 34 joints in the
robot map's canonical order. Bench2Dex's `GR00T_XE` remote reindex path maps
that order into the live Isaac articulation order before execution.

## Start the policy server

Run this in the CUDA environment that has Cosmos Policy installed. The manifest,
robot map, task embedding, and checkpoint must be available on that host.

```sh
python -m cosmos_policy.experiments.robot.bench2dex.runtime \
  --checkpoint /path/to/task03/checkpoint \
  --manifest /path/to/task03-manifest.json \
  --robot-map /path/to/Bench2Dex/robots/active_dof_maps.yml \
  --embedding /path/to/task03-prompt.npz \
  --checkpoint-id task03-step-4 \
  --host 127.0.0.1 --port 9000 --seed 7
```

When the URDF tree is outside the Bench2Dex checkout, set
`DEX2BENCH_DATASET_ROOT` to the dataset root containing `Robots_p/`.

## Run one simulator episode

From the Bench2Dex repository, use the robot key recorded in the task03
manifest. `--policy-name GR00T_XE` enables the runner's canonical-to-runtime
joint reindexing for remote policies. Set `--policy-display-name` to the same
stable value passed as `--checkpoint-id`; `per_episode.jsonl` records that
checkpoint label, the seed, the episode outcome, and any policy error.

```sh
python run_policy.py \
  --task scenes/03_wine_glass_plate_balance.yaml \
  --policy-type REMOTE --policy-name GR00T_XE \
  --remote-host 127.0.0.1 --remote-port 9000 \
  --robot-key multi_xarm7_with_ability --active-dof --enable-rgb \
  --policy-display-name task03-step-4 \
  --num-episodes 1 --episode-steps 400 --seed 7 \
  --output-dir outputs/cosmos-task03/checkpoint-id
```

Use the actual audited `robot_key`, server address, and checkpoint identity in
the command. An episode's actions are queried at 20 Hz; each returned 16-step
chunk is consumed before the next policy request. Set `--record-all
--record-dir <path>` when the HDF5 rollout itself is also needed.
