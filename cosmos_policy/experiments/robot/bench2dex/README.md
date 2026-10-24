# Bench2Dex task-specific training and closed-loop evaluation

The audit manifest is the task configuration source. It records the two-digit
task ID, robot key, named active joints, four policy cameras, 16-step horizon,
20 Hz control rate, matched demonstrations, and per-task statistics. Training
and inference reject a manifest whose robot map, task ID, camera order, or
active-joint dimensions disagree. Each task gets a task-specific model action
dimension, run name, normalization statistics, and checkpoint. Task 03 keeps
its existing 26-joint defaults and environment variable names.

For a new task, point the experiment loader at its ready audit before loading
the config. The action dimension and task ID are read from the manifest and
validated against the robot map:

```sh
export BENCH2DEX_MANIFEST=/path/to/task-manifest.json
export BENCH2DEX_ROBOT_MAP=/path/to/Bench2Dex/robots/active_dof_maps.yml
export BENCH2DEX_EMBEDDING=/path/to/task-prompt.npz
```

Launch the registered experiment whose suffix matches the audited task ID,
such as `cosmos_predict2_2b_480p_bench2dex_task06`. Keep its output root and
checkpoint directory separate from every other task and from Task 03.

`runtime.py` serves an audited checkpoint through Bench2Dex's `REMOTE`
length-prefixed RPC protocol. The adapter validates four named RGB views,
active-joint state, and the exact audited task description; it normalizes
proprioception, requests 16 absolute active-joint targets, denormalizes them,
expands URDF mimic joints, and returns the robot map's canonical runtime order.
Bench2Dex's `GR00T_XE` remote reindex path maps that order into the live Isaac
articulation order before execution.

## Start the policy server

Run this in the CUDA environment that has Cosmos Policy installed. The manifest,
robot map, task embedding, and checkpoint must be available on that host.

```sh
python -m cosmos_policy.experiments.robot.bench2dex.runtime \
  --checkpoint /path/to/task-checkpoint \
  --manifest /path/to/task-manifest.json \
  --robot-map /path/to/Bench2Dex/robots/active_dof_maps.yml \
  --embedding /path/to/task-prompt.npz \
  --checkpoint-id task06-step-4 \
  --host 127.0.0.1 --port 9000 --seed 7
```

When the URDF tree is outside the Bench2Dex checkout, set
`DEX2BENCH_DATASET_ROOT` to the dataset root containing `Robots_p/`.

## Run one simulator episode

From the Bench2Dex repository, set `ROBOT_KEY` to the value in the task
manifest. `--policy-name GR00T_XE` enables the runner's canonical-to-runtime
joint reindexing for remote policies. Set `--policy-display-name` to the same
stable value passed as `--checkpoint-id`; `per_episode.jsonl` records that
checkpoint label, the seed, the episode outcome, and any policy error.

```sh
python run_policy.py \
  --task scenes/06_fruit_bowl_loading.yaml \
  --policy-type REMOTE --policy-name GR00T_XE \
  --remote-host 127.0.0.1 --remote-port 9000 \
  --robot-key "$ROBOT_KEY" --active-dof --enable-rgb \
  --policy-display-name task06-step-4 \
  --num-episodes 1 --episode-steps 400 --seed 7 \
  --output-dir outputs/cosmos-task06/checkpoint-id
```

Use the actual audited `robot_key`, server address, and checkpoint identity in
the command. An episode's actions are queried at 20 Hz; each returned 16-step
chunk is consumed before the next policy request. Set `--record-all
--record-dir <path>` when the HDF5 rollout itself is also needed.
