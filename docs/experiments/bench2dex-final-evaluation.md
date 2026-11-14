# Bench2Dex final evaluation for a selected Cosmos checkpoint

The runner explicitly selects CPU physics (`--device cpu`); RGB rendering and policy inference still use GPU. New per-embodiment runs use the fixed 32k checkpoint under [ADR 0008](../adr/0008-task-count-training-budget.md).

Select a checkpoint using pilot seeds first. The final runner evaluates that fixed checkpoint on `none`, `cov_only`, `inv_only`, and `inv_cov`, with 50 episodes in each channel. The first three channels use the same task `replay-generalization` anchor directory; `inv_cov` uses independent samples. It uses Bench2Dex seed namespaces `100000000` and `200000000` (Full) and computes the episode limit from the scene's expert time at 1.5×. For example, task03 has 2164 expert physics steps and a 1082 policy-step limit; task06 has 1280 and a 640 policy-step limit. The earlier 1200-step pilots are not final comparisons.

For checkpoint comparison or runner validation, pass `--seed-namespace pilot`. This uses `300000000` for the anchor channels and `400000000` for Full. The task03 20k–50k training-budget curve uses 50 episodes per channel for every checkpoint. A shorter smoke check may use `--episodes-per-channel 10`, but its wider uncertainty makes it unsuitable for this curve. Put pilot and final runs in separate output directories. Final runs always require 50 episodes per channel, and the aggregator rejects pilot results as official final scores.

Run [bench2dex_final_eval.py](../../bin/bench2dex_final_eval.py) on one GPU instance with both Python environments, the Bench2Dex checkout and the selected checkpoint accessible. Set these paths to the verified files for the chosen task and checkpoint. Example for task03:

```sh
python bin/bench2dex_final_eval.py \
  --task-id 03 --checkpoint-id task03-selected \
  --manifest "$MANIFEST" --embedding "$EMBEDDING" \
  --base-model "$BASE_MODEL" --tokenizer "$TOKENIZER" \
  --checkpoint "$CHECKPOINT" --scene "$SCENE" \
  --anchor-dir "$ANCHOR_DIR" --robot-map "$ROBOT_MAP" \
  --policy-repo "$POLICY_REPO" --bench-repo "$BENCH_REPO" \
  --policy-python "$POLICY_PYTHON" --sim-python "$SIM_PYTHON" \
  --output /inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/task03/final/task03-selected
```

Every simulator attempt has its own directory and log. Restart the same command after interruption; it reads accepted episode indices, retries missing indices with the same seeds, and refuses a changed protocol. Valid unsuccessful episodes count against SR. Infrastructure errors and nonzero simulator exits remain in the attempt logs. The runner writes a channel summary, 50 selected per-episode rows, 50 rollout videos, and a four-channel task summary. The original HDF5 recordings remain under attempt directories.

## Progress, throughput and ETA

Report accepted episodes against the per-channel denominator, valid successes and
failures, infrastructure errors, current attempt, episode index and physics step.
Read accepted counts from `per_episode.jsonl` and `summary.json`; inspect the live
attempt's `simulation.log` to confirm its episode and step are advancing. An active
PID alone does not prove progress.

Estimate throughput from elapsed time per accepted episode in recent successful
attempts, excluding infrastructure retries. Estimate remaining time from each
channel's remaining episodes and observed rate; report a range when attempts vary.
GPU utilization is an instantaneous sample: low utilization alone does not mean
the evaluation is stalled while episode steps and policy logs keep advancing.
Record host generation and observation time with every estimate.

After all 26 selected checkpoints have final outputs, place or link each task directory under one evaluation root and run:

```sh
python bin/bench2dex_aggregate.py --eval-root /path/to/final-evaluations \
  --output /path/to/leaderboard-metrics --require-all-26
```

The aggregator emits `per_task.csv` and `overall.json`. SR and LSCR are equal-weight means over channels and tasks. Efficiency is computed from successful policy episodes only, then averaged over tasks with success. The aggregator reports partial task counts when `--require-all-26` is omitted; do not present partial results as a complete leaderboard ranking.
