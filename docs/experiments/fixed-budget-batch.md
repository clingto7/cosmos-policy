# Fixed-budget per-task training and evaluation

This is the workflow for new runs under [ADR 0006](../adr/0006-fixed-budget-two-gpu-batch.md).
Current jobs and relay scripts remain unchanged by owner instruction on 2026-11-03.
Switching those scripts to this workflow is a separate operational action.

## Fixed settings

| Setting | Contract |
| --- | --- |
| Training allocation | 2 × H100 80GB per task; two tasks per four-GPU node |
| Training budget | 25,000 optimizer updates; no task-specific extensions |
| Effective batch | 2 GPUs × 16 samples × 2 accumulation = 64 |
| Data | Task-specific audited 100 demonstrations, four RGB views, scene description prompt |
| Initialization | Common Cosmos-Predict2-2B-Video2World weights; independent run per task |
| Randomness | Explicit trainer seed 0 and sampler seed 0; preserve rank-derived seeding |
| Optimizer | Existing fused Adam, lr 1e-4, betas (0.9, 0.99), eps 1e-8, weight decay 0.1 |
| Schedule | Preserve existing LambdaLinearScheduler configuration; do not rescale its 30k boundary to 25k |
| Checkpoints | Full model, optimizer, scheduler and trainer state every 5,000 updates, at 5k, 10k, 15k, 20k and 25k |
| Selection | Always the validated 25k checkpoint; no best-of-checkpoints selection |
| Evaluation allocation | One 4090 per policy/simulator worker, unique port and output directory |
| Final | Four channels × 50 episodes = 200, namespaces 100000000 / 200000000 |
| Results | Per-task SR, LSCR and Eff. for all four channels and channel means; no overall score |

The training source currently used by the audited runs is `4369716`. The observed
Bench2Dex revision is `4f12207a8cdf28a83222baae2ce75471d425370a`. Admission must
record full source revisions, dirty-tree status, base-weight identity, environment,
manifest/statistics/prompt hashes, effective configuration and runner hash.
Changing implementation or protocol creates a new version; it must not silently
reuse an existing evaluation directory.

## One operational queue

Issue #10 is the batch control entry point. Use one shared task registry and one
dispatcher when this workflow is deployed. Other conversations may inspect status
or propose queue changes; they must not independently launch duplicate runs.
This document does not claim that dispatcher or registry is already implemented.

Each registry row must include task ID, run ID, protocol version, state, host,
GPU allocation, PID plus process start time, source/config/data hashes, target and
observed update, validated checkpoint path, resume lineage, final evaluation path,
accepted episode counts per channel, last heartbeat, and failure/retry reason.
Run admission must hold a task/run lock and a GPU-allocation lock, and check both
the registry and live processes. Reconcile after restart before admitting work.

States: queued, preflight, training, checkpoint-validation, final-ready,
final-running, complete; failures enter blocked with a reason. Final work is
tracked in #11 and may follow each frozen task checkpoint without waiting for
the entire training batch, provided its protocol is fixed.

## Per-task procedure

1. Verify the 100-episode manifest, robot active-joint mapping, four views,
   statistics, prompt, base weights, source and GPU availability. Resolve all
   storage symlinks: datasets and training outputs must be on private HDD2.
2. Freeze the effective configuration and start an independent two-GPU run.
   Record training throughput, memory and checkpoint timing. Do not infer a
   target from a directory name: some historical names contain `25k_to30000`.
3. Finish at 25k. Check all checkpoint components and successful save completion;
   a directory or `model/.metadata` alone is insufficient. Verify loading before
   declaring a checkpoint evaluation-ready. Keep checkpoint artifacts during
   this rollout; automated deletion requires a separately agreed retention rule.
4. Freeze checkpoint identity and protocol. Without a separate pilot, run 200
   final episodes directly following
   [the reference evaluation instructions](bench2dex-final-evaluation.md).
   Retry infrastructure failures using the same channel, episode index and seed;
   retain failed attempt logs. Valid policy failures count in SR. Repeated
   infrastructure failures block the task instead of producing a zero score.
5. Require exactly 50 unique accepted episodes per channel. Preserve results,
   seeds, videos, HDF5 and logs. Compute SR, LSCR and success-conditioned Eff.
   using the existing reference formulas. No successes means Eff. is undefined,
   not zero. Mark partial results explicitly; do not compute overall rankings.

Recovery uses the latest verified full checkpoint, unchanged GPU count and
configuration, and cumulative update count. Record every interruption. The current
trainer does not restore sampler position/RNG state as an exact uninterrupted
trajectory, so never claim bitwise-equivalent continuation.

## Transition inventory, 2026-11-03 22:11 CST

| Host | GPU pair 0,1 | GPU pair 2,3 |
| --- | --- | --- |
| cosmos03 | task21 about 22.0k / 25k | task12 about 21.6k / 40k |
| cosmos1029 | task26 about 4.5k / 30k | task27 about 4.5k / 30k |
| cosmos2nd | task42 about 1.6k / 30k | task43 about 1.6k / 30k |

The live relay processes still include task26→32, task27→34, task42→44,
task43→51 at 30k, and a cosmos03 relay that can launch another task12 after
task21. Another cosmos03 relay waits for a task12 30k run path although the
active task12 targets 40k. These are existing operational discrepancies, not
part of the new workflow. The owner explicitly requested leaving relays alone.

Single-4090 workers were evaluating task08 37.5k and task12 20k. The dual-4090
cosmos22 instance was running `gpu_keep_busy.py`, not an evaluation worker.
Existing 25k checkpoints are present for tasks 03, 06, 07, 08, 09 and 79;
this presence check is not a full resumability validation. Old checkpoint
eligibility and catch-up work are deferred.
