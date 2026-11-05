# Scale embodiment batches by task count and bind schedules to the run budget

On 2026-11-05, the owner replaced ADR 0007's uniform two-GPU, 50k recipe with
32k optimizer updates and task-count-dependent batches. Shared normalization,
standard descriptions, task-uniform sampling, one checkpoint per embodiment,
and restarting the policy server between tasks remain unchanged. Historical
per-task 25k runs are not changed or retrained by this decision.

| Tasks in embodiment | GPUs | Microbatch per GPU | Accumulation | Global batch | Total sampled windows | Expected windows per task |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 16 | 2 | 64 | 2,048,000 | 2,048,000 |
| 2 | 4 | 16 | 2 | 128 | 4,096,000 | 2,048,000 |
| 3 | 8 | 12 | 2 | 192 | 6,144,000 | 2,048,000 |

A step means an optimizer update, not a microbatch. The dataset stream length is
`global_batch * max_steps`; resume starts at `completed_updates * global_batch`.
Actual per-task counts fluctuate under uniform random sampling and are recorded.
The task count comes from the complete audited embodiment bundle, never a
hand-selected subset of tasks.

## Learning rate and recording

The owner requires the LR schedule to follow `max_steps`, rather than an inherited
fixed horizon, and approves keeping the same peak LR across all three batches.
The implementation retains peak LR `1e-4`, preserves the original warmup fraction
`1000 / 30000` (1,067 updates at 32k), and uses one smooth cosine decay to `6e-6`
at the configured endpoint. That minimum is the old recipe's eventual constant
LR. These warmup/minimum choices are implementation defaults, recorded explicitly;
they are not claims that the best LR has been experimentally established.

Audit found the previous experiment inherited `LambdaLinearScheduler` with a
30k first interval and a subsequent constant factor of 0.06. It was neither a
50k schedule nor automatically bound to the training stop. New runs explicitly
select their scheduler and do not alter the old per-task configuration.

Freeze the recipe, input/code hashes and environment in `run-identity.json`.
Save the resolved training configuration and optimizer settings. Record global
mean loss across all ranks and accumulated microbatches, LR used for each update,
next LR, completed optimizer updates, total windows and actual per-task counts.
A resumed run starts a new log session at the checkpoint's committed update;
records beyond that checkpoint remain historical attempts, not extra consumed
samples in the resumed trajectory. LR and optimizer state resume together.

Save full state every 5k updates and at the final 32k update. Keep the newest two
validated checkpoints and preserve the final one. The new final evaluation gate
requires 32k. Smoke runs use the same GPU/microbatch recipe but four updates and
saves every two updates, with their own four-step LR schedule and sample budget.
They are separate runs and cannot be resumed as production runs.

## Evaluation hardware

Future evaluation commands explicitly pass `--device cpu` to Bench2Dex
`run_policy.py` for CPU physics. The remote policy remains on CUDA and RGB
rendering still requires an RTX GPU. Record `physics_device: cpu` in the evaluation
protocol. Do not infer CPU physics from headless mode or a platform default.

The requested CPU setting is an experimental protocol choice; greater stability
has not been established by an A/B experiment here. Existing scores produced with
a different physics device retain their original provenance.

## Scope

These budgets equalize expected per-task exposure; they do not establish identical
optimization behavior or convergence across task mixtures. Final evaluation
remains task-level, four channels and 50 episodes per channel. Extensions beyond
the agreed budget remain a separate decision.
