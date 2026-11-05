# Per-embodiment training and task-level evaluation

This implements [ADR 0007](../adr/0007-per-embodiment-training-budget.md) and the revised budgets in [ADR 0008](../adr/0008-task-count-training-budget.md).
The CPU data/config/checkpoint contracts can be exercised locally. Real model
forward/backward and FSDP restore require the CUDA training environment. RGB
Isaac Sim evaluation requires a supported RTX GPU, separately from the H100
training allocation. The validation record below states what has been exercised.

## Prepare one bundle per embodiment

Keep existing per-task audits and T5 embeddings. Create a JSON catalog whose
paths are relative to the catalog file, or absolute:

```json
[
  {"manifest": "task03/manifest.json", "embedding": "task03/t5-embedding.npz"},
  {"manifest": "task79/manifest.json", "embedding": "task79/t5-embedding.npz"}
]
```

Supply every task of each included embodiment; a catalog may contain one full
embodiment group or all 26 tasks. Group membership is checked against Bench2Dex's
existing task statistics catalog, using its `task` and `robot_key` fields only.
No GR00T normalization values are reused. The builder verifies the audited
100 demonstrations per task by SHA256, inspects valid windows, recomputes task
statistics, and merges min/max bounds across the group's tasks. It rejects an
incomplete group, mixed joint orders, duplicate sources, changed data, mismatched
prompts, or incompatible embedding shapes. Standard prompts should be encoded
with the existing T5 embedding command before preparation.

```sh
python -m cosmos_policy.datasets.bench2dex_embodiment \
  --catalog "$CATALOG" \
  --robot-map "$BENCH_REPO/robots/active_dof_maps.yml" \
  --grouping "$BENCH_REPO/policy/GR00T_XE/xe_norm_stats.taskstats.json" \
  --output "$PRIVATE_HDD2/embodiment-bundles-v1"
```

The output is `<robot_key>.json` for each group. Existing files are never
overwritten; use a new versioned directory when inputs change. Bundles embed
the task audits and shared statistics and retain hashes of source audits,
embeddings and the grouping reference. Episode and embedding paths must remain
accessible to the training host. Evaluation needs the bundle, selected task's
original audit and embedding, robot map and URDF assets, but does not read the
training HDF5 files.

## Plan, prove, then train

Use the existing CUDA Cosmos environment for execution. For Chuangzhi,
`PRIVATE_HDD2` must resolve below
`/inspire/hdd2/project/roboticsystem2/ky26063`; code stays on HDD. Resolve
symlinks before choosing data or output locations.

The launcher defaults to a read-only plan. `BASE_MODEL` is the original
Video2World consolidated `.pt` file; `TOKENIZER` is its tokenizer checkpoint.

```sh
python -m cosmos_policy.experiments.robot.bench2dex.train_embodiment \
  --manifest "$BUNDLE" \
  --robot-map "$BENCH_REPO/robots/active_dof_maps.yml" \
  --base-model "$BASE_MODEL" --tokenizer "$TOKENIZER" \
  --output "$PRIVATE_HDD2/embodiment-training" \
  --run-name "$RUN_NAME"
```

On the allocated GPU host, expose exactly the required H100 GPUs with
`CUDA_VISIBLE_DEVICES`. Add `--smoke --execute` and use a separate run name for
the four-update GPU proof (saves at updates 2 and 4). This checks the production
microbatch/accumulation shape without launching 32k updates. Inspect finite
loss/gradients, actual base-weight initialization and save/load before admitting
the full run. For full training, omit `--smoke` and add `--execute`.

After the smoke run completes, rerun its identical command with
`--verify-restore --smoke --execute`. This forces a configured multi-GPU model/optimizer/
scheduler/trainer reload at update 4 and exits without further optimizer updates.
Without `--verify-restore`, an already-completed run exits before loading the model.

The bundle selects one of three explicit launch scripts:

| Tasks | Script | GPUs × microbatch × accumulation | Global batch | Steps | Total samples |
| --- | --- | --- | --- | --- | --- |
| 1 | [train_1_task.sh](../../bin/bench2dex/train_1_task.sh) | 2 × 16 × 2 | 64 | 32,000 | 2,048,000 |
| 2 | [train_2_task.sh](../../bin/bench2dex/train_2_task.sh) | 4 × 16 × 2 | 128 | 32,000 | 4,096,000 |
| 3 | [train_3_task.sh](../../bin/bench2dex/train_3_task.sh) | 8 × 12 × 2 | 192 | 32,000 | 6,144,000 |

All three retain peak LR `1e-4`; no automatic batch-based LR scaling is applied.
The complete LIBERO 40k schedule is scaled to the run budget: warmup is
`round(max_steps / 40)`, and the first cycle is `round(3 * max_steps / 4)`
(with bounds for tiny smoke runs). At 32k these are 800 and 24,000 updates.
The original `LambdaLinearScheduler` reaches `3e-5` at the first-cycle boundary,
then switches to constant `6e-6` for approximately the remaining 8k updates.
Its original boundary convention is retained: update 24,001 uses `3e-5`, and
update 24,002 first uses `6e-6`. Only time parameters change, not the formula or
LR levels. This supersedes both 511fd07's cosine and 2e92b19's omitted low-LR tail.
The standalone launcher accepts `--max-steps` for separate experiments; counts and
schedule durations follow it. Production/final evaluation remains fixed at 32k.
Use `--smoke --smoke-steps 8` to test the stage transition on GPU; the default
four-step smoke is too short to actually update at the low constant LR.

Set `PYTHON` to the existing Cosmos environment's Python, `CUDA_VISIBLE_DEVICES`
to the exact GPU allocation, and export `MANIFEST`, `ROBOT_MAP`, `BASE_MODEL`,
`TOKENIZER`, `OUTPUT`, and `RUN_NAME`. Then run the matching script, for example:

```sh
bash bin/bench2dex/train_2_task.sh             # print and review the frozen plan
bash bin/bench2dex/train_2_task.sh --execute   # start/resume the 32k run
```

The scripts assert the bundle task count and reuse existing CUDA libraries; they
install nothing. Use a separate `RUN_NAME` with `--smoke --execute` for a four-step
proof, and the same smoke command plus `--verify-restore` for a reload-only proof.
Smoke also shortens its dataset and LR horizon to four updates. It is never a
production checkpoint or a continuation of the old 50k recipe.

Each optimizer-update row in `metrics.jsonl` also records `task_action_mse`
and `task_action_l1`, averaged by actual window count across all ranks and both
microbatches. These are denoised action-latent diagnostics (the same representation
as the existing action loss), not joint-angle error or rollout success. A task
absent from an update has JSON `null`, not a fabricated zero. The row includes
`grad_norm_before_clip`, `grad_clip_norm`, `grad_clip_scale`, and `grad_clipped`.
The scale is the clipping coefficient; it is not a second clipping operation.
Diagnostics are detached from autograd and committed before checkpoint saving.
Aggregate clipping frequency can be computed as the fraction of committed rows
with `grad_clipped=true`; select the resumed trajectory using session boundaries.

Bundle loading rechecks the grouping reference SHA256 and complete task IDs, so
the pinned grouping file must remain available at `grouping_path`, including on
evaluation hosts. A modified subset cannot silently select another GPU budget.

Task selection
is uniform in expectation, followed by uniform valid-window selection within
the task. Neither equal counts in every batch nor episode-uniform sampling is
implied. All task samples retain their own standard prompt and embedding while
using the same embodiment normalization.

Runs live under
`<output>/cosmos_policy/bench2dex_embodiment/<run-name>/`.
`run-identity.json` freezes the bundle, robot map, base/tokenizer hashes, Python
source fingerprint and launch recipe. Reusing a run name with different inputs
fails. `launcher.log` contains the torchrun output. A local filesystem lock
prevents duplicate launches of the same run; GPU allocation across different
runs remains the operator's responsibility.

## Recovery and retention

Rerun the same full or smoke command to resume. The launcher selects the newest
validated full checkpoint, falling back to an older validated checkpoint when
the latest save is incomplete. It refuses unidentified/foreign runs and refuses
to restart from scratch over unvalidated checkpoint artifacts.

Synchronous saves contain model, optimizer, scheduler and trainer state. Only
after the completion pointer matches, every component's metadata references
existing complete shard byte ranges, and the serialized trainer iteration
matches the directory, a `bench2dex-checkpoint.json` marker is written. Retention
keeps the newest two validated checkpoints of this bundle and always preserves
32k (and historical 50k finals). Partial, unmarked, foreign and symlinked directories are not deleted.
Interrupted artifacts can therefore require later manual cleanup.

The final update is saved even when it is not divisible by 5k (32k saves after 30k).

The sampler resumes at `completed_updates * global_batch`, so worker prefetch cannot
advance the committed data position. Existing model RNG restoration is not
extended; bitwise equivalence to uninterrupted CUDA training is not claimed.
Storage validation checks completeness; real FSDP model/optimizer reload remains
part of the GPU proof. No periodic independent evaluation runs during training.

## Evaluate each task with the shared checkpoint

Use the existing [final runner](bench2dex-final-evaluation.md) once per task,
adding `--embodiment-manifest "$BUNDLE"`. Keep `--manifest` and `--embedding`
pointed at that task's original audited artifacts. Use the same embodiment's
`iter_000032000` checkpoint for all tasks in the group, and distinct evaluation
output directories. The runner enforces the validated 32k checkpoint and records
the bundle hash. Each invocation starts a new policy server. The wrapper explicitly passes
`--device cpu` to the simulator and records CPU physics; rendering and policy
inference remain on GPU.

The runtime verifies membership and artifact hashes, selects that task's fixed
prompt/embedding, and uses shared statistics for both proprio normalization and
action denormalization. Omitting the embodiment bundle for a marked shared
checkpoint fails rather than silently reverting to per-task statistics.

For a standalone smoke server, use the same extra flag with:

```sh
python -m cosmos_policy.experiments.robot.bench2dex.runtime \
  --checkpoint "$CHECKPOINT" --embodiment-manifest "$BUNDLE" \
  --manifest "$TASK_MANIFEST" --embedding "$TASK_EMBEDDING" \
  --robot-map "$BENCH_REPO/robots/active_dof_maps.yml" \
  --task-id "$TASK_ID" --host 127.0.0.1 --port 9000
```

The standalone server accepts validated intermediate checkpoints for smoke
verification. The final evaluation wrapper requires 32k. Existing per-task
checkpoint selection remains unchanged when no embodiment bundle is supplied;
the final runner selects CPU physics for both modes.

## CPU checks

In an approved CPU test environment, install
`tests/requirements-bench2dex-cpu.txt` with
`uv pip install --no-config --python <python> -r tests/requirements-bench2dex-cpu.txt`.
This is a lightweight test environment, not a replacement for the CUDA setup.

```sh
python -m pytest -q \
  tests/test_bench2dex_contract.py \
  tests/test_bench2dex_runtime.py \
  tests/test_bench2dex_embodiment.py \
  tests/test_bench2dex_checkpoint_retention.py \
  tests/test_bench2dex_budget.py
```

The checkpoint tests include real CPU DCP serialization and a two-rank Gloo
save with rank-zero-only validation. The existing conditioning/loss tests import
the full model stack and should also run in the CUDA environment before a long
training job.

## Parameter audit and training observations

`run-identity.json` records the budget, expected per-task exposure, scheduler
parameters, base/tokenizer/data/code hashes and environment versions.
`config.yaml` and `resolved-training.json` record the effective configuration,
including inherited model, optimizer and callback settings.

`metrics.jsonl` writes one `update` row per completed optimizer update:
`loss` is the mean over every rank and both accumulated microbatches;
`lr_used` is the LR that performed that update, while `lr_next` is the value
installed by `scheduler.step()`. Torch LambdaLR initializes at index zero, so
update k uses schedule(k-1), and the scheduler reaches its minimum after update
`max_steps`. `samples_seen`, `task_samples_this_update` and `task_samples_seen`
count actual consumed windows, not prefetched windows. `start` rows and session
IDs mark restarts. When resuming, use the latest session plus the committed prefix
through its starting checkpoint; do not sum replayed rows from abandoned attempts.
The journal is flushed to disk before checkpoint saving. Missing matching metrics
at resume are rejected rather than guessed.

The audit distinguishes budget-sensitive and invariant settings:

| Setting | Treatment |
| --- | --- |
| GPU count, microbatch, accumulation, global batch, stream length | Derived and cross-checked against bundle task count and `max_steps` |
| Warmup and LR decay | Full-schedule time scaling: 75% first cycle, 2.5% warmup, then constant low LR |
| Sampler resume offset | Completed optimizer updates × global batch |
| Checkpoints | Every 5k plus final; synchronous; latest two plus protected final |
| Loss/LR/sample metrics | Every optimizer update, globally reduced; scheduler advances once per update |
| Console/device monitoring | Existing 5-update interval; not a training horizon |
| Periodic validation | Disabled; generation callback scheduled beyond the run budget |
| Optimizer | Existing FusedAdamW, weight decay 0.1, betas (0.9, 0.99), epsilon 1e-8; peak LR 1e-4 |
| Precision and clipping | Existing model precision and master weights; gradient scaler disabled; norm clipping 1.0 with nonfinite checks |
| EMA | Disabled; no sample-dependent EMA schedule is active |
| Context parallelism | One; GPUs are data-parallel ranks, checked against the recipe |
| Tokenizer compilation | Existing startup optimization; not a budget or LR schedule |
| Action horizon and control rate | 16 targets at 20Hz; unrelated to optimizer steps |
| Image/noise/loss settings | Existing 224-resolution, four-view policy recipe; captured in the resolved config |

Loss stability and a falling LR are diagnostics, not a substitute for task-level
closed-loop evaluation. Historical per-task runs and their configuration are not
modified by these changes.

## Revised-budget validation status (2026-11-05)

The final local CPU suite passed 43 tests plus three subtests; Ruff, shell syntax
and whitespace checks also passed. Checks cover all three budgets, 2/4/8-way sampling partitions and resume
offsets, the actual PyTorch LR scheduler state round-trip, partial metrics logs,
launcher guards, and real DCP retention including preservation of the 32k final.
The three shell entry points are exercised with paths containing spaces.

An earlier snapshot of this change passed 46 tests plus three subtests in the
remote Cosmos environment. A four-H100, GB128, four-update smoke was started in
an isolated `embodiment-budget-v2-20261105` directory. The last observed log reached
update 2 with finite gradients and a validated complete checkpoint. Its completion,
new-schedule GPU restore, and CPU-physics simulator smoke are **not verified**.
Local callback scheduling safeguards were tightened after that remote snapshot;
its evidence does not validate the final source revision. Eight-GPU execution has
not run. Earlier GPU/simulator proof below applies to the previous recipe only.

The owner paused instance access during this phase. No further remote polling,
cleanup, or workloads are performed until requested. Resume these hardware checks
before declaring the new production recipes fully validated.

## Historical validation completed on 2026-11-05

This evidence predates ADR 0008 and does not validate the revised budgets.


Historical implementation revision `24fd6c0` was exercised with the actual audited task03
and task79 datasets (100 episodes each), sharing `multi_xarm7_with_ability`.

| Check | Result |
| --- | --- |
| Complete Bench2Dex test suite in the existing CUDA environment | 35 tests and 3 subtests passed, including conditioning/loss and CPU DCP corruption/retention cases |
| Dataset hashes, joint contracts and recomputed shared statistics | Both tasks admitted to the same bundle |
| Two H100 GPUs, microbatch 16, accumulation 2, global batch 64 | Four optimizer updates completed with finite losses and gradients |
| Synchronous saves at updates 2 and 4 | Model, optimizer, scheduler and trainer components validated |
| Fresh two-rank reload at update 4 | Completed without additional optimizer updates |
| Separate run seeded from update 2, then resumed through update 4 | Completed and saved; optimizer momentum changed; scheduler advanced from 2 to 4 |
| Same checkpoint with a newly started service for each task | Both returned finite `(16, 34)` physical joint command chunks |
| Same observation and seed, different standard task descriptions | Actions differed; wrong descriptions were rejected |
| Real-artifact guards | Missing bundle, wrong task embedding and wrong task ID rejected; update 4 rejected by the final-50k gate; shared normalization verified |
| RTX 4090 Isaac Sim smoke | Both tasks completed one 32-control-step episode in each of the four channels, with no episode errors |
| Output inspection | All eight recordings contained finite joint states/actions and nonblank four-camera RGB; pilot seeds matched; two exported MP4s decoded |

The H100 run sampled a peak of approximately 75 GiB per GPU. An earlier attempt
sharing each card with roughly 23 GiB of another workload ran out of memory.
Four smoke updates reduce duration, not the peak memory required by microbatch 16.

Isaac Sim 5.1 failed during rendering initialization on H100 with
`ERROR_DEVICE_LOST`; the same closed-loop checks passed on RTX 4090. NVIDIA
[does not support GPUs without RT cores, including H100, for Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).
Use H100 for training and a supported RTX host for RGB simulation.

Evidence is retained under the private HDD2 directory
`/inspire/hdd2/project/roboticsystem2/ky26063/bench2dex_stuff/hqb/embodiment-validation-20261105/`:

- `tests.log`, `real-artifact-guards.json` and `continuation-state-check.json`.
- `runs/cosmos_policy/bench2dex_embodiment/smoke_exclusive_v1/` contains the original smoke checkpoints and training/reload log; `resume_from2_v1/` contains continuation evidence.
- `runtime-rtx-v1/` contains both task RPC outputs, eight simulator commands/logs/recordings, two videos, `prompt-switch-check.json`, `complete.json` and `artifact-check.json`.
- `exclusive.exit`, `restore.exit`, `continue.exit` and `runtime-rtx.exit` are all zero. Test policy and simulator processes were stopped after completion.

This is functional acceptance for the representative two-task embodiment, not
50k convergence evidence, a throughput estimate, or all-26-task evaluation.
Single-task/three-task sampling and retention edge cases are covered by automated
tests rather than separate full-model runs. The eight short rollouts are not
benchmark scores; final evaluation still requires the trained 50k checkpoint and
50 episodes per channel for each task. Bitwise continuation equivalence is not
claimed because model RNG restoration is outside this implementation.
