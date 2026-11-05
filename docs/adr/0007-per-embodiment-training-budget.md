# Train one task-conditioned policy per embodiment with a fixed 50k budget

On 2026-11-05, the owner chose one multi-task checkpoint per Bench2Dex embodiment, following Bench2Dex's existing task grouping, with two H100 GPUs, microbatch 16 per GPU, gradient accumulation 2, global batch 64, and 50,000 optimizer updates per embodiment. The budget balances the owner's earlier per-task learning results at batch 64 and 25k updates against acceptable training time; it is a fixed budget, not a claim of multi-task convergence or equal per-task exposure.

> Budget superseded on 2026-11-05 by [ADR 0008](0008-task-count-training-budget.md).
> The shared data, normalization and task-conditioning decisions remain in effect.

## Accepted decisions

- Follow Bench2Dex's physical embodiment grouping, identified by `meta/robot_key`. Train one checkpoint per embodiment, including the single-task groups. Use the same original Video2World base weights for every group.
- Sample tasks uniformly within each embodiment, then sample valid windows uniformly within the selected task. Retain each task's audited demonstrations and the existing four-camera RGB, named active-joint, 16-step absolute-action, 20 Hz contract.
- Compute shared action and proprioception min-max statistics over all valid training frames of the embodiment's included tasks, excluding homing and invalid frames. Use the same statistics for training and every task's inference with that checkpoint.
- Use each task's full standard scene description and its matching precomputed text embedding. Supporting paraphrases or unseen instructions is outside this version's scope.
- Restart the policy server for each evaluated task, loading the same embodiment checkpoint with that task's standard description and embedding. Switching tasks inside a running server is not required.
- Train for 50k optimizer updates without periodic independent evaluation or checkpoint selection by validation performance. Save full resumable training state every 5k updates to limit lost progress after instance reclamation. Retain the latest two complete checkpoints during training; remove older checkpoints only after the new save has completed and passed validation. Preserve the final 50k complete training state. Saves must include the model, optimizer, scheduler, and other state required to resume training.
- Evaluate the fixed 50k checkpoint separately on each task, using the existing four-channel protocol with 50 episodes per channel and task-level metrics.

## Budget interpretation

Each embodiment receives 3.2 million sampled windows. With one, two, or three tasks, uniform task sampling gives expected per-task exposure of 3.2 million, 1.6 million, or approximately 1.067 million windows, respectively. The old batch-64, 25k per-task budget was 1.6 million windows; equal embodiment budgets therefore do not imply equal task exposure or convergence.

The owner chose to try uniform sampling and 50k first. Additional training beyond 50k and the reporting or classification of any such results are explicitly outside this phase. Revisit these questions when the 50k stage is reached; no extension budget or reporting policy is decided, and these deferred questions do not block this phase.

## Grouping evidence and scope

In the local Bench2Dex checkout, `policy/GR00T_CE/configs/embodiments.yaml` registers the 12 physical embodiments and `policy/GR00T_XE/xe_norm_stats.taskstats.json` records the existing 26-task grouping. `policy/GR00T_CE/data/discovery.py` derives identity from HDF5 metadata; `robots/active_dof_maps.yml` supplies the joint contract. These existing definitions guide grouping; actual training manifests must still validate robot identity and joint ordering. Scene filenames and action dimensions are not embodiment identities.

This decision scopes the planned per-embodiment training mode and replaces per-task isolation and statistics from ADR 0001 for that mode. Existing per-task runs and ADR 0006's historical protocol remain unchanged. Implementation and job launch are not authorized by this record.

Implementation commands, recovery semantics and GPU verification gates are documented in the [per-embodiment workflow](../experiments/per-embodiment-training.md).
