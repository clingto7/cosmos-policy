# Use a fixed 25k budget for new per-task runs

Decision date: 2026-11-03. Owner direction: train new checkpoints on two GPUs;
defer reconciliation and retraining of old checkpoints until later. Do not modify
the currently running jobs or their existing relay scripts in this change.

Each new task uses two H100 80GB GPUs, microbatch 16 per GPU, accumulation 2,
global batch 64, and 25,000 optimizer updates. Save full training state every
5,000 updates. Select the 25k checkpoint by budget. Each task starts from the same Video2World base weights using its audited
100 demonstrations and four RGB views. Preserve the existing optimizer and
learning-rate schedule, training seed 0 and sampler seed 0; freeze source and
configuration provenance before admission.

This is a common training-budget policy, not a claim that 25k is optimal or that
every task has converged. It supersedes the convergence-based budget-selection
requirement in ADR 0004 for the new batch. Task03's completed 20k–50k curve is
exploratory evidence; its non-monotonic scores do not establish a plateau.
Equal updates and batch size provide equal sampled-window budgets, not equal
epochs or equal compute across models and tasks.

After validating the fixed 25k checkpoint, freeze selection and run the reference
final protocol directly: 50 episodes per channel (200 episodes). There is no
separate pilot evaluation for new runs. Historical pilots remain exploratory
evidence. A valid low or zero final SR does not trigger extra training or
checkpoint shopping within this fixed-budget comparison. Infrastructure or data-contract failures block
that task and are diagnosed separately. Final seeds remain untouched until the
checkpoint and protocol are frozen. Retain ADR 0005's channel definitions,
matched anchors, expert-derived horizons and metric formulas.

Old runs, including four-GPU histories and runs extending past 25k, remain
historical evidence. Their admission to the uniform batch will be decided later;
do not delete them or silently relabel them as compliant new runs. Task03 may
be included in that later catch-up batch.

The operational contract is [the batch workflow](../experiments/fixed-budget-batch.md).
This document defines future behavior; it does not deploy a scheduler or change
live relay jobs. Issue #10 owns batch execution; issue #11 owns final evaluation.
