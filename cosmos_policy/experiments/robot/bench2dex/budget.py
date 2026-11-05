"""Single source of truth for optimizer-step budgets and sample exposure."""

import math
from dataclasses import asdict, dataclass

FINAL_STEPS = 32_000


@dataclass(frozen=True)
class Budget:
    task_count: int
    max_steps: int = FINAL_STEPS

    def __post_init__(self):
        if self.task_count not in (1, 2, 3) or self.max_steps < 2:
            raise ValueError("budget requires 1/2/3 tasks and at least two optimizer steps")

    @property
    def world_size(self):
        return {1: 2, 2: 4, 3: 8}[self.task_count]

    @property
    def microbatch(self):
        return 12 if self.task_count == 3 else 16

    @property
    def grad_accum_iter(self):
        return 2

    @property
    def global_batch(self):
        return self.world_size * self.microbatch * self.grad_accum_iter

    @property
    def total_samples(self):
        return self.global_batch * self.max_steps

    @property
    def warmup_steps(self):
        # Preserve the original 1000/30000 warmup fraction, including short smoke runs.
        return min(self.max_steps - 1, max(1, round(self.max_steps / 30)))

    def scheduler(self):
        return dict(
            warm_up_steps=[self.warmup_steps, 0],
            cycle_lengths=[self.max_steps, self.max_steps],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.06, 0.06],
        )

    def to_dict(self):
        return {
            **asdict(self),
            **{
                k: getattr(self, k)
                for k in (
                    "world_size",
                    "microbatch",
                    "grad_accum_iter",
                    "global_batch",
                    "total_samples",
                    "warmup_steps",
                )
            },
            "expected_samples_per_task": self.total_samples / self.task_count,
            "peak_lr": 1e-4,
            "min_lr_ratio": 0.06,
            "start_lr_ratio": 1e-6,
        }


class BudgetCosineScheduler:
    """Warmup then one cosine decay, clamped at the minimum beyond max_steps.

    LambdaLR starts at index zero; update k uses schedule(k-1), and its
    scheduler.step() sets schedule(k) for the next update.
    """

    def __init__(self, warm_up_steps, cycle_lengths, f_start, f_max, f_min, verbosity_interval=0):
        max_steps, warmup_steps = cycle_lengths[0], warm_up_steps[0]
        min_ratio, start_ratio = f_min[0], f_start[0]
        if f_max != [1.0, min_ratio] or f_min != [min_ratio, min_ratio] or f_start[1] != min_ratio:
            raise ValueError("expected a single cosine followed by a constant minimum")
        if not 0 < warmup_steps < max_steps or not 0 <= start_ratio <= min_ratio <= 1:
            raise ValueError("invalid cosine budget")
        self.max_steps, self.warmup_steps = max_steps, warmup_steps
        self.min_ratio, self.start_ratio = min_ratio, start_ratio

    def schedule(self, step, **kwargs):
        step = max(0, min(step, self.max_steps))
        if step < self.warmup_steps:
            return self.start_ratio + (1 - self.start_ratio) * step / self.warmup_steps
        t = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_ratio + (1 - self.min_ratio) * (1 + math.cos(math.pi * t)) / 2
