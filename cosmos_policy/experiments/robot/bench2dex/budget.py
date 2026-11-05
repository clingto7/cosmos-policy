"""Single source of truth for optimizer-step budgets and sample exposure."""

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
            cycle_lengths=[self.max_steps, 100000000000000],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
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
            "first_cycle_min_lr_ratio": 0.3,
            "constant_lr_ratio": 0.06,
            "start_lr_ratio": 1e-6,
        }
