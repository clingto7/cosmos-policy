import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_policy.datasets.bench2dex_embodiment import EmbodimentDataset, EmbodimentSampler
from cosmos_policy.experiments.robot.bench2dex.budget import Budget, BudgetCosineScheduler
from cosmos_policy.experiments.robot.bench2dex.metrics import MetricsJournal


@pytest.mark.parametrize(
    "tasks,gpus,micro,gb,total", [(1, 2, 16, 64, 2048000), (2, 4, 16, 128, 4096000), (3, 8, 12, 192, 6144000)]
)
def test_budget_rank_partition_and_resume(tasks, gpus, micro, gb, total):
    budget = Budget(tasks)
    assert (budget.world_size, budget.microbatch, budget.global_batch, budget.total_samples) == (gpus, micro, gb, total)
    assert budget.to_dict()["expected_samples_per_task"] == 2048000
    data = EmbodimentDataset([range(1)] * tasks, [str(i) for i in range(tasks)], total_samples=gb * 10)
    samplers = [EmbodimentSampler(data, gpus, r, gb) for r in range(gpus)]
    original = [list(s) for s in samplers]
    assert sorted(sum(original, [])) == list(range(gb * 10))
    for rank, sampler in enumerate(samplers):
        sampler.set_start_iteration(3)
        assert list(sampler) == original[rank][3 * micro * 2 :]
        sampler.set_start_iteration(10)
        assert list(sampler) == []
    with pytest.raises(ValueError):
        EmbodimentSampler(data, gpus, 0, gb + gpus)


@pytest.mark.parametrize("steps", [4, 100, 32000, 50000])
def test_cosine_warmup_end_and_real_torch_scheduler_resume(steps):
    budget = Budget(2, steps)
    curve = BudgetCosineScheduler(**budget.scheduler())
    assert curve.schedule(0) == pytest.approx(1e-6)
    assert curve.schedule(budget.warmup_steps) == pytest.approx(1)
    assert curve.schedule(steps) == pytest.approx(0.06)
    assert curve.schedule(steps + 100) == pytest.approx(0.06)
    values = [curve.schedule(i) for i in range(budget.warmup_steps, steps + 1)]
    assert np.all(np.diff(values) <= 0)
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([param], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, curve.schedule)
    for _ in range(3):
        optimizer.step()
        scheduler.step()
    state = scheduler.state_dict()
    optstate = optimizer.state_dict()
    other_opt = torch.optim.SGD([param], lr=1e-4)
    other = torch.optim.lr_scheduler.LambdaLR(other_opt, BudgetCosineScheduler(**budget.scheduler()).schedule)
    other_opt.load_state_dict(optstate)
    other.load_state_dict(state)
    optimizer.step()
    scheduler.step()
    other_opt.step()
    other.step()
    assert other.get_last_lr() == scheduler.get_last_lr()
    assert scheduler.get_last_lr()[0] == pytest.approx(1e-4 * curve.schedule(4))


def test_metrics_resume_excludes_uncheckpointed_updates_and_partial_tail(tmp_path):
    path = tmp_path / "metrics.jsonl"
    journal = MetricsJournal(path, ["03", "79"], 128, 0)
    journal.update(1, 1.0, [1e-10], [1e-4], {"03": 60, "79": 68})
    journal.update(2, 0.7, [1e-4], [8e-5], {"03": 66, "79": 62})
    journal.update(3, 0.6, [8e-5], [6e-5], {"03": 64, "79": 64})
    with path.open("a") as stream:
        stream.write('{"partial":')
    resumed = MetricsJournal(path, ["03", "79"], 128, 2)
    assert resumed.counts == {"03": 126, "79": 130}
    resumed.update(3, 0.5, [8e-5], [6e-5], {"03": 70, "79": 58})
    resumed.sync()
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    row = rows[-1]
    assert row["samples_seen"] == 384 and row["task_samples_seen"] == {"03": 196, "79": 188}
    assert rows[0]["session"] != row["session"]
    with pytest.raises(ValueError, match="global batch"):
        resumed.update(4, 0.5, [6e-5], [5e-5], {"03": 1, "79": 1})
    with pytest.raises(ValueError, match="matching metrics"):
        MetricsJournal(path, ["03", "79"], 128, 9)


@pytest.mark.parametrize("tasks", [1, 2, 3])
def test_training_scripts_require_inputs_and_pass_expected_task_guard(tmp_path, tasks):
    script = Path(__file__).resolve().parents[1] / f"bin/bench2dex/train_{tasks}_task.sh"
    spy = tmp_path / "python-spy"
    spy.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    spy.chmod(0o755)
    env = {**os.environ, "PYTHON": str(spy)}
    for name in ["MANIFEST", "ROBOT_MAP", "BASE_MODEL", "TOKENIZER", "OUTPUT", "RUN_NAME"]:
        env[name] = "path with spaces"
    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
    argv = result.stdout.splitlines()
    assert argv[argv.index("--expected-tasks") + 1] == str(tasks)
    assert argv[argv.index("--max-steps") + 1] == "32000"
    assert argv[argv.index("--manifest") + 1] == "path with spaces"
    assert "--execute" not in argv
