import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cosmos_policy.datasets.bench2dex_embodiment import EmbodimentDataset, EmbodimentSampler
from cosmos_policy.experiments.robot.bench2dex.budget import Budget
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


@pytest.mark.parametrize("steps", [4, 8, 100, 32000, 40000, 50000])
def test_original_linear_warmup_end_and_real_torch_scheduler_resume(steps, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "cosmos_policy._src.imaginaire.utils.log", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cosmos_policy._src.imaginaire.utils.distributed", SimpleNamespace())
    path = Path(__file__).parents[1] / "cosmos_policy/_src/imaginaire/functional/lr_scheduler.py"
    spec = importlib.util.spec_from_file_location("original_scheduler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    LinearScheduler = module.LambdaLinearScheduler
    budget = Budget(2, steps)
    curve = LinearScheduler(**budget.scheduler())
    assert curve.schedule(0) == pytest.approx(1e-6)
    assert curve.schedule(budget.warmup_steps) == pytest.approx(1)
    assert curve.schedule(budget.first_cycle_steps) == pytest.approx(0.3)
    assert curve.schedule(budget.first_cycle_steps + 1) == pytest.approx(0.06)
    # Preserve the original curve, with only its time axis scaled.
    midpoint = (budget.first_cycle_steps + budget.warmup_steps) / 2
    assert curve.schedule(midpoint) == pytest.approx(0.65)
    assert curve.schedule(steps + 100) == pytest.approx(0.06)
    values = [curve.schedule(i) for i in range(budget.warmup_steps, steps + 1)]
    assert np.all(np.diff(values) <= 0)
    param = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([param], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, curve.schedule)
    for _ in range(budget.first_cycle_steps):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    state = scheduler.state_dict()
    optstate = optimizer.state_dict()
    other_opt = torch.optim.SGD([param], lr=1e-4)
    other = torch.optim.lr_scheduler.LambdaLR(other_opt, LinearScheduler(**budget.scheduler()).schedule)
    other_opt.load_state_dict(optstate)
    other.load_state_dict(state)
    optimizer.step()
    scheduler.step()
    other_opt.step()
    other.step()
    assert other.get_last_lr() == scheduler.get_last_lr()
    assert scheduler.get_last_lr()[0] == pytest.approx(6e-6)


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


def test_metrics_callback_aggregates_microbatches_and_missing_tasks(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace
    # Only the callback framework/logging are stubbed; the aggregation and journal
    # below execute the real production code and real torch tensor operations.
    monkeypatch.setitem(sys.modules, "cosmos_policy._src.imaginaire.utils.log", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "cosmos_policy._src.imaginaire.utils.callback", SimpleNamespace(Callback=object))
    spec = importlib.util.spec_from_file_location("proof_under_test", Path(__file__).parents[1] / "cosmos_policy/bench2dex_proof.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    EmbodimentMetricsCallback = module.EmbodimentMetricsCallback

    callback = EmbodimentMetricsCallback(['03', '79'])
    callback.budget = SimpleNamespace(grad_accum_iter=2, world_size=1)
    callback.journal = MetricsJournal(tmp_path / 'metrics.jsonl', ['03', '79'], 4, 0)
    monkeypatch.setattr(torch.distributed, 'all_reduce', lambda values: None)
    model = SimpleNamespace(last_grad_clip_stats={'norm_before': torch.tensor(2.), 'clip_norm': 1.})
    optimizer = SimpleNamespace(param_groups=[{'lr': 1e-4}])
    for tasks, mse, l1 in [(['03', '03'], [1., 3.], [1., 1.]), (['03', '79'], [8., 10.], [4., 2.])]:
        data = {'task_id': tasks}
        callback.on_training_step_batch_start(model, data)
        callback.on_before_backward(model, torch.tensor(2.))
        if len(callback.losses) == 2:
            callback.on_before_optimizer_step(model, optimizer, None, None)
            optimizer.param_groups[0]['lr'] = 9e-5
            callback.on_before_zero_grad(model, optimizer, None)
        callback.on_training_step_batch_end(model, data, {
            'action_mse_per_sample': torch.tensor(mse), 'action_l1_per_sample': torch.tensor(l1)
        }, None)
    row = json.loads((tmp_path / 'metrics.jsonl').read_text().splitlines()[-1])
    assert row['task_action_mse'] == {'03': 4., '79': 10.}
    assert row['task_action_l1'] == {'03': 2., '79': 2.}
    assert row['grad_norm_before_clip'] == 2. and row['grad_clipped'] is True
    assert row['grad_clip_scale'] == pytest.approx(.5)
    assert row['lr_used'] == [1e-4] and row['lr_next'] == [9e-5]
    # A task absent from an entire update has null diagnostics, never NaN/zero.
    for _ in range(2):
        data = {'task_id': ['03', '03']}
        callback.on_training_step_batch_start(model, data, iteration=1)
        callback.on_before_backward(model, torch.tensor(1.), iteration=1)
        callback.on_training_step_batch_end(model, data, {
            'action_mse_per_sample': torch.ones(2), 'action_l1_per_sample': torch.ones(2)
        }, None, iteration=1)
    row = json.loads((tmp_path / 'metrics.jsonl').read_text().splitlines()[-1])
    assert row['task_action_mse']['79'] is None
    assert row['task_samples_seen'] == {'03': 7, '79': 1}


def test_complete_libero_recipe_time_scaling():
    budget = Budget(2, 32000)
    assert budget.first_cycle_steps == 24000 and budget.warmup_steps == 800
    original = Budget(2, 40000)
    assert original.first_cycle_steps == 30000 and original.warmup_steps == 1000
    assert budget.scheduler()["f_min"] == original.scheduler()["f_min"] == [0.3, 0.06]
