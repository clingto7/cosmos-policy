"""Exercise the production loop at reclamation-window and resume boundaries."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "start,end,interval,extra,expected",
    [
        (14999, 17500, 2500, [16500], [15000, 16500, 17500]),
        (16500, 17500, 2500, [16500], [17500]),
        (16499, 16500, 2500, [16500], [16500]),
        (17499, 17500, 2500, [17500], [17500]),
        (31999, 32000, 2500, [16500], [32000]),
        (32000, 32000, 2500, [16500], []),
        (0, 8, 2, [3], [2, 3, 4, 6, 8]),
    ],
)
def test_actual_trainer_save_boundaries(start, end, interval, extra, expected):
    source = Path(__file__).parents[1] / "cosmos_policy/trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CosmosPolicyTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "signal": NS(alarm=lambda _: None),
        "torch": NS(amp=NS(GradScaler=lambda *a, **k: None)),
        "distributed": NS(barrier=lambda: None),
        "log": Mock(),
        "misc": NS(to=lambda batch, **kwargs: batch),
        "maybe_enable_profiling": lambda *a, **k: nullcontext(),
        "maybe_enable_memory_snapshot": lambda *a, **k: nullcontext(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    config = NS(
        trainer=NS(memory_format=None, grad_scaler_args={}, distributed_parallelism="fsdp",
                   run_validation=False, max_iter=end, timeout_period=60,
                   straggler_detection=NS(analyze_dataloading=False)),
        optimizer=None, scheduler=None,
        checkpoint=NS(save_iter=interval, extra_save_iters=extra),
    )
    model = Mock(training=True)
    model.to.return_value = model
    model.init_optimizer_scheduler.return_value = (object(), object())
    saves, microbatches = [], []

    def step(*args, **kwargs):
        microbatches.append(kwargs["iteration"])
        return None, None, len(microbatches) % 2

    trainer = NS(
        config=config, callbacks=Mock(), training_step=step,
        checkpointer=NS(load=lambda *a: start,
                        save=lambda *a, iteration: saves.append((iteration, len(microbatches))),
                        finalize=Mock()),
        training_timer=lambda *a: nullcontext(),
        straggler_detector=NS(profile_section=lambda *a, **k: nullcontext(), generate_report=Mock()),
    )

    class Loader:
        sampler = Mock()

        def __iter__(self):
            return iter([{}] * (2 * (end - start)))

    loader = Loader()
    namespace["train"](trainer, model, loader, None)
    assert [i for i, _ in saves] == expected
    assert all(micro == (i - start) * 2 for i, micro in saves)
    loader.sampler.set_start_iteration.assert_called_once_with(start)
