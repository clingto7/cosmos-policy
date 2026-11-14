"""Manual requests, failures, and real two-rank checkpoint round trips on CPU."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from cosmos_policy.experiments.robot.bench2dex.checkpoint_control import CheckpointControl, submit
from cosmos_policy.experiments.robot.bench2dex.checkpoints import mark_and_prune, validate_checkpoint


def test_save_continue_stop_and_periodic_overlap(tmp_path):
    control = CheckpointControl(tmp_path)
    calls = []
    receipt = submit(tmp_path, "save")
    assert not receipt.exists()
    assert not control.after_update(13801, lambda: calls.append(13801), periodic=True)
    assert calls == [13801]
    assert json.loads(receipt.read_text())["iteration"] == 13801
    assert not control.after_update(13802, lambda: calls.append(13802))
    assert calls == [13801]  # Acknowledged requests must never execute twice.
    receipt = submit(tmp_path, "save-and-stop")
    assert control.after_update(13803, lambda: calls.append(13803))
    assert json.loads(receipt.read_text())["status"] == "saved"
    with pytest.raises(ValueError, match="not accepting"):
        submit(tmp_path, "save")
    control.close()
    assert json.loads((control.root / "session.json").read_text())["state"] == "stopped"


def test_failed_save_has_no_success_receipt(tmp_path):
    control = CheckpointControl(tmp_path)
    receipt = submit(tmp_path, "save-and-stop")

    def fail():
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        control.after_update(13801, fail)
    assert not receipt.exists()
    # A new process generation cannot replay this failed request.
    restarted = CheckpointControl(tmp_path)
    assert not restarted.after_update(10001, lambda: pytest.fail("stale request replayed"))


def test_invalid_request_and_unsupported_old_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        submit(tmp_path, "save")
    control = CheckpointControl(tmp_path)
    (control.directory / "bad.request.json").write_text("broken json")
    assert not control.after_update(1, lambda: pytest.fail("invalid request saved"))
    assert json.loads((control.directory / "bad.result.json").read_text())["status"] == "rejected"


def _distributed_worker(rank, root):
    root = Path(root)
    dist.init_process_group("gloo", init_method=f"file://{root / 'rendezvous'}", rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 100)
        control = CheckpointControl(root)
        dist.barrier()
        if rank == 0:
            submit(root, "save")
        dist.barrier()
        manifest = root / "manifest.json"
        if rank == 0:
            manifest.write_text("{}")
        dist.barrier()
        saved = []
        for iteration in range(1, 5):
            # Requests are handled only after both accumulated microbatches.
            for _ in range(2):
                model(torch.ones(1, 2)).sum().div(2).backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            def save():
                saved.append(iteration)
                path = root / "checkpoints" / f"iter_{iteration:09}"
                for component, state in {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "trainer": {"iteration": iteration},
                }.items():
                    dcp.save(state, checkpoint_id=path / component)
                if rank == 0:
                    (path.parent / "latest_checkpoint.txt").write_text(path.name)
                    mark_and_prune(path.parent, iteration, manifest)

            stop = control.after_update(iteration, save, periodic=iteration == 1)
            if stop:
                break
            if rank == 0:
                submit(root, "save-and-stop")
            dist.barrier()
        assert iteration == 2
        assert saved == [1, 2]  # Periodic + manual collision saves once on every rank.
        assert scheduler.last_epoch == 2
        control.close()
        (root / f"rank-{rank}.json").write_text(json.dumps(saved))
    finally:
        dist.destroy_process_group()


def test_two_rank_save_continue_then_stop_and_restore(tmp_path):
    mp.spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2, join=True)
    assert json.loads((tmp_path / "rank-0.json").read_text()) == [1, 2]
    assert json.loads((tmp_path / "rank-1.json").read_text()) == [1, 2]
    checkpoint = tmp_path / "checkpoints/iter_000000002"
    assert validate_checkpoint(checkpoint)["iteration"] == 2
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Populate state slots before DCP loading into the optimizer state dict.
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    dcp.load(state, checkpoint_id=checkpoint / "optim")
    optimizer.load_state_dict(state)
    assert all(s["step"].item() == 2 for s in optimizer.state.values())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 100)
    state = scheduler.state_dict()
    dcp.load(state, checkpoint_id=checkpoint / "scheduler")
    scheduler.load_state_dict(state)
    assert scheduler.last_epoch == 2
    receipts = list((tmp_path / "checkpoint-control").glob("*/*.result.json"))
    assert sorted(json.loads(p.read_text())["iteration"] for p in receipts) == [1, 2]


@pytest.mark.parametrize("action,expected_updates", [("save", 3), ("save-and-stop", 1)])
def test_production_train_loop_handles_request_after_accumulation(tmp_path, monkeypatch, action, expected_updates):
    # Execute the actual train method with CPU doubles for CUDA/model imports.
    import ast
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from unittest.mock import Mock

    import cosmos_policy.experiments.robot.bench2dex.checkpoint_control as module

    source = Path(__file__).parents[1] / "cosmos_policy/trainer.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CosmosPolicyTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    namespace = {
        "os": __import__("os"),
        "signal": NS(alarm=lambda _: None),
        "torch": NS(amp=NS(GradScaler=lambda *a, **k: None)),
        "distributed": NS(barrier=lambda: None),
        "log": Mock(),
        "misc": NS(to=lambda batch, **kwargs: batch),
        "maybe_enable_profiling": lambda *a, **k: nullcontext(),
        "maybe_enable_memory_snapshot": lambda *a, **k: nullcontext(),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    monkeypatch.setenv("BENCH2DEX_EMBODIMENT_MANIFEST", "bundle.json")
    original = module.CheckpointControl

    def control_factory(root):
        control = original(root)
        submit(root, action)
        return control

    monkeypatch.setattr(module, "CheckpointControl", control_factory)
    config = NS(
        trainer=NS(
            memory_format=None,
            grad_scaler_args={},
            distributed_parallelism="fsdp",
            run_validation=False,
            max_iter=3,
            timeout_period=60,
            straggler_detection=NS(analyze_dataloading=False),
        ),
        optimizer=None,
        scheduler=None,
        job=NS(path_local=tmp_path),
        checkpoint=NS(dcp_async_mode_enabled=False, save_iter=3),
    )
    model = Mock(training=True)
    model.to.return_value = model
    model.init_optimizer_scheduler.return_value = (object(), object())
    saves, microbatches = [], []

    def training_step(*args, **kwargs):
        microbatches.append(kwargs["iteration"])
        return None, None, len(microbatches) % 2

    trainer = NS(
        config=config,
        callbacks=Mock(),
        training_step=training_step,
        checkpointer=NS(
            load=lambda *a: 0, save=lambda *a, iteration: saves.append((iteration, len(microbatches))), finalize=Mock()
        ),
        training_timer=lambda *a: nullcontext(),
        straggler_detector=NS(profile_section=lambda *a, **k: nullcontext(), generate_report=Mock()),
    )

    class Loader:
        sampler = Mock()

        def __iter__(self):
            return iter(range(10))

    namespace["train"](trainer, model, Loader(), None)
    assert len(microbatches) == expected_updates * 2
    assert saves == ([(1, 2), (3, 6)] if action == "save" else [(1, 2)])
    trainer.callbacks.on_train_end.assert_called_once_with(model, iteration=expected_updates)
    trainer.checkpointer.finalize.assert_called_once()
    assert json.loads((tmp_path / "checkpoint-control/session.json").read_text())["state"] == "stopped"
