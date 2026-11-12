"""Runtime checks for the audited Bench2Dex training proof."""

import os

import torch

from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.callback import Callback


class FiniteTrainingStep(Callback):
    """Reject nonfinite loss or gradients before an optimizer can hide them."""

    def on_before_backward(self, model_ddp, loss: torch.Tensor, iteration: int = 0) -> None:
        if not torch.isfinite(loss.detach()).all().item():
            raise FloatingPointError(f"nonfinite Bench2Dex loss at iteration {iteration}")

    def on_after_backward(self, model_ddp, iteration: int = 0) -> None:
        gradients = [parameter.grad for parameter in model_ddp.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("Bench2Dex backward produced no gradients")
        finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()
        if not finite.item():
            raise FloatingPointError(f"nonfinite Bench2Dex gradients at iteration {iteration}")
        log.info(f"Bench2Dex iteration {iteration}: {len(gradients)} finite gradient tensors")


class EmbodimentCheckpointCallback(Callback):
    """Admit only complete synchronous saves to rolling retention."""

    def __init__(self, manifest_path: str):
        from cosmos_policy.experiments.robot.bench2dex.checkpoints import file_hash

        self.manifest_path = manifest_path
        self.manifest_sha256 = file_hash(manifest_path)

    def on_train_start(self, model, iteration: int = 0):
        import json
        from pathlib import Path

        identity = json.loads((Path(self.config.job.path_local) / "run-identity.json").read_text())
        if identity["manifest_sha256"] != self.manifest_sha256:
            raise ValueError("training manifest differs from the frozen run identity")

    def on_save_checkpoint_start(self, model, iteration: int = 0):
        from pathlib import Path

        from cosmos_policy._src.imaginaire.utils import distributed
        from cosmos_policy.experiments.robot.bench2dex.checkpoints import MARKER, file_hash

        if file_hash(self.manifest_path) != self.manifest_sha256:
            raise ValueError("embodiment manifest changed during training")

        # A retry can overwrite a previously interrupted save at this update.
        # Its old validation marker must not survive a second interrupted write.
        if distributed.is_rank0():
            marker = Path(self.trainer.checkpointer.save_dirname) / f"iter_{iteration:09}" / MARKER
            marker.unlink(missing_ok=True)
        distributed.barrier()

    def on_save_checkpoint_success(self, iteration: int = 0, elapsed_time: float = 0):
        from cosmos_policy._src.imaginaire.utils import distributed
        from cosmos_policy.experiments.robot.bench2dex.checkpoints import mark_and_prune

        if distributed.is_rank0():
            removed = mark_and_prune(self.trainer.checkpointer.save_dirname, iteration, self.manifest_path)
            log.info(f"Validated embodiment checkpoint {iteration}; removed {removed}")


class EmbodimentMetricsCallback(Callback):
    """Reduce loss over all ranks/microbatches and journal every optimizer update."""

    def __init__(self, task_ids):
        self.task_ids = list(task_ids)
        self.losses = []
        self.counts = dict.fromkeys(self.task_ids, 0)
        self.journal = None
        self.action_sums = []

    def on_train_start(self, model, iteration=0):
        import json
        from pathlib import Path

        from omegaconf import OmegaConf

        from cosmos_policy._src.imaginaire.utils import distributed
        from cosmos_policy.experiments.robot.bench2dex.budget import Budget
        from cosmos_policy.experiments.robot.bench2dex.metrics import MetricsJournal

        self.budget = Budget(len(self.task_ids), self.config.trainer.max_iter,
                             os.environ.get("BENCH2DEX_SINGLE_TASK_FOUR_GPU") == "1")
        root = Path(self.config.job.path_local)
        identity = json.loads((root / "run-identity.json").read_text())
        for key, value in self.budget.to_dict().items():
            if identity.get(key) != value:
                raise ValueError(f"resolved budget differs from launch identity: {key}")
        expected = self.budget.scheduler()
        for key, value in expected.items():
            if list(self.config.scheduler[key]) != value:
                raise ValueError(f"resolved scheduler differs from budget: {key}")
        if (
            self.config.optimizer.lr != identity["peak_lr"]
            or self.config.trainer.grad_accum_iter != self.budget.grad_accum_iter
            or self.config.dataloader_train.batch_size != self.budget.microbatch
            or self.config.dataloader_train.dataset.total_samples != self.budget.total_samples
            or distributed.get_world_size() != self.budget.world_size
            or self.config.trainer.grad_scaler_args.get("enabled", False)
        ):
            raise ValueError("resolved optimizer/data/distributed configuration differs from identity")
        if distributed.is_rank0():

            def serialize(value):
                if OmegaConf.is_config(value):
                    return OmegaConf.to_container(value, resolve=True)
                return str(value)

            (root / "resolved-training.json").write_text(json.dumps(self.config.to_dict(), default=serialize, indent=2))
            summary = json.loads((root / "dataset-summary.json").read_text())
            self.journal = MetricsJournal(
                root / "metrics.jsonl", self.task_ids, self.budget.global_batch, iteration,
                task_windows={t: summary["tasks"][t]["valid_windows"] for t in self.task_ids},
            )

            import socket

            device = torch.cuda.current_device()
            self.journal.append({
                "event": "runtime", "hostname": socket.gethostname(),
                "world_size": distributed.get_world_size(),
                "rank0_gpu": torch.cuda.get_device_name(device),
                "rank0_gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            })

    def on_training_step_batch_start(self, model, data_batch, iteration=0):
        for task_id in data_batch["task_id"]:
            self.counts[task_id] += 1

    def on_before_backward(self, model_ddp, loss, iteration=0):
        self.losses.append(loss.detach().double())

    def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
        self.lr_used = [float(group["lr"]) for group in optimizer.param_groups]

    def on_before_zero_grad(self, model, optimizer, scheduler, iteration=0):
        self.lr_next = [float(g["lr"]) for g in optimizer.param_groups]
        model = getattr(model, "module", model)
        stats = model.last_grad_clip_stats
        norm = float(stats["norm_before"].item())
        limit = float(stats["clip_norm"])
        self.grad_stats = {
            "grad_norm_before_clip": norm,
            "grad_clip_norm": limit,
            "grad_clip_scale": min(1.0, limit / (norm + 1e-6)),
            "grad_clipped": norm > limit,
        }

    def on_training_step_batch_end(self, model, data_batch, output_batch, loss, iteration=0):
        import torch.distributed as dist

        mse = output_batch["action_mse_per_sample"].detach().double()
        l1 = output_batch["action_l1_per_sample"].detach().double()
        if mse.shape != (len(data_batch["task_id"]),) or l1.shape != mse.shape:
            raise ValueError("action diagnostics must contain one value per sampled window")
        self.action_sums.append(torch.stack([
            value[torch.tensor([t == task for t in data_batch["task_id"]], device=value.device)].sum()
            for task in self.task_ids for value in (mse, l1)
        ]))
        if len(self.losses) < self.budget.grad_accum_iter:
            return
        if len(self.losses) != self.budget.grad_accum_iter:
            raise ValueError("loss accumulation does not match optimizer update")
        values = torch.cat([
            torch.stack([sum(self.losses), *[
                self.losses[0].new_tensor(self.counts[t]) for t in self.task_ids
            ]]),
            torch.stack(self.action_sums).sum(0),
        ])
        dist.all_reduce(values)
        values = values.tolist()
        counts = dict(zip(self.task_ids, (int(v) for v in values[1:1 + len(self.task_ids)])))
        action = values[1 + len(self.task_ids):]
        if self.journal is not None:
            self.journal.update(
                iteration + 1,
                values[0] / (self.budget.world_size * self.budget.grad_accum_iter),
                self.lr_used, self.lr_next, counts,
                diagnostics={
                    **self.grad_stats,
                    "task_action_mse": {t: action[2*i] / counts[t] if counts[t] else None
                                        for i, t in enumerate(self.task_ids)},
                    "task_action_l1": {t: action[2*i+1] / counts[t] if counts[t] else None
                                       for i, t in enumerate(self.task_ids)},
                },
            )
        self.losses.clear()
        self.action_sums.clear()
        self.counts = dict.fromkeys(self.task_ids, 0)

    def on_save_checkpoint_start(self, model, iteration=0):
        if self.journal is not None:
            self.journal.sync()

    def on_train_end(self, model, iteration=0):
        if self.journal is not None:
            self.journal.sync()
