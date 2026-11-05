"""Runtime checks for the audited Bench2Dex training proof."""

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
