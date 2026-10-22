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
