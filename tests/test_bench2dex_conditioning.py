"""Demo action targets stay denoising targets, not rollout conditions."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel


class _Condition:
    def __init__(self, batch_size: int, latent: torch.Tensor) -> None:
        self.condition_video_input_mask_B_C_T_H_W = torch.zeros(
            batch_size, 1, latent.shape[2], latent.shape[3], latent.shape[4]
        )
        self.gt_frames = latent

    def set_video_condition(self, *, gt_frames: torch.Tensor, **kwargs):
        self.gt_frames = gt_frames
        return self


def test_demo_mask_never_conditions_action_but_rollout_world_model_does() -> None:
    batch_size = 2
    latent = torch.zeros(batch_size, 16, 13, 2, 2)
    condition = _Condition(batch_size, latent)
    model = object.__new__(CosmosPolicyVideo2WorldModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        min_num_conditional_frames=1,
        max_num_conditional_frames=1,
        conditional_frames_probs=None,
        mask_value_prediction_loss_for_policy_prediction=True,
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
    )
    model.tensor_kwargs = {}
    batch = {
        "rollout_data_mask": torch.tensor([0, 1]),
        # Deliberately set demo subtype mask to 1 to verify rollout gating.
        "world_model_sample_mask": torch.tensor([1, 1]),
        "value_function_sample_mask": torch.tensor([0, 0]),
        "action_latent_idx": torch.tensor([6, 6]),
        "value_latent_idx": torch.tensor([12, 12]),
        "actions": torch.zeros(batch_size, 16, 26),
    }

    with (
        patch.object(CosmosPolicyDiffusionModel, "get_data_and_condition", return_value=(None, latent, condition)),
        patch(
            "cosmos_policy.models.policy_video2world_model.replace_latent_with_action_chunk",
            side_effect=lambda frames, actions, action_indices: frames,
        ),
    ):
        _, _, result = model.get_data_and_condition(batch)

    action_mask = result.condition_video_input_mask_B_C_T_H_W[:, :, 6]
    assert torch.count_nonzero(action_mask[0]) == 0
    assert torch.all(action_mask[1] == 1)
