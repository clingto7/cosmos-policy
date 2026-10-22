"""The public policy loss supervises action and future state, never return placeholders."""

from types import SimpleNamespace

import torch

from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel


def test_demo_loss_uses_four_future_views_and_ignores_value_placeholder() -> None:
    weight = torch.nn.Parameter(torch.tensor(0.5))

    class SDE:
        def marginal_prob(self, x0, sigma):
            return x0, torch.zeros_like(sigma)

    model = SimpleNamespace(
        config=SimpleNamespace(
            mask_loss_for_action_future_state_prediction=False,
            mask_value_prediction_loss_for_policy_prediction=True,
            mask_current_state_action_for_value_prediction=False,
            mask_future_state_for_qvalue_prediction=False,
            action_loss_multiplier=1,
        ),
        sde=SDE(),
        get_per_sigma_loss_weights=lambda sigma: torch.ones_like(sigma),
        denoise=lambda xt, sigma, condition: SimpleNamespace(
            x0=weight * xt.mean(dim=2, keepdim=True).expand_as(xt)
        ),
    )
    indices = lambda value: torch.tensor([value])

    def loss(value: float):
        _, masked, _, _ = CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma(
            model,
            torch.ones(1, 16, 13, 2, 2),
            SimpleNamespace(),
            torch.zeros(1, 16, 13, 2, 2),
            torch.ones(1, 13),
            action_chunk=torch.ones(1, 16, 2),
            action_indices=indices(6),
            proprio=torch.ones(1, 2),
            current_proprio_indices=indices(1),
            future_proprio=torch.ones(1, 2),
            future_proprio_indices=indices(7),
            future_wrist_image_indices=indices(8),
            future_wrist_image2_indices=indices(9),
            future_image_indices=indices(10),
            future_image2_indices=indices(11),
            rollout_data_mask=indices(0),
            world_model_sample_mask=indices(0),
            value_function_sample_mask=indices(0),
            value_function_return=torch.tensor([value]),
            value_indices=indices(12),
        )
        return masked

    baseline = loss(-100)
    assert torch.equal(baseline, loss(1_000_000))
    assert torch.all(baseline[:, :, 6:12] > 0)
    assert torch.count_nonzero(baseline[:, :, :6]) == 0
    assert torch.count_nonzero(baseline[:, :, 12]) == 0
    baseline.mean().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad)
