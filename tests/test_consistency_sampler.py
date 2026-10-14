from __future__ import annotations

import torch
from torch import nn

from ugm.samplers.base import SamplingResult
from ugm.samplers.consistency import ConsistencySampler


class EndpointZeroModel(nn.Module):
    """Return velocity that makes x + (1-t)v exactly zero."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x, t, cond=None):
        del cond
        self.calls += 1
        scale = 1.0 - t
        while scale.ndim < x.ndim:
            scale = scale[..., None]
        return -x / scale


def test_one_two_four_step_sampling_has_matching_nfe_and_endpoint() -> None:
    initial = torch.randn(3, 1, 2, 2)
    for steps in (1, 2, 4):
        model = EndpointZeroModel()
        torch.manual_seed(11)
        output = ConsistencySampler(steps).sample(
            model=model,
            shape=tuple(initial.shape),
            device="cpu",
            initial_noise=initial,
            num_steps=steps,
        )
        assert model.calls == steps
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)


def test_consistency_sampling_is_seed_reproducible() -> None:
    class ConstantModel(nn.Module):
        def forward(self, x, t, cond=None):
            del t, cond
            return torch.ones_like(x)

    initial = torch.zeros(2, 1, 2, 2)
    sampler = ConsistencySampler(4)
    torch.manual_seed(123)
    first = sampler.sample(
        ConstantModel(),
        shape=tuple(initial.shape),
        device="cpu",
        initial_noise=initial,
    )
    torch.manual_seed(123)
    second = sampler.sample(
        ConstantModel(),
        shape=tuple(initial.shape),
        device="cpu",
        initial_noise=initial,
    )
    assert torch.equal(first, second)


def test_consistency_trajectory_records_each_network_evaluation() -> None:
    initial = torch.randn(2, 1, 2, 2)
    output = ConsistencySampler(2).sample(
        EndpointZeroModel(),
        shape=tuple(initial.shape),
        device="cpu",
        initial_noise=initial,
        return_trajectory=True,
    )
    assert isinstance(output, SamplingResult)
    assert output.trajectory.shape[0] == 3
    assert torch.equal(output.trajectory[0], initial)
    assert torch.equal(output.timesteps, torch.tensor([0.0, 0.0, 0.5]))


def test_consistency_sampler_rejects_unsupported_budget() -> None:
    try:
        ConsistencySampler(3)
    except ValueError as error:
        assert "1, 2, or 4" in str(error)
    else:
        raise AssertionError("Expected unsupported consistency steps to fail.")
