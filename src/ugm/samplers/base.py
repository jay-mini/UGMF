from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SamplingResult:
    """Final samples plus an explicitly requested sampling trajectory.

    ``trajectory[0]`` is the initial noise and ``trajectory[-1]`` is the
    final sample. ``timesteps`` records the time of every stored state.
    """

    samples: torch.Tensor
    trajectory: torch.Tensor
    timesteps: torch.Tensor


SampleOutput = torch.Tensor | SamplingResult


class BaseSampler(ABC):
    """Common interface and input handling shared by all samplers."""

    @abstractmethod
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        objective: nn.Module | None = None,
        shape: tuple[int, ...] | None = None,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        device: torch.device | str = "cuda",
        initial_noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> SampleOutput:
        raise NotImplementedError

    @staticmethod
    def _require_shape(
        shape: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if shape is None:
            raise ValueError("shape must be provided.")
        return tuple(shape)

    @staticmethod
    def _prepare_initial_noise(
        shape: tuple[int, ...],
        device: torch.device,
        initial_noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if not shape or shape[0] <= 0:
            raise ValueError("shape must include a positive batch dimension.")

        if initial_noise is None:
            return torch.randn(shape, device=device)

        if tuple(initial_noise.shape) != tuple(shape):
            raise ValueError(
                "initial_noise shape does not match requested shape: "
                f"got {tuple(initial_noise.shape)}, expected {tuple(shape)}."
            )
        if not initial_noise.is_floating_point():
            raise TypeError("initial_noise must be a floating-point tensor.")

        # Clone so sampling never mutates a caller-owned tensor in place.
        return initial_noise.to(device=device).clone()

    @staticmethod
    def _prepare_condition(
        cond: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if cond is None:
            return None
        if cond.ndim == 0 or cond.shape[0] != batch_size:
            raise ValueError(
                "cond must have the same batch dimension as shape: "
                f"got {tuple(cond.shape)}, expected first dimension "
                f"{batch_size}."
            )
        return cond.to(device=device)

    @staticmethod
    def _resolve_num_steps(
        num_steps: int | None,
        default: int,
    ) -> int:
        resolved = default if num_steps is None else int(num_steps)
        if resolved <= 0:
            raise ValueError("num_steps must be positive.")
        return resolved

    @staticmethod
    def _format_output(
        samples: torch.Tensor,
        trajectory: list[torch.Tensor] | None,
        timesteps: torch.Tensor,
    ) -> SampleOutput:
        if trajectory is None:
            return samples
        if len(trajectory) != timesteps.numel():
            raise RuntimeError(
                "Internal sampler error: trajectory and timestep lengths differ."
            )
        return SamplingResult(
            samples=samples,
            trajectory=torch.stack(trajectory, dim=0),
            timesteps=timesteps,
        )
