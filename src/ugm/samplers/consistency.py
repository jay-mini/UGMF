from __future__ import annotations

import torch
from torch import nn

from ugm.models.conditioning import guided_prediction, validate_guidance_scale
from ugm.samplers.base import BaseSampler, SampleOutput


class ConsistencySampler(BaseSampler):
    """One/few-step sampler matching the FM consistency training convention.

    With ``K`` network evaluations, the model is evaluated at
    ``t_i = i / K`` for ``i=0,...,K-1``.  The first input is Gaussian noise.
    Before every later evaluation, the previous data-end estimate is mixed
    with fresh Gaussian noise at the next time:

        x_t = (1 - t) * noise + t * x1_estimate.

    The boundary prediction is ``x + (1-t)F(x,t)``.  Supported budgets are
    deliberately fixed to 1, 2, and 4 evaluations for this first scheme.
    """

    SUPPORTED_STEPS = frozenset({1, 2, 4})

    def __init__(self, num_steps: int = 1) -> None:
        self.num_steps = self._validate_steps(num_steps)

    @classmethod
    def _validate_steps(cls, value: int) -> int:
        steps = int(value)
        if steps not in cls.SUPPORTED_STEPS:
            raise ValueError("Consistency sampling steps must be one of 1, 2, or 4.")
        return steps

    @staticmethod
    def _boundary_prediction(
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        velocity = guided_prediction(
            model,
            x,
            t,
            cond,
            guidance_scale,
        )
        scale = 1.0 - t.to(dtype=x.dtype)
        while scale.ndim < x.ndim:
            scale = scale[..., None]
        return x + scale * velocity

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
        del objective
        shape = self._require_shape(shape)
        device = torch.device(device)
        steps = self._validate_steps(
            self.num_steps if num_steps is None else num_steps
        )
        guidance_scale = validate_guidance_scale(guidance_scale)
        model.eval()
        current = self._prepare_initial_noise(shape, device, initial_noise)
        cond = self._prepare_condition(cond, shape[0], device)
        evaluation_times = torch.arange(
            steps,
            device=device,
            dtype=current.dtype,
        ) / steps
        states = [current.clone()] if return_trajectory else None
        estimate = current

        for index, scalar_time in enumerate(evaluation_times):
            t = scalar_time.expand(shape[0])
            estimate = self._boundary_prediction(
                model,
                current,
                t,
                cond,
                guidance_scale,
            )
            if not torch.isfinite(estimate).all():
                raise FloatingPointError(
                    f"Non-finite consistency sample after evaluation {index + 1}."
                )
            if states is not None:
                states.append(estimate.clone())
            if index + 1 < steps:
                next_time = evaluation_times[index + 1]
                fresh_noise = torch.randn(
                    shape,
                    device=device,
                    dtype=current.dtype,
                )
                current = (
                    (1.0 - next_time) * fresh_noise
                    + next_time * estimate
                )

        # trajectory[0] is initial noise. Later entries are data-end estimates
        # produced by evaluations at the corresponding duplicated/ascending
        # evaluation time, rather than physical x_t states.
        trajectory_times = torch.cat(
            [evaluation_times[:1], evaluation_times]
        )
        return self._format_output(estimate, states, trajectory_times)
