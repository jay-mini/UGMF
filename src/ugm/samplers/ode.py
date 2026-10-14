from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ugm.models.conditioning import guided_prediction, validate_guidance_scale
from ugm.samplers.base import BaseSampler, SampleOutput


ODESolver = Literal["euler", "heun"]


class ODESampler(BaseSampler):
    """Integrate ``dx/dt = v_theta(x, t, cond)`` with Euler or Heun.

    Euler uses one velocity evaluation per step. Heun is the explicit
    trapezoidal method and uses two:

        k1 = v(x_n, t_n)
        x_predict = x_n + h * k1
        k2 = v(x_predict, t_{n+1})
        x_{n+1} = x_n + h/2 * (k1 + k2)
    """

    def __init__(
        self,
        num_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
        solver: ODESolver = "euler",
    ) -> None:
        self.num_steps = self._resolve_num_steps(num_steps, default=50)
        self.t_start = float(t_start)
        self.t_end = float(t_end)

        if not self.t_start < self.t_end:
            raise ValueError("t_start must be smaller than t_end.")
        if solver not in {"euler", "heun"}:
            raise ValueError("solver must be 'euler' or 'heun'.")
        self.solver: ODESolver = solver

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
        """Generate samples, optionally returning all ``N + 1`` states."""

        # Preserve the former ``sample(model, shape)`` positional ODE call.
        if shape is None and isinstance(objective, tuple):
            shape = objective
            objective = None
        shape = self._require_shape(shape)
        del objective  # The velocity model fully defines this ODE at sampling.
        device = torch.device(device)
        model.eval()

        steps = self._resolve_num_steps(num_steps, self.num_steps)
        guidance_scale = validate_guidance_scale(guidance_scale)
        x = self._prepare_initial_noise(shape, device, initial_noise)
        cond = self._prepare_condition(cond, shape[0], device)
        states = [x.clone()] if return_trajectory else None

        times = torch.linspace(
            self.t_start,
            self.t_end,
            steps + 1,
            device=device,
            dtype=x.dtype,
        )

        for index in range(steps):
            t_scalar = times[index]
            next_t_scalar = times[index + 1]
            t = t_scalar.expand(shape[0])
            next_t = next_t_scalar.expand(shape[0])

            if self.solver == "euler":
                x = self.euler_step(
                    model,
                    x,
                    t,
                    next_t,
                    cond=cond,
                    guidance_scale=guidance_scale,
                )
            else:
                x = self.heun_step(
                    model,
                    x,
                    t,
                    next_t,
                    cond=cond,
                    guidance_scale=guidance_scale,
                )

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    f"Non-finite ODE sample after step {index + 1}."
                )
            if states is not None:
                states.append(x.clone())

        return self._format_output(x, states, times)

    @staticmethod
    @torch.no_grad()
    def euler_step(
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        dt = ODESampler._expand_to_data(next_t - t, x)
        velocity = guided_prediction(
            model,
            x,
            t,
            cond,
            guidance_scale,
        )
        return x + dt * velocity

    @staticmethod
    @torch.no_grad()
    def heun_step(
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        next_t: torch.Tensor,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        dt = ODESampler._expand_to_data(next_t - t, x)
        velocity_at_start = guided_prediction(
            model,
            x,
            t,
            cond,
            guidance_scale,
        )
        predicted_x = x + dt * velocity_at_start
        velocity_at_end = guided_prediction(
            model,
            predicted_x,
            next_t,
            cond,
            guidance_scale,
        )
        return x + 0.5 * dt * (velocity_at_start + velocity_at_end)

    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Take one step using this sampler's configured solver."""

        if self.solver == "euler":
            return self.euler_step(
                model,
                x,
                t,
                s,
                cond=cond,
                guidance_scale=guidance_scale,
            )
        return self.heun_step(
            model,
            x,
            t,
            s,
            cond=cond,
            guidance_scale=guidance_scale,
        )

    @staticmethod
    def _expand_to_data(
        values: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        while values.ndim < x.ndim:
            values = values[..., None]
        return values


class EulerODESampler(ODESampler):
    """Backward-compatible Euler-only constructor."""

    def __init__(
        self,
        num_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> None:
        super().__init__(num_steps, t_start, t_end, solver="euler")


class HeunODESampler(ODESampler):
    """Convenience constructor for the second-order Heun solver."""

    def __init__(
        self,
        num_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ) -> None:
        super().__init__(num_steps, t_start, t_end, solver="heun")
