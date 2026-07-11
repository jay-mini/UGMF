# src/ugm/samplers/ode.py

from __future__ import annotations

import torch
from torch import nn


class EulerODESampler:
    """
    Euler ODE sampler for Flow Matching / Rectified Flow.

    Given a learned velocity field:

        dx_t / dt = v_theta(x_t, t)

    We sample from noise x_0 ~ N(0, I) and integrate from t=0 to t=1:

        x_{t+dt} = x_t + dt * v_theta(x_t, t)

    This sampler can be reused later for:
        1. Flow Matching sampling;
        2. Rectified Flow sampling;
        3. generating teacher trajectories for distillation.
    """

    def __init__(
        self,
        num_steps: int = 50,
        t_start: float = 0.0,
        t_end: float = 1.0,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        self.num_steps = int(num_steps)
        self.t_start = float(t_start)
        self.t_end = float(t_end)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor | None = None,
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """
        Generate samples by integrating from Gaussian noise.

        Parameters
        ----------
        model:
            Velocity model v_theta(x, t).
        shape:
            Output tensor shape, e.g. (B, C, H, W).
        cond:
            Optional conditioning.
        device:
            Sampling device.

        Returns
        -------
        x:
            Generated samples.
        """

        device = torch.device(device)
        model.eval()

        x = torch.randn(shape, device=device)

        ts = torch.linspace(
            self.t_start,
            self.t_end,
            self.num_steps + 1,
            device=device,
        )

        for i in range(self.num_steps):
            t_scalar = ts[i]
            t_next = ts[i + 1]
            dt = t_next - t_scalar

            t = torch.full(
                size=(shape[0],),
                fill_value=float(t_scalar),
                device=device,
            )

            velocity = model(x, t, cond=cond)
            x = x + dt * velocity

        return x

    @torch.no_grad()
    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        A single Euler step from time t to time s.

        This is useful later for consistency distillation, where we often need
        teacher transitions from x_t to x_s.

        Parameters
        ----------
        x:
            Current state x_t.
        t:
            Current time, shape [B].
        s:
            Target time, shape [B].
        cond:
            Optional conditioning.

        Returns
        -------
        x_s:
            One-step Euler approximation at time s.
        """

        dt = s - t
        while dt.ndim < x.ndim:
            dt = dt[..., None]

        velocity = model(x, t, cond=cond)
        return x + dt * velocity