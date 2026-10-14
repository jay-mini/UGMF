from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseObjective, LossOutput


class RectifiedFlowObjective(BaseObjective):
    """Flow matching on teacher-induced endpoint pairs.

    A paired dataset supplies ``x0`` (the original Gaussian noise) and ``x1``
    (the sample obtained by integrating a teacher flow from that exact
    noise).  The student is trained on the straight interpolation

        x_t = (1 - t) x0 + t x1,
        u_t = x1 - x0.

    Unlike ordinary independent-coupling flow matching, this objective must
    never resample ``x0``: preserving the teacher coupling is the defining
    part of the reflow/rectification step.
    """

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= t_min < t_max <= 1.0:
            raise ValueError(
                "t_min and t_max must satisfy 0 <= t_min < t_max <= 1."
            )
        self.t_min = float(t_min)
        self.t_max = float(t_max)

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> LossOutput:
        x0, x1, cond = self._validate_batch(batch)
        t = self.t_min + (self.t_max - self.t_min) * torch.rand(
            x0.shape[0],
            device=x0.device,
            dtype=x0.dtype,
        )
        return self.compute_loss_with_fixed_time(model, batch, t)

    def compute_loss_with_fixed_time(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        t: torch.Tensor,
    ) -> LossOutput:
        """Compute the RF loss at caller-provided times for deterministic tests."""

        x0, x1, cond = self._validate_batch(batch)
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError("t must have shape [batch_size].")
        if not torch.isfinite(t).all():
            raise ValueError("t contains non-finite values.")
        if torch.any(t < self.t_min) or torch.any(t > self.t_max):
            raise ValueError(
                f"All times must satisfy {self.t_min} <= t <= {self.t_max}."
            )

        t = t.to(device=x0.device, dtype=x0.dtype)
        t_view = self._expand_to_data(t, x0)
        xt = (1.0 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        predicted_velocity = model(xt, t, cond=cond)
        loss = F.mse_loss(
            predicted_velocity,
            target_velocity,
            reduction="mean",
        )

        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "loss_rf": loss.detach(),
            },
            aux={"t": t.detach()},
        )

    @staticmethod
    def _validate_batch(
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if "x0" not in batch or "x1" not in batch:
            raise KeyError(
                "RectifiedFlowObjective requires paired 'x0' and 'x1' tensors."
            )
        x0 = batch["x0"]
        x1 = batch["x1"]
        cond = batch.get("cond")
        if x0.shape != x1.shape:
            raise ValueError(
                f"x0 and x1 must have identical shapes, got {x0.shape} and {x1.shape}."
            )
        if x0.ndim < 2 or x0.shape[0] == 0:
            raise ValueError("Paired tensors must have a non-empty batch dimension.")
        if not x0.is_floating_point() or not x1.is_floating_point():
            raise TypeError("x0 and x1 must be floating-point tensors.")
        if not torch.isfinite(x0).all() or not torch.isfinite(x1).all():
            raise ValueError("Paired tensors contain non-finite values.")
        if cond is not None and (cond.ndim != 1 or cond.shape[0] != x0.shape[0]):
            raise ValueError("cond must have shape [batch_size].")
        return x0, x1, cond

    @staticmethod
    def _expand_to_data(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while values.ndim < x.ndim:
            values = values[..., None]
        return values
