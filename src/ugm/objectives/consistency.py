from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from ugm.models.conditioning import guided_prediction, validate_guidance_scale

from .base import BaseObjective, LossOutput


class ConsistencyDistillationObjective(BaseObjective):
    """Adjacent-time consistency distillation for an FM teacher.

    Project time runs from noise at ``t=0`` to data at ``t=1``.  For an
    adjacent pair ``t_current < t_next``, the frozen teacher takes one Euler
    step toward data.  The online student at the earlier state is trained to
    match an EMA target at the later state.

    The raw student network retains the FM velocity architecture.  It is
    converted to a consistency function through the exact boundary
    parameterization

        f(x, t) = x + (1 - t) F(x, t),

    so ``f(x, 1) = x`` independently of network parameters.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        *,
        num_time_steps: int = 20,
        target_decay: float = 0.999,
        target_update_every: int = 1,
        teacher_guidance_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_time_steps <= 0:
            raise ValueError("num_time_steps must be positive.")
        if not 0.0 <= target_decay < 1.0:
            raise ValueError("target_decay must satisfy 0 <= decay < 1.")
        if target_update_every <= 0:
            raise ValueError("target_update_every must be positive.")

        # The teacher is intentionally not registered as a child module. Its
        # immutable weights remain in the source checkpoint instead of being
        # duplicated in every distillation checkpoint.
        object.__setattr__(self, "teacher", teacher)
        self.target = copy.deepcopy(student)
        self.num_time_steps = int(num_time_steps)
        self.target_decay = float(target_decay)
        self.target_update_every = int(target_update_every)
        self.teacher_guidance_scale = validate_guidance_scale(
            teacher_guidance_scale
        )
        self.register_buffer(
            "num_target_updates",
            torch.zeros((), dtype=torch.long),
        )

        self._freeze_model(self.teacher)
        self._freeze_model(self.target)

    @staticmethod
    def _freeze_model(model: nn.Module) -> None:
        model.requires_grad_(False)
        model.eval()

    def train(self, mode: bool = True):
        # BaseTrainer calls objective.train(). The teacher and EMA target must
        # remain deterministic and frozen even while the objective is active.
        super().train(mode)
        self.teacher.eval()
        self.target.eval()
        return self

    @staticmethod
    def _expand_to_data(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while values.ndim < x.ndim:
            values = values[..., None]
        return values

    @classmethod
    def boundary_prediction(
        cls,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``x + (1-t)F(x,t)`` with an exact data-end boundary."""

        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [batch_size].")
        if torch.any(t < 0.0) or torch.any(t > 1.0):
            raise ValueError("Consistency times must lie in [0, 1].")
        scale = cls._expand_to_data(1.0 - t.to(dtype=x.dtype), x)
        return x + scale * model(x, t, cond=cond)

    @classmethod
    @torch.no_grad()
    def teacher_euler_step(
        cls,
        teacher: nn.Module,
        x_current: torch.Tensor,
        t_current: torch.Tensor,
        t_next: torch.Tensor,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Advance from a noisier time to the adjacent cleaner time."""

        if not torch.all(t_next > t_current):
            raise ValueError(
                "FM consistency distillation requires t_next > t_current "
                "because time runs from noise (0) to data (1)."
            )
        dt = cls._expand_to_data(t_next - t_current, x_current)
        velocity = guided_prediction(
            teacher,
            x_current,
            t_current,
            cond,
            guidance_scale,
        )
        return x_current + dt * velocity

    def compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> LossOutput:
        x_data = batch["x"]
        noise = torch.randn_like(x_data)
        indices = torch.randint(
            low=0,
            high=self.num_time_steps,
            size=(x_data.shape[0],),
            device=x_data.device,
        )
        return self.compute_loss_with_fixed_inputs(
            model=model,
            batch=batch,
            noise=noise,
            time_indices=indices,
        )

    def compute_loss_with_fixed_inputs(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        time_indices: torch.Tensor,
    ) -> LossOutput:
        x_data = batch["x"]
        cond = batch.get("cond")
        if noise.shape != x_data.shape:
            raise ValueError("noise and batch['x'] must have identical shapes.")
        if not torch.isfinite(noise).all() or not torch.isfinite(x_data).all():
            raise ValueError("Distillation inputs contain non-finite values.")
        if time_indices.shape != (x_data.shape[0],):
            raise ValueError("time_indices must have shape [batch_size].")
        if time_indices.is_floating_point() or time_indices.is_complex():
            raise TypeError("time_indices must use an integer dtype.")
        if torch.any(time_indices < 0) or torch.any(
            time_indices >= self.num_time_steps
        ):
            raise ValueError(
                f"time_indices must lie in [0, {self.num_time_steps - 1}]."
            )

        dtype = x_data.dtype
        t_current = time_indices.to(dtype=dtype) / self.num_time_steps
        t_next = (time_indices.to(dtype=dtype) + 1) / self.num_time_steps
        t_view = self._expand_to_data(t_current, x_data)
        x_current = (1.0 - t_view) * noise + t_view * x_data

        with torch.no_grad():
            x_next = self.teacher_euler_step(
                self.teacher,
                x_current,
                t_current,
                t_next,
                cond=cond,
                guidance_scale=self.teacher_guidance_scale,
            )
            target_prediction = self.boundary_prediction(
                self.target,
                x_next,
                t_next,
                cond=cond,
            )

        student_prediction = self.boundary_prediction(
            model,
            x_current,
            t_current,
            cond=cond,
        )
        loss = F.mse_loss(
            student_prediction,
            target_prediction.detach(),
            reduction="mean",
        )

        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "loss_consistency": loss.detach(),
                "t_current_mean": t_current.detach().mean(),
                "t_next_mean": t_next.detach().mean(),
            },
            aux={
                "t_current": t_current.detach(),
                "t_next": t_next.detach(),
            },
        )

    @torch.no_grad()
    def update_target(self, student: nn.Module, global_step: int) -> bool:
        """EMA-update the target after a successful student optimizer step."""

        if global_step <= 0:
            raise ValueError("global_step must be positive after an update.")
        if global_step % self.target_update_every != 0:
            return False

        student_parameters = dict(student.named_parameters())
        for name, target_parameter in self.target.named_parameters():
            target_parameter.lerp_(
                student_parameters[name].detach(),
                1.0 - self.target_decay,
            )
        student_buffers = dict(student.named_buffers())
        for name, target_buffer in self.target.named_buffers():
            target_buffer.copy_(student_buffers[name])
        self.num_target_updates.add_(1)
        self.target.eval()
        return True
