from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseObjective, LossOutput

class FlowMatchingObjective(BaseObjective):
    """
    Basic independent coupling Flow Matching objective.

    We sample:

    x_0 ~ N(0, I)
    x_1 ~ p_data
    t ~ Uniform(0, 1)

    Construct the linear path:
    x_t = (1 - t) * x_0 + t * x_1

    The target vector field is:

    u_t = x_1 - x_0

    The model is trained by:
    
    E_{x_0, x_1, t} [ || v_theta(x_t, t) - u_t ||^2 ]

    This abjective is also the base form used by Rectified Flow.
    The difference between vanilla FM and Rectified Flow usually lies in how the pair (x_0, x_1) is sampled.
    """

    def __init__(self, t_min: float = 0.0, t_max: float = 1.0):
        super().__init__()
        
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError("t_min and t_max should satisfy 0.0 <= t_min < t_max <= 1.0")
        
        self.t_min = t_min
        self.t_max = t_max

    def compute_loss(
            self,
            model: nn.Module,
            batch: dict[str, torch.Tensor],
    ) -> LossOutput:
        x1 = batch["x"]
        cond = batch.get("cond", None)

        batch_size = x1.shape[0]
        device = x1.device

        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, device=device)

        t_view = self._expand_to_data(t, x1)

        xt = (1 - t_view) * x0 + t_view * x1
        target_velocity = x1 - x0
        
        pred_velocity = model(xt, t, cond=cond)

        loss = F.mse_loss(pred_velocity, target_velocity, reduction="mean")

        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "loss_fm": loss.detach(),
            },
            aux={
                "t": t.detach(),
            },
        )

    @staticmethod
    def _expand_to_data(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        while values.ndim < x.ndim:
            values = values[..., None]
        return values
    

