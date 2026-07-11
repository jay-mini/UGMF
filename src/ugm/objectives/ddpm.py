from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseObjective, LossOutput

class DDPMobjective(BaseObjective):
    """
    DDPM epsilon-prediction objective.

    Forward process:

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    Training objectives:

    L_simple = E[||epsilon - epsilon_theta(x_t, t)||^2]

    The model receives normalized continuous time t in [0, 1],
    while the diffusion schedule internally uses integer timesteps.
    """

    def __init__(self, num_timesteps: int=1000, beta_start: float=1e-4, beta_end: float=0.02, prediction_type: str="epsilon", **kwargs):
        super().__init__()

        if prediction_type not in {"epsilon", "x0"}:
            raise ValueError(f"Invalid prediction type: {prediction_type}. Must be 'epsilon' or 'x0'.")
        
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # Create the beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def compute_loss(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> LossOutput:
        x0 = batch["x"]  # shape: (B, C, H, W)
        cond = batch.get("cond", None)

        batch_size = x0.shape[0]
        device = x0.device

        t_int = torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=device,
        )

        noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars[t_int]
        alpha_bar_t = self._expand_to_data(alpha_bar_t, x0)

        xt = alpha_bar_t.sqrt() * x0 + (1 - alpha_bar_t).sqrt() * noise

        # Normalize t to [0, 1]
        t = t_int.float() / (self.num_timesteps - 1)

        pred = model(xt, t, cond=cond)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x0
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}. Must be 'epsilon' or 'x0'.")
        
        loss = F.mse_loss(pred, target)

        return LossOutput(
            loss=loss,
            metrics={
                "loss": loss.detach(),
                "loss_ddpm": loss.detach(),
            },
            aux={
                "t_int": t_int.detach(),
                "t": t.detach(),
            },
            )

    @staticmethod
    def _expand_to_data(
            values: torch.Tensor,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expand a 1D tensor of shape (B,) to match the shape of x (B, C, H, W).
        """
        while values.dim() < x.dim():
            values = values[..., None]
        return values
    
    


