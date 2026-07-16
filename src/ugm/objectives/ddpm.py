from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseObjective, LossOutput
from ugm.utils.tensor import extract

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

        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0,
        )

        posterior_variance = (
            betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
        )
        poster_variance = torch.clamp(posterior_variance, min=1e-20)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)


    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        if noise.shape != x0.shape:
            raise ValueError(
                f"Noise shape {noise.shape} must match x0 shape {x0.shape}"
            )
        
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bars, timesteps, x0.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bars, timesteps, x0.shape)

        return (sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise)
    

    def predict_x0_from_noise(self, xt: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bars, timesteps, xt.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bars, timesteps, xt.shape)

        return (
            xt - sqrt_one_minus_alpha_bar_t * noise_pred
        ) / sqrt_alpha_bar_t.clamp_min(1e-12)
    

    def normalize_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        denominator = max(self.num_timesteps - 1, 1)
        return timesteps.float() / denominator


    def compute_loss(self, model: nn.Module, batch: dict[str, torch.Tensor]) -> LossOutput:
        x0 = batch["x"]  # shape: (B, C, H, W)
        cond = batch.get("cond", None)

        batch_size = x0.shape[0]
        device = x0.device

        timesteps = torch.randint(
            low=0,
            high=self.num_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)

        # Normalize t to [0, 1]
        t = self.normalize_timesteps(timesteps)

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
                "t_int": timesteps.detach(),
                "t": t.detach(),
            },
            )
    
    def compute_loss_with_fixed_noise(self, model: nn.Module, batch: dict[str, torch.Tensor], time_steps: torch.Tensor | None = None, noise: torch.Tensor | None = None) -> LossOutput:
        x0 = batch["x"]
        cond = batch.get("cond", "None")

        batch_size = x0.shape[0]
        device = x0.device

        if time_steps is None:
            time_steps = torch.randint(
                0,
                self.num_timesteps,
                (batch_size,),
                device=x0.device,
            )
        else:
            time_steps = torch.tensor(time_steps, device=device)

        if noise is None:
            noise = torch.randn_like(x0)
        else:
            noise = torch.tensor(noise, device=device)

        xt = self.q_sample(x0, time_steps, noise)

        t = self.normalize_timesteps(time_steps)

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
                "t_int": time_steps.detach(),
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
    
    


