from __future__ import annotations

import torch
from torch import nn

class DDPMAncestralSampler:
    """
    DDPM ancestral sampler.

    Reverse process:

        X_T ~ N(0, I)

        p_{theta}(x_{t-1} | x_t) = N(mu_{theta}(x_t, t), beta_tilde_t I)

    For epsilon-prediction models:

        mu_theta = 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_bar_t) * eps_theta(x_t, t))

    where:

        beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t

    This sampler assumes the training objective is DDPMObjectives.
    """

    def __init__(self, clip_x0: bool = True):
        
        self.clip_x0 = clip_x0

    @torch.no_grad()

    def sample(
        self,
        model: nn.Module,
        objective: nn.Module,
        shape: tuple[int, ...],
        cond: torch.Tensor | None=None,
        device: torch.device | str = "cuda",
    ) -> torch.Tensor:
        """
        Generate samples from a trained DDPM model.

        Parameters
        ----------
        model:
            Noise-prediction or x0-prediction model.
        objective:
            A DDPMObjective instance that stores betas, alpha_bars.
        shape:
            Sample shape, e.g. (B, C, H, W)
        cond:
            Optional class labels or other conditioning.
        device: 
            Sampling device.

        Returns:
        --------
        x:
            Generated samples in the same normalized range as training data.


        """

        device = torch.device(device)
        model.eval()

        if not hasattr(objective, "num_timesteps"):
            raise ValueError("objective mus be a DDPMObjective-like object")
        
        num_timesteps = int(objective.num_timesteps)

        betas = objective.betas.to(device)
        alphas = objective.alphas.to(device)
        alpha_bar = objective.alpha_bar.to(device)

        x = torch.randn(shape, device=device)

        batch_size = shape[0]

        for t_index in reversed(range(num_timesteps)):
            t_int = torch.full(
                size=(batch_size, ),
                fill_value=t_index,
                device=device,
                dtype=torch.long,
            )

            t = t_int.float() / float(num_timesteps - 1)

            beta_t = betas[t_int]
            alpha_t = alphas[t_int]
            alpha_bar_t = alpha_bar[t_int]

            beta_t_view = self._expand_to_data(beta_t, x)
            alpha_t_view = self._expand_to_data(alpha_t, x)
            alpha_bar_t_view = self._expand_to_data(alpha_bar_t, x)

            pred = model(x, t, cond=cond)

            if objective.prediction_type == "epsilon":
                eps_pred = pred
                x0_pred = (
                    x - torch.sqrt(1.0 - alpha_bar_t_view) * eps_pred
                ) / torch.sqrt(alpha_bar_t_view)

            elif objective.prediction_type == "x0":
                x0_pred = pred
                eps_pred = (
                    x - torch.sqrt(alpha_bar_t_view) * x0_pred
                ) / torch.sqrt(1.0 - alpha_bar_t_view)

            else:
                raise ValueError(
                    f"Unsupported prediction_type: {objective.prediction_type}"
                )
            
            if self.clip_x0:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            if t_index == 0:
                x = x0_pred
                continue

            alpha_bar_prev = alpha_bar[t_int - 1]
            alpha_bar_prev_view = self._expand_to_data(alpha_bar_prev, x)

            posterior_variance = (
                beta_t_view
                * (1.0 - alpha_bar_prev_view)
                / (1.0 - alpha_bar_t_view)
            )

            posterior_mean = (
                1.0 / torch.sqrt(alpha_bar_t_view)
            ) * (
                x 
                - beta_t_view
                / torch.sqrt(1.0 - alpha_bar_t_view)
                * eps_pred
            )

            noise = torch.rand_like(x)

            x = posterior_mean + torch.sqrt(posterior_variance) * noise

        return x


    @staticmethod
    def _expand_to_data(
        values: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert shape [B] to [B, 1, 1, 1] for image tensors.
        """
        while values.ndim < x.ndim:
            values = values[..., None]
        return values



