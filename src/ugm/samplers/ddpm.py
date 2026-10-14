from __future__ import annotations

import torch
from torch import nn

from ugm.models.conditioning import guided_prediction, validate_guidance_scale
from ugm.samplers.base import BaseSampler, SampleOutput
from ugm.utils.tensor import extract


class DDPMAncestralSampler(BaseSampler):
    """
    DDPM ancestral sampler.

    Reverse process:

        X_T ~ N(0, I)

        p_{theta}(x_{t-1} | x_t) = N(mu_{theta}(x_t, t), beta_tilde_t I)

        x_t -> x_{t-1} = mu_{theta}(x_t, t) + sqrt(beta_tilde_t) * z, z ~ N(0, I)

    For epsilon-prediction models:

        mu_theta = 1 / sqrt(alpha_t) * (x_t - beta_t / sqrt(1-alpha_bar_t) * eps_theta(x_t, t))

    where:

        beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t

    This sampler assumes the training objective is DDPMObjectives.
    """

    def __init__(self, clip_x0: bool = True):
        
        self.clip_x0 = clip_x0

    def p_sample(
        self,
        model: nn.Module,
        objective: nn.Module,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Perform one reverse step:
            x_t -> x_{t-1}

        Parameters
        ----------
        model:
            Noise-prediction or x0-prediction model.

        objective:
            A DDPMObjective instance that stores betas, alpha_bars.

        xt:
            Current sample at timestep t.

        timesteps:
            Integer tensor of shape [B] indicating the current timestep for each sample.

        cond:
            Optional class labels or other conditioning.

        Returns
        -------
        x_prev:
            Sample at timestep t-1.
        """

        model_time = objective.normalize_timesteps(timesteps)

        pred = guided_prediction(
            model,
            xt,
            model_time,
            cond,
            guidance_scale,
        )

        beta_t = extract(objective.betas, timesteps, xt.shape)
        alpha_t = extract(objective.alphas, timesteps, xt.shape)
        alpha_bar_t = extract(objective.alpha_bars, timesteps, xt.shape)
        posterior_variance = extract(objective.posterior_variance, timesteps, xt.shape)

        if objective.prediction_type == "epsilon":
            eps_pred = pred
            x0_pred = (
                xt - torch.sqrt(1.0 - alpha_bar_t) * eps_pred
            ) / torch.sqrt(alpha_bar_t).clamp_min(1e-12)

        elif objective.prediction_type == "x0":
            x0_pred = pred
            eps_pred = (
                xt - torch.sqrt(alpha_bar_t) * x0_pred
            ) / torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-12)
        else:
            raise ValueError(
                f"Unsupported prediction_type: {objective.prediction_type}"
            )
        
        if self.clip_x0:
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # Make sure eps_pred is consistent with the clipped x0_pred
            eps_pred = objective.predict_eps_from_x0(
                xt=xt,
                timesteps=timesteps,
                x0_pred=x0_pred,
            )
        
        posterior_mean = (
            1.0 / torch.sqrt(alpha_t)
        ) * (
            xt - beta_t / torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-12) * eps_pred
        )

        noise = torch.randn_like(xt)

        nonzero_mask = (
            timesteps != 0
        ).float()  # No noise when t == 0

        nonzero_mask = nonzero_mask.reshape(
            xt.shape[0], *((1,) * (len(xt.shape) - 1))
        )

        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

        return x_prev

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

        if objective is None:
            raise ValueError("DDPM sampling requires a DDPM objective.")
        shape = self._require_shape(shape)

        device = torch.device(device)
        model.eval()
        objective.to(device)

        steps = self._resolve_num_steps(num_steps, objective.num_timesteps)
        guidance_scale = validate_guidance_scale(guidance_scale)
        if steps != objective.num_timesteps:
            raise ValueError(
                "DDPM ancestral sampling requires all training timesteps; "
                f"got num_steps={steps}, expected {objective.num_timesteps}. "
                "Use DDIM when a shortened schedule is required."
            )

        x = self._prepare_initial_noise(shape, device, initial_noise)
        batch_size = shape[0]
        cond = self._prepare_condition(cond, batch_size, device)
        states = [x.clone()] if return_trajectory else None

        for step in reversed(range(steps)):
            t = torch.full(
                (batch_size,),
                fill_value=step,
                dtype=torch.long,
                device=device,
            )
            x = self.p_sample(
                model,
                objective,
                x,
                t,
                cond=cond,
                guidance_scale=guidance_scale,
            )

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    f"Non-finite values encountered in sample at step {step}"
                )
            if states is not None:
                states.append(x.clone())

        trajectory_times = torch.arange(
            steps - 1,
            -2,
            -1,
            device=device,
            dtype=torch.long,
        )
        return self._format_output(x, states, trajectory_times)

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



