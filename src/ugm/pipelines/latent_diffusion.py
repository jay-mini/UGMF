from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ugm.models.adapters.vae_adapter import FrozenAutoencoderKL
from ugm.samplers.base import BaseSampler, SamplingResult


@dataclass(frozen=True)
class LatentPipelineOutput:
    images: torch.Tensor
    latents: torch.Tensor
    latent_trajectory: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None


class LatentDiffusionPipeline:
    """Compose a frozen VAE, latent generator, objective, and sampler."""

    def __init__(
        self,
        *,
        vae: FrozenAutoencoderKL,
        model: nn.Module,
        sampler: BaseSampler,
        objective: nn.Module | None = None,
    ) -> None:
        self.vae = vae.freeze()
        self.model = model
        self.sampler = sampler
        self.objective = objective

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.vae.encode(
            images,
            sample_posterior=sample_posterior,
            generator=generator,
        )

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        cond: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        initial_noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> LatentPipelineOutput:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        shape = (
            batch_size,
            self.vae.latent_channels,
            self.vae.latent_size,
            self.vae.latent_size,
        )
        sampled = self.sampler.sample(
            model=self.model,
            objective=self.objective,
            shape=shape,
            cond=cond,
            guidance_scale=guidance_scale,
            device=self.device,
            initial_noise=initial_noise,
            num_steps=num_steps,
            return_trajectory=return_trajectory,
        )
        if isinstance(sampled, SamplingResult):
            latents = sampled.samples
            trajectory = sampled.trajectory
            timesteps = sampled.timesteps
        else:
            latents = sampled
            trajectory = None
            timesteps = None
        images = self.vae.decode(latents)
        return LatentPipelineOutput(
            images=images,
            latents=latents,
            latent_trajectory=trajectory,
            timesteps=timesteps,
        )


__all__ = ["LatentDiffusionPipeline", "LatentPipelineOutput"]
