from __future__ import annotations

from typing import Any

import torch

from ugm.models.adapters.vae_adapter import FrozenAutoencoderKL
from ugm.trainers.base_trainer import BaseTrainer


class LatentDiffusionTrainer(BaseTrainer):
    """BaseTrainer variant that encodes pixel batches with a frozen VAE."""

    def __init__(
        self,
        *args: Any,
        vae: FrozenAutoencoderKL,
        sample_posterior: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vae = vae.to(self.device).freeze()
        self.sample_posterior = bool(sample_posterior)
        self.last_metrics: dict[str, float] = {}
        self.last_metrics: dict[str, float] = {}

    def _prepare_batch(self, raw_batch: Any) -> dict[str, Any]:
        batch = super()._prepare_batch(raw_batch)
        images = batch["x"]
        if not torch.is_tensor(images):
            raise TypeError("Latent training expects a tensor in batch['x'].")
        batch["x"] = self.vae.encode(
            images,
            sample_posterior=self.sample_posterior,
        )
        return batch

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        metrics = super()._train_step(batch)
        self.last_metrics = dict(metrics)
        return metrics

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        metrics = super()._train_step(batch)
        self.last_metrics = dict(metrics)
        return metrics


__all__ = ["LatentDiffusionTrainer"]
