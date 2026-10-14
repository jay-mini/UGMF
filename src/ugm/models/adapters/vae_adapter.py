from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _autoencoder_kl_class():
    try:
        from diffusers import AutoencoderKL
    except ImportError as error:
        raise ImportError(
            "VAE latent diffusion requires the optional Diffusers dependency. "
            "Install the project with `pip install -e .[diffusers]`."
        ) from error
    return AutoencoderKL


class FrozenAutoencoderKL(nn.Module):
    """Frozen Diffusers ``AutoencoderKL`` with explicit latent scaling.

    Diffusers VAE checkpoints define the scale used by their diffusion model:

        z_model = scaling_factor * z_vae
        x_hat   = decode(z_model / scaling_factor)

    The adapter also owns the deterministic image-shape/channel conversion so
    the exact inverse operation is used at decode time.  This is important for
    MNIST: a 1x28x28 image is repeated to RGB and padded to 3x32x32 before the
    standard Stable-Diffusion VAE is called.
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        image_channels: int,
        image_size: int,
        vae_image_size: int | None = None,
        resize_mode: str = "pad",
        scaling_factor: float | None = None,
    ) -> None:
        super().__init__()
        if vae.__class__.__name__ != "AutoencoderKL":
            raise TypeError(
                "FrozenAutoencoderKL requires diffusers.AutoencoderKL, got "
                f"{vae.__class__.__name__}."
            )
        if image_channels <= 0 or image_size <= 0:
            raise ValueError("image_channels and image_size must be positive.")
        if resize_mode not in {"pad", "resize"}:
            raise ValueError("resize_mode must be 'pad' or 'resize'.")

        config = vae.config
        vae_input_channels = int(getattr(config, "in_channels"))
        if image_channels not in {vae_input_channels, 1}:
            raise ValueError(
                "Input channels must match the VAE, except grayscale input can "
                "be repeated for an RGB VAE."
            )
        if image_channels == 1 and vae_input_channels not in {1, 3}:
            raise ValueError("Grayscale conversion only supports 1- or 3-channel VAEs.")

        block_channels = tuple(int(v) for v in config.block_out_channels)
        downsample_factor = 2 ** (len(block_channels) - 1)
        resolved_vae_size = int(vae_image_size or image_size)
        if resolved_vae_size <= 0 or resolved_vae_size % downsample_factor != 0:
            raise ValueError(
                "vae_image_size must be positive and divisible by the VAE "
                f"downsample factor ({downsample_factor})."
            )
        if resize_mode == "pad" and resolved_vae_size < image_size:
            raise ValueError("pad mode requires vae_image_size >= image_size.")

        resolved_scale = (
            getattr(config, "scaling_factor", None)
            if scaling_factor is None
            else scaling_factor
        )
        if resolved_scale is None:
            raise ValueError(
                "The VAE has no scaling_factor; provide one explicitly."
            )
        resolved_scale = float(resolved_scale)
        if not torch.isfinite(torch.tensor(resolved_scale)) or resolved_scale <= 0:
            raise ValueError("scaling_factor must be finite and positive.")

        self.vae = vae
        self.image_channels = int(image_channels)
        self.image_size = int(image_size)
        self.vae_image_size = resolved_vae_size
        self.resize_mode = resize_mode
        self.scaling_factor = resolved_scale
        self.vae_input_channels = vae_input_channels
        self.latent_channels = int(config.latent_channels)
        self.downsample_factor = downsample_factor
        self.latent_size = resolved_vae_size // downsample_factor

        difference = resolved_vae_size - image_size
        self._padding = (
            difference // 2,
            difference - difference // 2,
            difference // 2,
            difference - difference // 2,
        )
        self.freeze()

    def freeze(self) -> "FrozenAutoencoderKL":
        self.vae.requires_grad_(False)
        super().train(False)
        return self

    def train(self, mode: bool = True) -> "FrozenAutoencoderKL":
        # A frozen feature extractor must not silently re-enable dropout or
        # training-time normalization if its parent pipeline enters train mode.
        del mode
        super().train(False)
        self.vae.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.vae.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.image_channels, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                "images must have shape [B, C, H, W] with C,H,W="
                f"{expected}; got {tuple(images.shape)}."
            )
        if not images.is_floating_point():
            raise TypeError("images must be floating point and normalized to [-1, 1].")
        images = images.to(device=self.device, dtype=self.dtype)
        if self.image_channels == 1 and self.vae_input_channels == 3:
            images = images.repeat(1, 3, 1, 1)
        if self.vae_image_size != self.image_size:
            if self.resize_mode == "pad":
                images = F.pad(images, self._padding, value=-1.0)
            else:
                images = F.interpolate(
                    images,
                    size=(self.vae_image_size, self.vae_image_size),
                    mode="bilinear",
                    align_corners=False,
                )
        return images

    def _restore_images(self, images: torch.Tensor) -> torch.Tensor:
        if self.vae_image_size != self.image_size:
            if self.resize_mode == "pad":
                left, right, top, bottom = self._padding
                height_end = images.shape[-2] - bottom if bottom else images.shape[-2]
                width_end = images.shape[-1] - right if right else images.shape[-1]
                images = images[..., top:height_end, left:width_end]
            else:
                images = F.interpolate(
                    images,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
        if self.image_channels == 1 and images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        return images.clamp(-1.0, 1.0)

    @torch.no_grad()
    def encode(
        self,
        images: torch.Tensor,
        *,
        sample_posterior: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prepared = self._prepare_images(images)
        posterior = self.vae.encode(prepared, return_dict=True).latent_dist
        latent = (
            posterior.sample(generator=generator)
            if sample_posterior
            else posterior.mode()
        )
        return (latent * self.scaling_factor).detach()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_channels, self.latent_size, self.latent_size)
        if latents.ndim != 4 or tuple(latents.shape[1:]) != expected:
            raise ValueError(
                "latents must have shape [B, C, H, W] with C,H,W="
                f"{expected}; got {tuple(latents.shape)}."
            )
        latents = latents.to(device=self.device, dtype=self.dtype)
        decoded = self.vae.decode(
            latents / self.scaling_factor,
            return_dict=True,
        ).sample
        return self._restore_images(decoded)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        image_channels: int,
        image_size: int,
        vae_image_size: int | None = None,
        resize_mode: str = "pad",
        scaling_factor: float | None = None,
        subfolder: str | None = None,
        local_files_only: bool = False,
        **load_kwargs: Any,
    ) -> "FrozenAutoencoderKL":
        AutoencoderKL = _autoencoder_kl_class()
        kwargs = dict(load_kwargs)
        kwargs.setdefault("low_cpu_mem_usage", False)
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(
            vae,
            image_channels=image_channels,
            image_size=image_size,
            vae_image_size=vae_image_size,
            resize_mode=resize_mode,
            scaling_factor=scaling_factor,
        )


__all__ = ["FrozenAutoencoderKL"]
