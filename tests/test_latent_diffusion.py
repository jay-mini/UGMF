from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ugm.models.adapters.vae_adapter import FrozenAutoencoderKL
from ugm.models.unet import SimpleUNet
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.pipelines.latent_diffusion import LatentDiffusionPipeline
from ugm.samplers.ode import ODESampler
from ugm.trainers.latent_trainer import LatentDiffusionTrainer


diffusers = pytest.importorskip("diffusers")


def _tiny_autoencoder():
    return diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=16,
        scaling_factor=0.5,
    )


def _adapter() -> FrozenAutoencoderKL:
    return FrozenAutoencoderKL(
        _tiny_autoencoder(),
        image_channels=1,
        image_size=8,
        vae_image_size=16,
        resize_mode="pad",
    )


def test_vae_encode_scale_decode_and_local_pretrained_load() -> None:
    torch.manual_seed(0)
    images = torch.randn(2, 1, 8, 8).clamp(-1, 1)
    with tempfile.TemporaryDirectory() as temporary_dir:
        model_dir = Path(temporary_dir) / "vae"
        _tiny_autoencoder().save_pretrained(model_dir, safe_serialization=True)
        adapter = FrozenAutoencoderKL.from_pretrained(
            model_dir,
            image_channels=1,
            image_size=8,
            vae_image_size=16,
            resize_mode="pad",
            local_files_only=True,
        )

        prepared = adapter._prepare_images(images)
        raw_mode = adapter.vae.encode(prepared).latent_dist.mode()
        scaled = adapter.encode(images, sample_posterior=False)
        decoded = adapter.decode(scaled)

    assert scaled.shape == (2, 4, 8, 8)
    assert torch.allclose(scaled, raw_mode * 0.5)
    assert decoded.shape == images.shape
    assert torch.isfinite(decoded).all()
    assert adapter.scaling_factor == pytest.approx(0.5)
    assert not adapter.training
    assert not adapter.vae.training
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_latent_training_updates_generator_but_not_frozen_vae() -> None:
    torch.manual_seed(1)
    vae = _adapter()
    model = SimpleUNet(
        in_channels=4,
        out_channels=4,
        base_channels=8,
        time_dim=32,
    )
    objective = FlowMatchingObjective()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(
        TensorDataset(
            torch.randn(4, 1, 8, 8).clamp(-1, 1),
            torch.arange(4, dtype=torch.long),
        ),
        batch_size=4,
    )
    with tempfile.TemporaryDirectory() as temporary_dir:
        trainer = LatentDiffusionTrainer(
            model=model,
            objective=objective,
            optimizer=optimizer,
            train_loader=loader,
            vae=vae,
            sample_posterior=False,
            device="cpu",
            output_dir=temporary_dir,
            mixed_precision="no",
            log_every=100,
            save_every=100,
        )
        batch = trainer._prepare_batch(next(iter(loader)))
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        metrics = trainer._train_step(batch)

    assert batch["x"].shape == (4, 4, 8, 8)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert trainer.global_step == 1
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    assert all(parameter.grad is None for parameter in vae.parameters())
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_parameter_ids.isdisjoint(id(p) for p in vae.parameters())


class _ZeroVelocity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, cond=None):
        del t, cond
        return torch.zeros_like(x) + self.anchor * 0.0


def test_latent_pipeline_fixed_noise_sampling_and_decode() -> None:
    vae = _adapter()
    model = _ZeroVelocity()
    noise = torch.randn(2, 4, 8, 8)
    pipeline = LatentDiffusionPipeline(
        vae=vae,
        model=model,
        sampler=ODESampler(num_steps=2, solver="heun"),
        objective=None,
    )
    output = pipeline.sample(
        batch_size=2,
        initial_noise=noise,
        num_steps=2,
        return_trajectory=True,
    )

    assert torch.equal(output.latents, noise)
    assert output.images.shape == (2, 1, 8, 8)
    assert output.latent_trajectory is not None
    assert output.latent_trajectory.shape == (3, 2, 4, 8, 8)
    assert output.timesteps is not None
    assert output.timesteps.shape == (3,)
    assert torch.isfinite(output.images).all()
