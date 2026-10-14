from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from ugm.models.adapters.diffusers_adapter import (  # noqa: E402
    ADAPTER_CONFIG_NAME,
    DiffusersUNet2DAdapter,
    create_diffusers_unet2d,
)
from ugm.models.build import build_model  # noqa: E402
from ugm.pipelines.diffusers_pipeline import (  # noqa: E402
    DiffusersComponentPipeline,
)


def _config(**overrides):
    config = {
        "model_type": "diffusers_unet2d",
        "image_size": 8,
        "in_channels": 1,
        "out_channels": 1,
        "class_cond": False,
        "num_classes": 10,
        "num_timesteps": 100,
        "diffusers_timestep_scale": 99,
        "diffusers_layers_per_block": 1,
        "diffusers_block_out_channels": [16, 32],
        "diffusers_down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
        "diffusers_up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
        "diffusers_norm_num_groups": 8,
        "diffusers_attention_head_dim": 8,
        "diffusers_add_attention": True,
    }
    config.update(overrides)
    return config


def test_adapter_matches_native_call_and_supports_backward() -> None:
    torch.manual_seed(5)
    adapter = create_diffusers_unet2d(_config())
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0.0, 0.5])

    expected = adapter.component(
        sample=x,
        timestep=t * 99,
        return_dict=True,
    ).sample
    actual = adapter(x, t)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    actual.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in adapter.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_save_reload_roundtrip_preserves_output(tmp_path: Path) -> None:
    torch.manual_seed(11)
    pipeline = DiffusersComponentPipeline(create_diffusers_unet2d(_config()))
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0.25, 0.75])
    before = pipeline(x, t)

    saved = pipeline.save_pretrained(tmp_path / "adapter")
    reloaded = DiffusersComponentPipeline.from_pretrained(saved)
    after = reloaded(x, t)

    torch.testing.assert_close(after, before, rtol=0.0, atol=2e-6)
    metadata = json.loads(
        (saved / ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
    )
    assert metadata["component_class"] == "UNet2DModel"
    assert metadata["timestep_scale"] == 99
    assert (saved / "component" / "config.json").is_file()
    assert (saved / "component" / "diffusion_pytorch_model.safetensors").is_file()


def test_conditional_adapter_maps_null_label_to_reserved_embedding() -> None:
    adapter = create_diffusers_unet2d(_config(class_cond=True, num_classes=3))
    x = torch.randn(2, 1, 8, 8)
    t = torch.rand(2)
    unconditional = adapter(x, t, cond=None)
    null_labels = adapter(x, t, cond=torch.full((2,), -1, dtype=torch.long))
    torch.testing.assert_close(unconditional, null_labels, rtol=0.0, atol=0.0)


def test_unified_builder_creates_diffusers_adapter() -> None:
    model = build_model(_config())
    assert isinstance(model, DiffusersUNet2DAdapter)
    assert model(torch.randn(2, 1, 8, 8), torch.rand(2)).shape == (2, 1, 8, 8)
