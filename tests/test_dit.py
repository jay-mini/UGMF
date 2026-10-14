from __future__ import annotations

import torch

from ugm.models.build import build_model
from ugm.models.dit import DiTTiny, PatchEmbed
from ugm.models.unet import SimpleUNet
from ugm.objectives.flow_matching import FlowMatchingObjective


def _dit_config(**overrides):
    config = {
        "model_type": "dit_tiny",
        "image_size": 28,
        "in_channels": 1,
        "out_channels": 1,
        "class_cond": False,
        "num_classes": 10,
        "dit_patch_size": 4,
        "dit_hidden_size": 32,
        "dit_depth": 2,
        "dit_num_heads": 4,
        "dit_mlp_ratio": 2.0,
    }
    config.update(overrides)
    return config


def test_patch_embedding_and_unpatchified_output_shapes() -> None:
    patch_embed = PatchEmbed(in_channels=1, hidden_size=32, patch_size=4)
    x = torch.randn(3, 1, 28, 28)
    tokens = patch_embed(x)
    assert tokens.shape == (3, 49, 32)

    model = DiTTiny(
        image_size=28,
        in_channels=1,
        patch_size=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )
    output = model(x, torch.tensor([0.0, 0.5, 1.0]))
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert model.position_embedding.shape == (1, 49, 32)


def test_dit_forward_backward_and_parameter_update() -> None:
    torch.manual_seed(7)
    model = build_model(_dit_config())
    objective = FlowMatchingObjective()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    # Two steps exercise the progressive gradient path created by zero-init:
    # step one updates the output head; step two reaches patch/token features.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = objective.compute_loss(model, {"x": torch.randn(4, 1, 28, 28)})
        assert torch.isfinite(output.loss)
        output.loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()

    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), before[name])
    ]
    assert "final_layer.projection.weight" in changed
    assert "patch_embed.projection.weight" in changed


def test_dit_class_null_condition_matches_unconditional_branch() -> None:
    model = build_model(_dit_config(class_cond=True, num_classes=10))
    t = torch.rand(3)
    unconditional = model._conditioning(t, None, batch_size=3)
    null_labels = model._conditioning(
        t,
        torch.full((3,), -1, dtype=torch.long),
        batch_size=3,
    )
    assert torch.equal(unconditional, null_labels)


def test_builder_switches_unet_and_dit_with_the_same_interface() -> None:
    common = {
        "image_size": 28,
        "in_channels": 1,
        "out_channels": 1,
        "class_cond": False,
        "num_classes": 10,
        "base_channels": 8,
        "time_dim": 32,
    }
    unet = build_model({**common, "model_type": "unet"})
    dit = build_model({**common, **_dit_config()})
    assert isinstance(unet, SimpleUNet)
    assert isinstance(dit, DiTTiny)

    x = torch.randn(2, 1, 28, 28)
    t = torch.rand(2)
    assert unet(x, t).shape == x.shape
    assert dit(x, t).shape == x.shape
    # The common trainer keeps dataset labels in the batch even for an
    # unconditional run; both model families must safely ignore them.
    assert dit(x, t, cond=torch.tensor([1, 2])).shape == x.shape


def test_dit_rejects_incompatible_patch_grid() -> None:
    try:
        DiTTiny(image_size=28, patch_size=6, hidden_size=32, depth=1, num_heads=4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("Expected an incompatible patch grid to be rejected.")
