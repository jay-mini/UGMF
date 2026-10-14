from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

from ugm.models.adapters.lora import (
    LORA_CONFIG_NAME,
    LORA_WEIGHTS_NAME,
    LoRALinear,
    assert_only_lora_trainable,
    inject_lora,
    load_lora_adapter,
    lora_parameter_names,
    lora_parameters,
    save_lora_adapter,
)
from ugm.models.dit import DiTTiny
from ugm.objectives.flow_matching import FlowMatchingObjective


TARGETS = ("blocks.*.mlp.0", "blocks.*.mlp.2", "final_layer.projection")


def _model() -> DiTTiny:
    return DiTTiny(
        image_size=8,
        in_channels=1,
        patch_size=4,
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    )


def test_lora_injection_freezes_backbone_and_preserves_initial_output() -> None:
    torch.manual_seed(10)
    model = _model().eval()
    x = torch.randn(3, 1, 8, 8)
    t = torch.rand(3)
    before = model(x, t)

    injected = inject_lora(
        model,
        target_modules=TARGETS,
        rank=2,
        alpha=4.0,
        dropout=0.1,
    )
    model.eval()
    after = model(x, t)

    assert injected == [
        "blocks.0.mlp.0",
        "blocks.0.mlp.2",
        "blocks.1.mlp.0",
        "blocks.1.mlp.2",
        "final_layer.projection",
    ]
    assert torch.equal(before, after)
    assert sum(isinstance(module, LoRALinear) for module in model.modules()) == 5
    assert_only_lora_trainable(model)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)


def test_only_lora_parameters_update() -> None:
    torch.manual_seed(11)
    model = _model()
    inject_lora(
        model,
        target_modules=TARGETS,
        rank=2,
        alpha=2.0,
    )
    optimizer = torch.optim.AdamW(lora_parameters(model), lr=1e-2)
    objective = FlowMatchingObjective()
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = objective.compute_loss(model, {"x": torch.randn(4, 1, 8, 8)}).loss
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    assert changed
    assert changed.issubset(set(lora_parameter_names(model)))
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if name not in set(lora_parameter_names(model))
    )


def test_adapter_only_save_load_roundtrip_has_identical_output() -> None:
    torch.manual_seed(12)
    model = _model()
    base_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    inject_lora(
        model,
        target_modules=TARGETS,
        rank=2,
        alpha=4.0,
        dropout=0.05,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".lora_B"):
                parameter.normal_(mean=0.0, std=0.02)
    model.eval()
    x = torch.randn(2, 1, 8, 8)
    t = torch.tensor([0.25, 0.75])
    expected = model(x, t)

    with tempfile.TemporaryDirectory() as temporary_dir:
        adapter_dir = Path(temporary_dir) / "adapter"
        save_lora_adapter(
            model,
            adapter_dir,
            target_modules=TARGETS,
            rank=2,
            alpha=4.0,
            dropout=0.05,
            base_model={"model_type": "dit_tiny"},
        )
        metadata = json.loads(
            (adapter_dir / LORA_CONFIG_NAME).read_text(encoding="utf-8")
        )
        tensors = load_file(adapter_dir / LORA_WEIGHTS_NAME)

        reloaded = _model()
        reloaded.load_state_dict(base_state)
        loaded_metadata = load_lora_adapter(reloaded, adapter_dir)
        reloaded.eval()
        actual = reloaded(x, t)

    assert metadata == loaded_metadata
    assert set(tensors) == set(metadata["tensor_names"])
    assert tensors
    assert all(name.endswith((".lora_A", ".lora_B")) for name in tensors)
    assert torch.equal(expected, actual)
