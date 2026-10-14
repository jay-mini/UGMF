from __future__ import annotations

import fnmatch
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


LORA_CONFIG_NAME = "adapter_config.json"
LORA_WEIGHTS_NAME = "adapter_model.safetensors"
LORA_FORMAT = "ugmf.lora_linear"
LORA_VERSION = 1


def _safetensors_functions():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise ImportError(
            "LoRA adapter serialization requires safetensors. Install the "
            "project with `pip install -e .[diffusers]`."
        ) from error
    return load_file, save_file


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual.

    ``B`` starts at zero, so injection preserves the backbone's output:

        y = W x + b + (alpha / rank) B A x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear modules.")
        if rank <= 0 or rank > min(base_layer.in_features, base_layer.out_features):
            raise ValueError(
                "rank must satisfy 0 < rank <= min(in_features, out_features)."
            )
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base_layer.in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        low_rank = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + low_rank * self.scaling


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _set_child(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def inject_lora(
    model: nn.Module,
    *,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Freeze a model and replace matching ``nn.Linear`` modules with LoRA."""

    patterns = tuple(str(pattern) for pattern in target_modules)
    if not patterns:
        raise ValueError("target_modules must contain at least one glob pattern.")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("The model already contains LoRA modules.")

    candidates = [
        name
        for name, module in model.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    ]
    if not candidates:
        raise ValueError(
            "No nn.Linear modules matched target patterns: "
            f"{list(patterns)}."
        )

    model.requires_grad_(False)
    for name in candidates:
        parent, child_name = _parent_and_child(model, name)
        original = parent[int(child_name)] if child_name.isdigit() else getattr(parent, child_name)
        _set_child(
            parent,
            child_name,
            LoRALinear(
                original,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
    return candidates


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, _ in model.named_parameters()
        if name.endswith(".lora_A") or name.endswith(".lora_B")
    ]


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    names = set(lora_parameter_names(model))
    return [parameter for name, parameter in model.named_parameters() if name in names]


def assert_only_lora_trainable(model: nn.Module) -> None:
    lora_names = set(lora_parameter_names(model))
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not lora_names:
        raise RuntimeError("The model contains no LoRA parameters.")
    if trainable_names != lora_names:
        raise RuntimeError(
            "Only LoRA parameters may be trainable; "
            f"expected={sorted(lora_names)}, actual={sorted(trainable_names)}."
        )


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    names = set(lora_parameter_names(model))
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if name in names
    }


def save_lora_adapter(
    model: nn.Module,
    save_directory: str | Path,
    *,
    target_modules: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
    base_model: Mapping[str, Any] | None = None,
) -> Path:
    """Save adapter tensors only plus reconstruction metadata."""

    assert_only_lora_trainable(model)
    _, save_file = _safetensors_functions()
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    weights_path = save_directory / LORA_WEIGHTS_NAME
    temporary_weights = weights_path.with_suffix(weights_path.suffix + ".tmp")
    state = lora_state_dict(model)
    save_file(state, str(temporary_weights))
    os.replace(temporary_weights, weights_path)

    metadata = {
        "format": LORA_FORMAT,
        "version": LORA_VERSION,
        "adapter_type": "lora",
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_modules": list(target_modules),
        "injected_modules": [
            name for name, module in model.named_modules() if isinstance(module, LoRALinear)
        ],
        "tensor_names": sorted(state),
        "base_model": dict(base_model or {}),
    }
    config_path = save_directory / LORA_CONFIG_NAME
    temporary_config = config_path.with_suffix(config_path.suffix + ".tmp")
    with temporary_config.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    os.replace(temporary_config, config_path)
    return save_directory


def load_lora_adapter(
    model: nn.Module,
    adapter_directory: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Inject and load a saved adapter into an unwrapped base model."""

    load_file, _ = _safetensors_functions()
    adapter_directory = Path(adapter_directory)
    with (adapter_directory / LORA_CONFIG_NAME).open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if metadata.get("format") != LORA_FORMAT:
        raise ValueError("Unsupported LoRA adapter format.")
    if int(metadata.get("version", -1)) != LORA_VERSION:
        raise ValueError("Unsupported LoRA adapter version.")

    injected = inject_lora(
        model,
        target_modules=metadata["target_modules"],
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        dropout=float(metadata["dropout"]),
    )
    if strict and injected != metadata.get("injected_modules"):
        raise RuntimeError(
            "Adapter target modules do not match the base model: "
            f"saved={metadata.get('injected_modules')}, current={injected}."
        )
    state = load_file(str(adapter_directory / LORA_WEIGHTS_NAME), device="cpu")
    current = dict(model.named_parameters())
    expected_names = set(lora_parameter_names(model))
    if strict and set(state) != expected_names:
        raise RuntimeError(
            "Adapter tensor names do not match injected LoRA parameters: "
            f"saved={sorted(state)}, current={sorted(expected_names)}."
        )
    with torch.no_grad():
        for name, value in state.items():
            if name not in current:
                if strict:
                    raise RuntimeError(f"Unexpected adapter tensor: {name}.")
                continue
            parameter = current[name]
            if tuple(parameter.shape) != tuple(value.shape):
                raise RuntimeError(
                    f"Shape mismatch for {name}: saved={tuple(value.shape)}, "
                    f"current={tuple(parameter.shape)}."
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    assert_only_lora_trainable(model)
    return metadata


__all__ = [
    "LORA_CONFIG_NAME",
    "LORA_WEIGHTS_NAME",
    "LoRALinear",
    "assert_only_lora_trainable",
    "inject_lora",
    "load_lora_adapter",
    "lora_parameter_names",
    "lora_parameters",
    "lora_state_dict",
    "save_lora_adapter",
]
