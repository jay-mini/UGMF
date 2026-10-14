from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from ugm.models.adapters.diffusers_adapter import (
    ADAPTER_CONFIG_NAME,
    DiffusersUNet2DAdapter,
    create_diffusers_unet2d,
)
from ugm.models.dit import DiTTiny
from ugm.models.unet import SimpleUNet


MODEL_CONFIG_KEYS = (
    "model_type",
    "base_channels",
    "time_dim",
    "dit_patch_size",
    "dit_hidden_size",
    "dit_depth",
    "dit_num_heads",
    "dit_mlp_ratio",
    "diffusers_model_path",
    "diffusers_subfolder",
    "diffusers_timestep_scale",
    "diffusers_local_files_only",
    "diffusers_layers_per_block",
    "diffusers_block_out_channels",
    "diffusers_down_block_types",
    "diffusers_up_block_types",
    "diffusers_norm_num_groups",
    "diffusers_attention_head_dim",
    "diffusers_add_attention",
)


def resolved_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit architecture values suitable for saved configs."""

    return {
        "model_type": str(config.get("model_type", "unet")).lower(),
        "base_channels": int(config.get("base_channels", 64)),
        "time_dim": int(config.get("time_dim", 256)),
        "dit_patch_size": int(config.get("dit_patch_size", 4)),
        "dit_hidden_size": int(config.get("dit_hidden_size", 192)),
        "dit_depth": int(config.get("dit_depth", 6)),
        "dit_num_heads": int(config.get("dit_num_heads", 3)),
        "dit_mlp_ratio": float(config.get("dit_mlp_ratio", 4.0)),
        "diffusers_model_path": config.get("diffusers_model_path"),
        "diffusers_subfolder": config.get("diffusers_subfolder"),
        "diffusers_timestep_scale": config.get("diffusers_timestep_scale"),
        "diffusers_local_files_only": bool(
            config.get("diffusers_local_files_only", False)
        ),
        "diffusers_layers_per_block": int(
            config.get("diffusers_layers_per_block", 1)
        ),
        "diffusers_block_out_channels": list(
            config.get("diffusers_block_out_channels", (32, 64))
        ),
        "diffusers_down_block_types": list(
            config.get(
                "diffusers_down_block_types",
                ("DownBlock2D", "AttnDownBlock2D"),
            )
        ),
        "diffusers_up_block_types": list(
            config.get(
                "diffusers_up_block_types",
                ("AttnUpBlock2D", "UpBlock2D"),
            )
        ),
        "diffusers_norm_num_groups": int(
            config.get("diffusers_norm_num_groups", 8)
        ),
        "diffusers_attention_head_dim": int(
            config.get("diffusers_attention_head_dim", 8)
        ),
        "diffusers_add_attention": bool(
            config.get("diffusers_add_attention", True)
        ),
    }


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build a model with the common ``model(x, t, cond=None)`` interface.

    Missing ``model_type`` means ``unet`` for compatibility with checkpoints
    created before the model factory was introduced.
    """

    architecture = resolved_model_config(config)
    model_type = architecture["model_type"]
    in_channels = int(config["in_channels"])
    out_channels = int(config.get("out_channels", in_channels))
    num_classes = (
        int(config.get("num_classes", 10))
        if bool(config.get("class_cond", False))
        else None
    )

    if model_type == "unet":
        return SimpleUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=architecture["base_channels"],
            time_dim=architecture["time_dim"],
            num_classes=num_classes,
        )
    if model_type == "dit_tiny":
        return DiTTiny(
            image_size=int(config["image_size"]),
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=architecture["dit_patch_size"],
            hidden_size=architecture["dit_hidden_size"],
            depth=architecture["dit_depth"],
            num_heads=architecture["dit_num_heads"],
            mlp_ratio=architecture["dit_mlp_ratio"],
            num_classes=num_classes,
        )
    if model_type == "diffusers_unet2d":
        model_path = architecture["diffusers_model_path"]
        if model_path is None:
            return create_diffusers_unet2d({**config, **architecture})
        from pathlib import Path

        local_path = Path(str(model_path))
        if (local_path / ADAPTER_CONFIG_NAME).is_file():
            return DiffusersUNet2DAdapter.from_pretrained(
                local_path,
                local_files_only=True,
            )
        return DiffusersUNet2DAdapter.from_diffusers_pretrained(
            str(model_path),
            subfolder=architecture["diffusers_subfolder"],
            timestep_scale=architecture["diffusers_timestep_scale"],
            null_class_index=(num_classes if num_classes is not None else None),
            local_files_only=architecture["diffusers_local_files_only"],
        )
    raise ValueError(
        f"Unknown model_type {model_type!r}; expected 'unet', 'dit_tiny', "
        "or 'diffusers_unet2d'."
    )


__all__ = ["MODEL_CONFIG_KEYS", "build_model", "resolved_model_config"]
