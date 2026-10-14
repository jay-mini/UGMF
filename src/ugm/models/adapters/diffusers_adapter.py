from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ugm.models.conditioning import UNCONDITIONAL_CLASS_LABEL


ADAPTER_CONFIG_NAME = "ugmf_adapter_config.json"
ADAPTER_FORMAT = "ugmf.diffusers_unet2d_adapter"
ADAPTER_VERSION = 1


def _diffusers_unet2d_class():
    try:
        from diffusers import UNet2DModel
    except ImportError as error:
        raise ImportError(
            "Diffusers integration requires the optional dependency. Install "
            "the project with `pip install -e .[diffusers]`."
        ) from error
    return UNet2DModel


def create_diffusers_unet2d(config: Mapping[str, Any]) -> nn.Module:
    """Create a small UNet2DModel from an explicit UGMF configuration."""

    UNet2DModel = _diffusers_unet2d_class()
    in_channels = int(config["in_channels"])
    out_channels = int(config.get("out_channels", in_channels))
    image_size = int(config["image_size"])
    block_out_channels = tuple(
        int(value)
        for value in config.get("diffusers_block_out_channels", (32, 64))
    )
    down_block_types = tuple(
        str(value)
        for value in config.get(
            "diffusers_down_block_types",
            ("DownBlock2D", "AttnDownBlock2D"),
        )
    )
    up_block_types = tuple(
        str(value)
        for value in config.get(
            "diffusers_up_block_types",
            ("AttnUpBlock2D", "UpBlock2D"),
        )
    )
    if not (
        len(block_out_channels) == len(down_block_types) == len(up_block_types)
    ):
        raise ValueError(
            "Diffusers block_out_channels, down_block_types, and "
            "up_block_types must have the same length."
        )

    class_cond = bool(config.get("class_cond", False))
    num_classes = int(config.get("num_classes", 10)) if class_cond else None
    null_class_index = num_classes if class_cond else None
    num_class_embeds = num_classes + 1 if class_cond else None
    component = UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=int(config.get("diffusers_layers_per_block", 1)),
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        norm_num_groups=int(config.get("diffusers_norm_num_groups", 8)),
        attention_head_dim=int(config.get("diffusers_attention_head_dim", 8)),
        add_attention=bool(config.get("diffusers_add_attention", True)),
        num_class_embeds=num_class_embeds,
        num_train_timesteps=int(config.get("num_timesteps", 1000)),
    )
    return DiffusersUNet2DAdapter(
        component,
        timestep_scale=config.get("diffusers_timestep_scale"),
        null_class_index=null_class_index,
    )


class DiffusersUNet2DAdapter(nn.Module):
    """Expose a Diffusers UNet2DModel through ``model(x, t, cond=None)``.

    UGMF objectives use normalized continuous time. Diffusers UNet2DModel
    commonly uses schedule indices, so the adapter multiplies normalized time
    by ``timestep_scale`` before forwarding it to the wrapped component.
    """

    def __init__(
        self,
        component: nn.Module,
        *,
        timestep_scale: float | None = None,
        null_class_index: int | None = None,
    ) -> None:
        super().__init__()
        if component.__class__.__name__ != "UNet2DModel":
            raise TypeError(
                "DiffusersUNet2DAdapter currently supports UNet2DModel, got "
                f"{component.__class__.__name__}."
            )
        config = component.config
        if timestep_scale is None:
            num_train_timesteps = getattr(config, "num_train_timesteps", None)
            timestep_scale = (
                max(int(num_train_timesteps) - 1, 1)
                if num_train_timesteps is not None
                else 1.0
            )
        timestep_scale = float(timestep_scale)
        if not torch.isfinite(torch.tensor(timestep_scale)) or timestep_scale <= 0:
            raise ValueError("timestep_scale must be finite and positive.")
        if null_class_index is not None and int(null_class_index) < 0:
            raise ValueError("null_class_index must be non-negative when provided.")

        self.component = component
        self.timestep_scale = timestep_scale
        self.null_class_index = (
            None if null_class_index is None else int(null_class_index)
        )

    @property
    def component_config(self) -> Any:
        return self.component.config

    def _class_labels(
        self,
        cond: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        num_class_embeds = getattr(self.component.config, "num_class_embeds", None)
        if num_class_embeds is None:
            return None
        if cond is None:
            if self.null_class_index is None:
                raise ValueError(
                    "The wrapped conditional component requires class labels, "
                    "but no null_class_index is configured."
                )
            return torch.full(
                (batch_size,),
                self.null_class_index,
                device=device,
                dtype=torch.long,
            )
        if cond.ndim != 1 or cond.shape[0] != batch_size:
            raise ValueError("Class conditions must have shape [batch_size].")
        if cond.is_floating_point() or cond.is_complex():
            raise TypeError("Class conditions must use an integer tensor dtype.")

        labels = cond.to(device=device, dtype=torch.long)
        null_mask = labels == UNCONDITIONAL_CLASS_LABEL
        if torch.any(null_mask):
            if self.null_class_index is None:
                raise ValueError("Received null labels without null_class_index.")
            labels = labels.masked_fill(null_mask, self.null_class_index)
        if torch.any(labels < 0) or torch.any(labels >= int(num_class_embeds)):
            raise ValueError(
                f"Class labels must lie in [0, {int(num_class_embeds) - 1}]."
            )
        return labels

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError("t must have shape [batch_size].")
        if not torch.isfinite(t).all():
            raise ValueError("t contains non-finite values.")
        component_time = t.to(device=x.device, dtype=torch.float32)
        component_time = component_time * self.timestep_scale
        class_labels = self._class_labels(cond, x.shape[0], x.device)
        output = self.component(
            sample=x,
            timestep=component_time,
            class_labels=class_labels,
            return_dict=True,
        )
        sample = output.sample
        if sample.shape != x.shape:
            raise RuntimeError(
                "The Diffusers component returned an incompatible shape: "
                f"{tuple(sample.shape)} instead of {tuple(x.shape)}."
            )
        return sample

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool = True,
    ) -> Path:
        save_directory = Path(save_directory)
        component_directory = save_directory / "component"
        component_directory.mkdir(parents=True, exist_ok=True)
        self.component.save_pretrained(
            component_directory,
            safe_serialization=safe_serialization,
        )
        metadata = {
            "format": ADAPTER_FORMAT,
            "version": ADAPTER_VERSION,
            "component_class": "UNet2DModel",
            "component_subdir": "component",
            "timestep_scale": self.timestep_scale,
            "null_class_index": self.null_class_index,
            "safe_serialization": bool(safe_serialization),
        }
        save_directory.mkdir(parents=True, exist_ok=True)
        with (save_directory / ADAPTER_CONFIG_NAME).open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
        return save_directory

    @classmethod
    def from_diffusers_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str | None = None,
        timestep_scale: float | None = None,
        null_class_index: int | None = None,
        local_files_only: bool = False,
        **load_kwargs: Any,
    ) -> "DiffusersUNet2DAdapter":
        UNet2DModel = _diffusers_unet2d_class()
        kwargs = dict(load_kwargs)
        kwargs.setdefault("low_cpu_mem_usage", False)
        component = UNet2DModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
            **kwargs,
        )
        return cls(
            component,
            timestep_scale=timestep_scale,
            null_class_index=null_class_index,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_directory: str | Path,
        *,
        local_files_only: bool = True,
        **load_kwargs: Any,
    ) -> "DiffusersUNet2DAdapter":
        pretrained_directory = Path(pretrained_directory)
        metadata_path = pretrained_directory / ADAPTER_CONFIG_NAME
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if metadata.get("format") != ADAPTER_FORMAT:
            raise ValueError(f"Unsupported adapter format in {metadata_path}.")
        if int(metadata.get("version", -1)) != ADAPTER_VERSION:
            raise ValueError(f"Unsupported adapter version in {metadata_path}.")
        if metadata.get("component_class") != "UNet2DModel":
            raise ValueError("Only UNet2DModel adapter directories are supported.")
        return cls.from_diffusers_pretrained(
            pretrained_directory / metadata["component_subdir"],
            timestep_scale=float(metadata["timestep_scale"]),
            null_class_index=metadata.get("null_class_index"),
            local_files_only=local_files_only,
            **load_kwargs,
        )


__all__ = [
    "ADAPTER_CONFIG_NAME",
    "DiffusersUNet2DAdapter",
    "create_diffusers_unet2d",
]
