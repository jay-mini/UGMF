from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "objective": "ddpm",
    "dataset": "mnist",
    "data_dir": "data",
    "output_dir": None,

    "batch_size": 128,
    "num_epochs": 20,
    "max_steps": None,

    "lr": 2e-4,
    "weight_decay": 1e-4,

    "scheduler": "cosine",
    "scheduler_total_steps": None,
    "scheduler_min_lr": 0.0,

    "model_type": "unet",
    "base_channels": 64,
    "time_dim": 256,
    "dit_patch_size": 4,
    "dit_hidden_size": 192,
    "dit_depth": 6,
    "dit_num_heads": 3,
    "dit_mlp_ratio": 4.0,
    "diffusers_model_path": None,
    "diffusers_subfolder": None,
    "diffusers_timestep_scale": None,
    "diffusers_local_files_only": False,
    "diffusers_layers_per_block": 1,
    "diffusers_block_out_channels": [32, 64],
    "diffusers_down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
    "diffusers_up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
    "diffusers_norm_num_groups": 8,
    "diffusers_attention_head_dim": 8,
    "diffusers_add_attention": True,

    "num_workers": 4,
    "mixed_precision": "no",
    "grad_clip": 1.0,

    "log_every": 100,
    "save_every": 1000,
    "seed": 42,
    "class_cond": False,
    "cond_drop_prob": 0.1,
    "guidance_scale": 1.0,

    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "prediction_type": "epsilon",

    "t_min": 0.0,
    "t_max": 1.0,

    "ema_decay": 0.999,
    "ema_update_after_step": 100,
    "ema_update_every": 1,

    "resume": None,
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)

    if loaded is None:
        return {}

    if not isinstance(loaded, Mapping):
        raise TypeError(
            "The top level of a YAML configuration must be a mapping."
        )

    return dict(loaded)


def merge_configs(
    defaults: Mapping[str, Any],
    yaml_config: Mapping[str, Any],
    cli_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    allowed_keys = set(defaults)

    unknown_yaml_keys = set(yaml_config) - allowed_keys
    if unknown_yaml_keys:
        raise ValueError(
            "Unknown YAML configuration keys: "
            f"{sorted(unknown_yaml_keys)}"
        )

    unknown_cli_keys = set(cli_overrides) - allowed_keys
    if unknown_cli_keys:
        raise ValueError(
            "Unknown command-line configuration keys: "
            f"{sorted(unknown_cli_keys)}"
        )

    return {
        **defaults,
        **yaml_config,
        **cli_overrides,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if config["objective"] not in {"ddpm", "fm"}:
        raise ValueError(
            "objective must be 'ddpm' or 'fm'."
        )

    if config["dataset"] not in {"mnist", "cifar10"}:
        raise ValueError(
            "dataset must be 'mnist' or 'cifar10'."
        )

    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive.")

    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be positive.")

    if config["max_steps"] is not None and config["max_steps"] <= 0:
        raise ValueError("max_steps must be positive when provided.")

    if config["lr"] <= 0:
        raise ValueError("lr must be positive.")

    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be non-negative.")

    if config["scheduler"] not in {"none", "cosine"}:
        raise ValueError("scheduler must be 'none' or 'cosine'.")

    if (
        config["scheduler_total_steps"] is not None
        and config["scheduler_total_steps"] <= 0
    ):
        raise ValueError(
            "scheduler_total_steps must be positive when provided."
        )

    if not 0.0 <= config["scheduler_min_lr"] <= config["lr"]:
        raise ValueError(
            "scheduler_min_lr must satisfy 0 <= scheduler_min_lr <= lr."
        )

    if config["model_type"] not in {
        "unet",
        "dit_tiny",
        "diffusers_unet2d",
    }:
        raise ValueError(
            "model_type must be 'unet', 'dit_tiny', or 'diffusers_unet2d'."
        )

    if config["base_channels"] <= 0:
        raise ValueError("base_channels must be positive.")

    if config["time_dim"] <= 0:
        raise ValueError("time_dim must be positive.")

    if config["dit_patch_size"] <= 0:
        raise ValueError("dit_patch_size must be positive.")

    if config["dit_hidden_size"] <= 0 or config["dit_hidden_size"] % 4 != 0:
        raise ValueError("dit_hidden_size must be positive and divisible by 4.")

    if config["dit_depth"] <= 0:
        raise ValueError("dit_depth must be positive.")

    if config["dit_num_heads"] <= 0:
        raise ValueError("dit_num_heads must be positive.")

    if config["dit_hidden_size"] % config["dit_num_heads"] != 0:
        raise ValueError("dit_hidden_size must be divisible by dit_num_heads.")

    if config["dit_mlp_ratio"] <= 0:
        raise ValueError("dit_mlp_ratio must be positive.")

    if config["diffusers_timestep_scale"] is not None and (
        not isinstance(config["diffusers_timestep_scale"], (int, float))
        or not math.isfinite(config["diffusers_timestep_scale"])
        or config["diffusers_timestep_scale"] <= 0
    ):
        raise ValueError("diffusers_timestep_scale must be finite and positive.")

    for key in (
        "diffusers_layers_per_block",
        "diffusers_norm_num_groups",
        "diffusers_attention_head_dim",
    ):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    block_channels = config["diffusers_block_out_channels"]
    down_blocks = config["diffusers_down_block_types"]
    up_blocks = config["diffusers_up_block_types"]
    if not all(isinstance(values, (list, tuple)) for values in (
        block_channels,
        down_blocks,
        up_blocks,
    )):
        raise TypeError("Diffusers block settings must be lists or tuples.")
    if not len(block_channels) == len(down_blocks) == len(up_blocks):
        raise ValueError("Diffusers block setting lengths must match.")
    if not block_channels or any(int(value) <= 0 for value in block_channels):
        raise ValueError("diffusers_block_out_channels must be positive.")

    image_size = config.get("image_size")
    if (
        config["model_type"] == "dit_tiny"
        and image_size is not None
        and image_size % config["dit_patch_size"] != 0
    ):
        raise ValueError("image_size must be divisible by dit_patch_size.")

    if config["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative.")

    if config["mixed_precision"] not in {
        "no",
        "fp16",
        "bf16",
    }:
        raise ValueError(
            "mixed_precision must be 'no', 'fp16', or 'bf16'."
        )

    if config["grad_clip"] is not None and config["grad_clip"] <= 0:
        raise ValueError("grad_clip must be positive when provided.")

    if config["log_every"] <= 0:
        raise ValueError("log_every must be positive.")

    if config["save_every"] <= 0:
        raise ValueError("save_every must be positive.")

    if not 0.0 <= config["cond_drop_prob"] <= 1.0:
        raise ValueError("cond_drop_prob must be in [0, 1].")

    if (
        not isinstance(config["guidance_scale"], (int, float))
        or not math.isfinite(config["guidance_scale"])
        or config["guidance_scale"] < 0.0
    ):
        raise ValueError(
            "guidance_scale must be finite and non-negative."
        )

    if not (0.0 < config["ema_decay"] <= 1.0):
        raise ValueError("ema_decay must be in (0, 1].")

    if config["ema_update_after_step"] < 0:
        raise ValueError("ema_update_after_step must be non-negative.")

    if config["ema_update_every"] <= 0:
        raise ValueError("ema_update_every must be positive.")

    if config["objective"] == "ddpm":
        if config["prediction_type"] not in {"epsilon", "x0"}:
            raise ValueError(
                "prediction_type must be 'epsilon' or 'x0'."
            )

        if config["num_timesteps"] <= 0:
            raise ValueError(
                "num_timesteps must be positive."
            )

        if not (
            0.0
            < config["beta_start"]
            <= config["beta_end"]
            < 1.0
        ):
            raise ValueError(
                "DDPM beta values must satisfy "
                "0 < beta_start <= beta_end < 1."
            )

    if config["objective"] == "fm":
        if not (
            0.0
            <= config["t_min"]
            < config["t_max"]
            <= 1.0
        ):
            raise ValueError(
                "Flow Matching time range must satisfy "
                "0 <= t_min < t_max <= 1."
            )
