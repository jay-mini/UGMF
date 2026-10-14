from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ugm.data.rectified_flow import (  # noqa: E402
    RectifiedFlowPairDataset,
    ShardShuffleSampler,
)
from ugm.models.build import (  # noqa: E402
    MODEL_CONFIG_KEYS,
    build_model,
    resolved_model_config,
)
from ugm.objectives.rectified_flow import RectifiedFlowObjective  # noqa: E402
from ugm.trainers.finetune_trainer import (  # noqa: E402
    RectifiedFlowTrainer,
    initialize_from_checkpoint,
)
from ugm.utils.ema import EMA  # noqa: E402


RF_DEFAULT_CONFIG: dict[str, Any] = {
    "objective": "rectified_flow",
    "pairs_dir": None,
    "init_checkpoint": None,
    "init_use_ema": False,
    "output_dir": "outputs/rectified_flow",
    "batch_size": 128,
    "num_epochs": 10,
    "max_steps": None,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "scheduler_total_steps": None,
    "scheduler_min_lr": 0.0,
    "num_workers": 4,
    "mixed_precision": "no",
    "grad_clip": 1.0,
    "log_every": 100,
    "save_every": 1000,
    "seed": 42,
    "t_min": 0.0,
    "t_max": 1.0,
    "cond_drop_prob": 0.1,
    "ema_decay": 0.999,
    "ema_update_after_step": 100,
    "ema_update_every": 1,
    "resume": None,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a Flow Matching checkpoint with RF endpoint pairs.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pairs_dir", type=str)
    parser.add_argument("--init_checkpoint", type=str)
    parser.add_argument("--init_use_ema", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--scheduler", choices=["none", "cosine"])
    parser.add_argument("--scheduler_total_steps", type=int)
    parser.add_argument("--scheduler_min_lr", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"])
    parser.add_argument("--grad_clip", type=float)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--t_min", type=float)
    parser.add_argument("--t_max", type=float)
    parser.add_argument("--cond_drop_prob", type=float)
    parser.add_argument("--ema_decay", type=float)
    parser.add_argument("--ema_update_after_step", type=int)
    parser.add_argument("--ema_update_every", type=int)
    parser.add_argument("--resume", type=str)
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("The top level of the RF configuration must be a mapping.")
    return dict(value)


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    parsed = vars(build_parser().parse_args(argv))
    config_path = Path(parsed.pop("config"))
    yaml_config = _load_yaml(config_path)
    allowed = set(RF_DEFAULT_CONFIG)
    unknown = set(yaml_config) - allowed
    if unknown:
        raise ValueError(f"Unknown RF YAML configuration keys: {sorted(unknown)}")
    unknown_cli = set(parsed) - allowed
    if unknown_cli:
        raise ValueError(f"Unknown RF CLI configuration keys: {sorted(unknown_cli)}")
    config = {**RF_DEFAULT_CONFIG, **yaml_config, **parsed}
    validate_config(config)
    config["config_path"] = str(config_path.resolve())
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config["objective"] != "rectified_flow":
        raise ValueError("objective must be 'rectified_flow'.")
    if not config["pairs_dir"] or not config["init_checkpoint"]:
        raise ValueError("pairs_dir and init_checkpoint are required.")
    for key in ("batch_size", "num_epochs", "log_every", "save_every"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if config["max_steps"] is not None and int(config["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if float(config["lr"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("lr must be positive and weight_decay non-negative.")
    if config["scheduler"] not in {"none", "cosine"}:
        raise ValueError("scheduler must be 'none' or 'cosine'.")
    if config["scheduler_total_steps"] is not None and int(config["scheduler_total_steps"]) <= 0:
        raise ValueError("scheduler_total_steps must be positive when provided.")
    if not 0 <= float(config["scheduler_min_lr"]) <= float(config["lr"]):
        raise ValueError("scheduler_min_lr must satisfy 0 <= min_lr <= lr.")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative.")
    if config["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16.")
    if config["grad_clip"] is not None and float(config["grad_clip"]) <= 0:
        raise ValueError("grad_clip must be positive when provided.")
    if not 0 <= float(config["cond_drop_prob"]) <= 1:
        raise ValueError("cond_drop_prob must be in [0, 1].")
    if not 0 <= float(config["t_min"]) < float(config["t_max"]) <= 1:
        raise ValueError("RF time range must satisfy 0 <= t_min < t_max <= 1.")
    if not 0 < float(config["ema_decay"]) <= 1:
        raise ValueError("ema_decay must be in (0, 1].")
    if int(config["ema_update_after_step"]) < 0 or int(config["ema_update_every"]) <= 0:
        raise ValueError("Invalid EMA update settings.")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(saved, dict) and saved:
        return dict(saved)
    for filename in ("config.json", "args.json"):
        candidate = path.parent.parent / filename
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as file:
                value = json.load(file)
            if isinstance(value, dict):
                return value
    raise FileNotFoundError(f"Cannot locate the model config for {path}.")


def build_scheduler(
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
):
    if config["scheduler"] == "none":
        config["scheduler_total_steps"] = None
        return None
    total_steps = config["scheduler_total_steps"]
    if total_steps is None:
        total_steps = int(config["num_epochs"]) * steps_per_epoch
        config["scheduler_total_steps"] = total_steps
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(total_steps),
        eta_min=float(config["scheduler_min_lr"]),
    )


def validate_resume_compatibility(active: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    keys = (
        "objective", "pairs_dir", "init_checkpoint", "init_use_ema",
        "batch_size", "lr", "weight_decay", "scheduler",
        "scheduler_total_steps", "scheduler_min_lr", "t_min", "t_max",
        "cond_drop_prob", "ema_decay", "ema_update_after_step",
        "ema_update_every", "in_channels", "image_size", "class_cond",
        "num_classes", *MODEL_CONFIG_KEYS,
    )
    mismatches = [
        f"{key}: active={active[key]!r}, checkpoint={saved[key]!r}"
        for key in keys
        if key in active and key in saved and active[key] != saved[key]
    ]
    if mismatches:
        raise ValueError("RF resume configuration is incompatible: " + "; ".join(mismatches))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_finetuning(config: dict[str, Any]) -> RectifiedFlowTrainer:
    set_seed(int(config["seed"]))
    pairs_dir = _resolve(str(config["pairs_dir"]))
    init_checkpoint = _resolve(str(config["init_checkpoint"]))
    output_dir = _resolve(str(config["output_dir"]))
    config["pairs_dir"] = str(pairs_dir)
    config["init_checkpoint"] = str(init_checkpoint)
    config["output_dir"] = str(output_dir)
    if not init_checkpoint.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {init_checkpoint}")

    dataset = RectifiedFlowPairDataset(pairs_dir, verify=True)
    manifest = dataset.manifest
    init_config = _load_checkpoint_config(init_checkpoint)
    if str(init_config.get("objective", "")).lower() != "fm":
        raise ValueError("init_checkpoint must be a Flow Matching checkpoint.")
    sample_shape = tuple(int(value) for value in manifest["sample_shape"])
    in_channels = int(init_config.get("in_channels", sample_shape[0]))
    image_size = int(init_config.get("image_size", sample_shape[-1]))
    if sample_shape != (in_channels, image_size, image_size):
        raise ValueError(
            f"Paired sample shape {sample_shape} is incompatible with the initialized model."
        )
    class_cond = bool(init_config.get("class_cond", False))
    if class_cond != bool(manifest.get("conditional", False)):
        raise ValueError("Pair conditioning does not match the initialized model.")
    num_classes = int(init_config.get("num_classes", 10)) if class_cond else None
    if class_cond and int(manifest.get("num_classes", -1)) != num_classes:
        raise ValueError("Pair num_classes does not match the initialized model.")

    model_config = {
        **init_config,
        "in_channels": in_channels,
        "out_channels": int(init_config.get("out_channels", in_channels)),
        "image_size": image_size,
        "num_classes": num_classes,
        "class_cond": class_cond,
    }
    model = build_model(model_config)
    initialized = initialize_from_checkpoint(
        model,
        init_checkpoint,
        use_ema=bool(config["init_use_ema"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    objective = RectifiedFlowObjective(
        t_min=float(config["t_min"]),
        t_max=float(config["t_max"]),
    ).to(device)
    data_generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        sampler=ShardShuffleSampler(dataset, generator=data_generator),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    if len(loader) == 0:
        raise ValueError("The paired dataset produced an empty DataLoader.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_scheduler(config, optimizer, len(loader))
    ema = EMA(
        model,
        decay=float(config["ema_decay"]),
        update_after_step=int(config["ema_update_after_step"]),
        update_every=int(config["ema_update_every"]),
    )
    config.update(
        {
            "device": str(device),
            "dataset": manifest.get("dataset"),
            "num_pairs": len(dataset),
            "pair_manifest": str(pairs_dir / "manifest.json"),
            "teacher_checkpoint": manifest["teacher"]["checkpoint"],
            "teacher_sampling": dict(manifest["generation"]),
            "initialization_source": initialized.source,
            "initialization_checkpoint_global_step": initialized.checkpoint_global_step,
            "in_channels": in_channels,
            "out_channels": int(init_config.get("out_channels", in_channels)),
            "image_size": image_size,
            **resolved_model_config(init_config),
            "class_cond": class_cond,
            "num_classes": num_classes,
        }
    )
    trainer = RectifiedFlowTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        device=device,
        output_dir=output_dir,
        mixed_precision=str(config["mixed_precision"]),
        grad_clip=config["grad_clip"],
        log_every=int(config["log_every"]),
        save_every=int(config["save_every"]),
        config=config,
        ema=ema,
        cond_drop_prob=float(config["cond_drop_prob"]) if class_cond else 0.0,
    )
    if config["resume"] is not None:
        resume_path = _resolve(str(config["resume"]))
        config["resume"] = str(resume_path)
        state = trainer.load_checkpoint(resume_path)
        validate_resume_compatibility(config, state.config)
        print(f"Resumed RF training at epoch={state.epoch}, step={state.global_step}")
    else:
        print(
            f"Initialized RF student from {initialized.source} weights at "
            f"teacher step {initialized.checkpoint_global_step}."
        )
    trainer.train(
        num_epochs=int(config["num_epochs"]),
        max_steps=config["max_steps"],
    )
    return trainer


def main() -> None:
    run_finetuning(parse_config())


if __name__ == "__main__":
    main()
