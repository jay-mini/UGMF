from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make src/importable when running:
#       python scripts/train.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ugm.models.build import MODEL_CONFIG_KEYS, build_model
from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.trainers.base_trainer import BaseTrainer
from ugm.utils.config import (
    DEFAULT_CONFIG,
    load_yaml_config,
    merge_configs,
    validate_config,
)
from ugm.utils.ema import EMA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DDPM or Flow Matching.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to an experiment YAML file.",
    )

    parser.add_argument(
        "--objective",
        choices=["ddpm", "fm"],
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
    )
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--max_steps", type=int)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
    )
    parser.add_argument("--scheduler_total_steps", type=int)
    parser.add_argument("--scheduler_min_lr", type=float)

    parser.add_argument("--base_channels", type=int)
    parser.add_argument("--time_dim", type=int)
    parser.add_argument(
        "--model_type",
        choices=["unet", "dit_tiny", "diffusers_unet2d"],
    )
    parser.add_argument("--dit_patch_size", type=int)
    parser.add_argument("--dit_hidden_size", type=int)
    parser.add_argument("--dit_depth", type=int)
    parser.add_argument("--dit_num_heads", type=int)
    parser.add_argument("--dit_mlp_ratio", type=float)
    parser.add_argument("--diffusers_model_path", type=str)
    parser.add_argument("--diffusers_subfolder", type=str)
    parser.add_argument("--diffusers_timestep_scale", type=float)
    parser.add_argument(
        "--diffusers_local_files_only",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--diffusers_layers_per_block", type=int)
    parser.add_argument("--diffusers_block_out_channels", type=int, nargs="+")
    parser.add_argument("--diffusers_down_block_types", type=str, nargs="+")
    parser.add_argument("--diffusers_up_block_types", type=str, nargs="+")
    parser.add_argument("--diffusers_norm_num_groups", type=int)
    parser.add_argument("--diffusers_attention_head_dim", type=int)
    parser.add_argument(
        "--diffusers_add_attention",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--num_workers", type=int)

    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--grad_clip", type=float)

    parser.add_argument("--log_every", type=int)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--seed", type=int)

    parser.add_argument(
        "--class_cond",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--cond_drop_prob", type=float)
    parser.add_argument("--guidance_scale", type=float)

    parser.add_argument("--num_timesteps", type=int)
    parser.add_argument("--beta_start", type=float)
    parser.add_argument("--beta_end", type=float)
    parser.add_argument(
        "--prediction_type",
        choices=["epsilon", "x0"],
    )

    parser.add_argument("--t_min", type=float)
    parser.add_argument("--t_max", type=float)

    parser.add_argument("--ema_decay", type=float)
    parser.add_argument(
        "--ema_update_after_step",
        type=int,
    )
    parser.add_argument(
        "--ema_update_every",
        type=int,
    )

    parser.add_argument("--resume", type=str)

    return parser


def parse_config(
    argv: list[str] | None = None,
) -> dict:
    parser = build_parser()

    parsed = vars(parser.parse_args(argv))

    config_path = Path(parsed.pop("config"))
    yaml_config = load_yaml_config(config_path)

    effective_config = merge_configs(
        defaults=DEFAULT_CONFIG,
        yaml_config=yaml_config,
        cli_overrides=parsed,
    )

    validate_config(effective_config)

    effective_config["config_path"] = str(
        config_path.resolve()
    )

    return effective_config


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataset(
        dataset_name: str,
        data_dir: str,
):
    """
    Build normalized dataset.

    Images are normalized to [-1, 1], which is standard for diffusion/FM imaging training.
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        in_channels = 1
        image_size = 28
        num_classes = 10
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        in_channels = 3
        image_size = 32
        num_classes = 10

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, in_channels, image_size, num_classes


def build_objective(config: dict):
    if config["objective"] == "ddpm":
        return DDPMobjective(
            num_timesteps=config["num_timesteps"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
            prediction_type=config["prediction_type"],
        )
    
    if config["objective"] == "fm":
        return FlowMatchingObjective(
            t_min=config["t_min"],
            t_max=config["t_max"],
        )
    
    raise ValueError(f"Unknown objective: {config['objective']}")


def build_scheduler(
    config: dict,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
):
    """Build the per-optimizer-step learning-rate scheduler."""
    scheduler_name = config["scheduler"]

    if scheduler_name == "none":
        config["scheduler_total_steps"] = None
        return None

    if scheduler_name == "cosine":
        total_steps = config["scheduler_total_steps"]

        if total_steps is None:
            # max_steps is an interruption boundary, not the intended
            # learning-rate horizon. This keeps the schedule stable when a
            # run is stopped and later resumed with a larger max_steps.
            total_steps = config["num_epochs"] * steps_per_epoch

        if total_steps <= 0:
            raise ValueError("The scheduler requires at least one step.")

        config["scheduler_total_steps"] = int(total_steps)

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(total_steps),
            eta_min=float(config["scheduler_min_lr"]),
        )

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def validate_resume_compatibility(
    active_config: dict,
    saved_config: dict,
) -> None:
    """Reject changes to stateful settings during checkpoint resume."""
    stateful_keys = (
        "objective",
        "dataset",
        *MODEL_CONFIG_KEYS,
        "class_cond",
        "cond_drop_prob",
        "num_timesteps",
        "beta_start",
        "beta_end",
        "prediction_type",
        "t_min",
        "t_max",
        "lr",
        "weight_decay",
        "scheduler",
        "scheduler_total_steps",
        "scheduler_min_lr",
        "ema_decay",
        "ema_update_after_step",
        "ema_update_every",
    )

    mismatches = []
    for key in stateful_keys:
        if (
            key in active_config
            and key in saved_config
            and active_config[key] != saved_config[key]
        ):
            mismatches.append(
                f"{key}: active={active_config[key]!r}, "
                f"checkpoint={saved_config[key]!r}"
            )

    if mismatches:
        raise ValueError(
            "Resume configuration is incompatible with the checkpoint: "
            + "; ".join(mismatches)
        )


def main() -> None:
    config = parse_config()
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    data_dir = data_dir.resolve()
    config["data_dir"] = str(data_dir)

    train_dataset, in_channels, image_size, num_classes = build_dataset(
        dataset_name=config["dataset"],
        data_dir=str(data_dir),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    config.update(
        {
            "device": str(device),
            "in_channels": in_channels,
            "out_channels": in_channels,
            "image_size": image_size,
            "num_classes": num_classes,
        }
    )
    model = build_model(config).to(device)

    objective = build_objective(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = build_scheduler(
        config=config,
        optimizer=optimizer,
        steps_per_epoch=len(train_loader),
    )

    ema = EMA(
        model=model,
        decay=config["ema_decay"],
        update_after_step=config["ema_update_after_step"],
        update_every=config["ema_update_every"],
    )

    if config["output_dir"] is None:
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / f"{config['objective']}_{config['dataset']}"
        )
    else:
        output_dir = Path(config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    output_dir = output_dir.resolve()

    if config["resume"] is not None:
        resume_path = Path(config["resume"])
        if not resume_path.is_absolute():
            resume_path = PROJECT_ROOT / resume_path
        config["resume"] = str(resume_path.resolve())

    config["output_dir"] = str(output_dir)

    trainer = BaseTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
        mixed_precision=config["mixed_precision"],
        grad_clip=config["grad_clip"],
        log_every=config["log_every"],
        save_every=config["save_every"],
        config=config,
        ema=ema,
        cond_drop_prob=(
            config["cond_drop_prob"]
            if config["class_cond"]
            else 0.0
        ),
    )

    if config["resume"] is not None:
        resume_path = Path(config["resume"])

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )
        
        resume_state = trainer.load_checkpoint(
            resume_path,
            strict_model=True,
            strict_ema=True,
            strict_objective=True,
            restore_rng=True,
        )

        validate_resume_compatibility(
            active_config=config,
            saved_config=resume_state.config,
        )

        print(
            f"Resume from {resume_state.checkpoint_path}:"
            f"epoch = {resume_state.epoch}, "
            f"global_step={resume_state.global_step}"
        )


    trainer.train(
        num_epochs=config["num_epochs"],
        max_steps=config["max_steps"],
    )


if __name__ == "__main__":
    main()
