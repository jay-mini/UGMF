from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.train import build_dataset, build_scheduler  # noqa: E402
from ugm.models.build import (  # noqa: E402
    MODEL_CONFIG_KEYS,
    build_model,
    resolved_model_config,
)
from ugm.objectives.consistency import (  # noqa: E402
    ConsistencyDistillationObjective,
)
from ugm.trainers.distill_trainer import (  # noqa: E402
    ConsistencyDistillationTrainer,
)


DEFAULT_DISTILL_CONFIG: dict[str, Any] = {
    "objective": "consistency_distillation",
    "teacher_checkpoint": None,
    "teacher_use_ema": False,
    "dataset": None,
    "data_dir": "data",
    "output_dir": "outputs/consistency_distillation",
    "batch_size": 128,
    "num_epochs": 10,
    "max_steps": None,
    "lr": 1e-5,
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
    "num_time_steps": 20,
    "target_decay": 0.999,
    "target_update_every": 1,
    "teacher_guidance_scale": 1.0,
    "cond_drop_prob": 0.1,
    "resume": None,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill an FM teacher into a boundary-consistency student.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--teacher_checkpoint", type=str)
    parser.add_argument("--teacher_use_ema", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"])
    parser.add_argument("--data_dir", type=str)
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
    parser.add_argument("--num_time_steps", type=int)
    parser.add_argument("--target_decay", type=float)
    parser.add_argument("--target_update_every", type=int)
    parser.add_argument("--teacher_guidance_scale", type=float)
    parser.add_argument("--cond_drop_prob", type=float)
    parser.add_argument("--resume", type=str)
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Distillation config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("The distillation YAML top level must be a mapping.")
    return dict(value)


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    parsed = vars(build_parser().parse_args(argv))
    config_path = Path(parsed.pop("config"))
    yaml_config = _load_yaml(config_path)
    allowed = set(DEFAULT_DISTILL_CONFIG)
    unknown_yaml = set(yaml_config) - allowed
    unknown_cli = set(parsed) - allowed
    if unknown_yaml:
        raise ValueError(f"Unknown distillation YAML keys: {sorted(unknown_yaml)}")
    if unknown_cli:
        raise ValueError(f"Unknown distillation CLI keys: {sorted(unknown_cli)}")
    config = {**DEFAULT_DISTILL_CONFIG, **yaml_config, **parsed}
    validate_config(config)
    config["config_path"] = str(config_path.resolve())
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config["objective"] != "consistency_distillation":
        raise ValueError("objective must be consistency_distillation.")
    if not config["teacher_checkpoint"]:
        raise ValueError("teacher_checkpoint is required.")
    if config["dataset"] is not None and config["dataset"] not in {"mnist", "cifar10"}:
        raise ValueError("dataset must be mnist or cifar10.")
    for key in ("batch_size", "num_epochs", "log_every", "save_every", "num_time_steps"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if config["max_steps"] is not None and int(config["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive when provided.")
    if float(config["lr"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("lr must be positive and weight_decay non-negative.")
    if config["scheduler"] not in {"none", "cosine"}:
        raise ValueError("scheduler must be none or cosine.")
    if config["scheduler_total_steps"] is not None and int(config["scheduler_total_steps"]) <= 0:
        raise ValueError("scheduler_total_steps must be positive when provided.")
    if not 0 <= float(config["scheduler_min_lr"]) <= float(config["lr"]):
        raise ValueError("scheduler_min_lr must lie in [0, lr].")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative.")
    if config["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be no, fp16, or bf16.")
    if config["grad_clip"] is not None and float(config["grad_clip"]) <= 0:
        raise ValueError("grad_clip must be positive when provided.")
    if not 0 <= float(config["target_decay"]) < 1:
        raise ValueError("target_decay must lie in [0, 1).")
    if int(config["target_update_every"]) <= 0:
        raise ValueError("target_update_every must be positive.")
    if float(config["teacher_guidance_scale"]) < 0:
        raise ValueError("teacher_guidance_scale must be non-negative.")
    if not 0 <= float(config["cond_drop_prob"]) <= 1:
        raise ValueError("cond_drop_prob must lie in [0, 1].")


def _resolve(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _checkpoint_config(path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    value = checkpoint.get("config")
    if isinstance(value, dict) and value:
        return dict(value)
    for filename in ("config.json", "args.json"):
        candidate = path.parent.parent / filename
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as file:
                value = json.load(file)
            if isinstance(value, dict):
                return value
    raise FileNotFoundError(f"Cannot locate teacher config for {path}.")


def _build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    return build_model(config)


def load_teacher_and_student(
    checkpoint_path: Path,
    use_ema: bool,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid teacher checkpoint: {checkpoint_path}")
    teacher_config = _checkpoint_config(checkpoint_path, checkpoint)
    if str(teacher_config.get("objective", "")).lower() != "fm":
        raise ValueError("Consistency distillation currently requires an FM teacher.")
    if use_ema:
        ema_state = checkpoint.get("ema")
        if not isinstance(ema_state, dict) or "ema_model" not in ema_state:
            raise KeyError("teacher_use_ema is true, but checkpoint has no EMA state.")
        source_state = ema_state["ema_model"]
        source = "ema"
    else:
        source_state = checkpoint["model"]
        source = "online"
    teacher = _build_model(teacher_config)
    student = _build_model(teacher_config)
    teacher.load_state_dict(source_state, strict=True)
    student.load_state_dict(source_state, strict=True)
    metadata = {
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "teacher_state_source": source,
    }
    return teacher, student, teacher_config, metadata


def _validate_resume(active: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    keys = (
        "teacher_checkpoint", "teacher_use_ema", "dataset", "batch_size",
        "lr", "weight_decay", "scheduler", "scheduler_total_steps",
        "scheduler_min_lr", "num_time_steps", "target_decay",
        "target_update_every", "teacher_guidance_scale", "cond_drop_prob",
        *MODEL_CONFIG_KEYS, "class_cond", "num_classes",
    )
    differences = [
        f"{key}: active={active[key]!r}, checkpoint={saved[key]!r}"
        for key in keys
        if key in active and key in saved and active[key] != saved[key]
    ]
    if differences:
        raise ValueError("Incompatible distillation resume config: " + "; ".join(differences))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_distillation(config: dict[str, Any]) -> ConsistencyDistillationTrainer:
    _seed_all(int(config["seed"]))
    teacher_path = _resolve(str(config["teacher_checkpoint"]))
    output_dir = _resolve(str(config["output_dir"]))
    data_dir = _resolve(str(config["data_dir"]))
    config["teacher_checkpoint"] = str(teacher_path)
    config["output_dir"] = str(output_dir)
    config["data_dir"] = str(data_dir)
    if config["resume"] is not None:
        config["resume"] = str(_resolve(str(config["resume"])))
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    teacher, student, teacher_config, metadata = load_teacher_and_student(
        teacher_path,
        bool(config["teacher_use_ema"]),
    )
    teacher_dataset = str(teacher_config.get("dataset", "")).lower()
    dataset_name = config["dataset"] or teacher_dataset
    if dataset_name != teacher_dataset:
        raise ValueError(
            f"Configured dataset {dataset_name!r} does not match teacher {teacher_dataset!r}."
        )
    dataset, in_channels, image_size, num_classes = build_dataset(
        dataset_name,
        str(data_dir),
    )
    class_cond = bool(teacher_config.get("class_cond", False))
    if in_channels != int(teacher_config["in_channels"]):
        raise ValueError("Dataset channels do not match teacher architecture.")
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)
    student.to(device)
    objective = ConsistencyDistillationObjective(
        teacher=teacher,
        student=student,
        num_time_steps=int(config["num_time_steps"]),
        target_decay=float(config["target_decay"]),
        target_update_every=int(config["target_update_every"]),
        teacher_guidance_scale=float(config["teacher_guidance_scale"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = build_scheduler(config, optimizer, len(loader))
    config.update(
        {
            **metadata,
            "dataset": dataset_name,
            "device": str(device),
            "in_channels": in_channels,
            "out_channels": int(teacher_config.get("out_channels", in_channels)),
            "image_size": image_size,
            **resolved_model_config(teacher_config),
            "class_cond": class_cond,
            "num_classes": num_classes if class_cond else None,
            "time_direction": "noise t=0 -> data t=1",
            "teacher_step": "one adjacent Euler step toward increasing t",
            "boundary_parameterization": "f(x,t)=x+(1-t)F(x,t)",
        }
    )
    trainer = ConsistencyDistillationTrainer(
        model=student,
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
        ema=None,
        cond_drop_prob=float(config["cond_drop_prob"]) if class_cond else 0.0,
    )
    if config["resume"] is not None:
        state = trainer.load_checkpoint(Path(config["resume"]))
        _validate_resume(config, state.config)
        print(f"Resumed distillation at epoch={state.epoch}, step={state.global_step}")
    else:
        print(
            f"Initialized student and EMA target from {metadata['teacher_state_source']} "
            f"teacher weights at step {metadata['teacher_checkpoint_global_step']}."
        )
    trainer.train(
        num_epochs=int(config["num_epochs"]),
        max_steps=config["max_steps"],
    )
    return trainer


def main() -> None:
    run_distillation(parse_config())


if __name__ == "__main__":
    main()
