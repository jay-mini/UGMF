from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.train import build_dataset, build_objective
from ugm.models.adapters.lora import (
    assert_only_lora_trainable,
    inject_lora,
    load_lora_adapter,
    lora_parameter_names,
    lora_parameters,
    save_lora_adapter,
)
from ugm.models.build import build_model
from ugm.trainers.base_trainer import BaseTrainer
from ugm.utils.config import load_yaml_config


class LoggingTrainer(BaseTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.step_log: list[dict[str, Any]] = []

    def _train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        metrics = super()._train_step(batch)
        self.step_log.append(
            {
                "global_step": self.global_step,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                **metrics,
            }
        )
        return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune the existing DiT-Tiny backbone with LoRA only.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_checkpoint")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    return parser


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    parsed = vars(build_parser().parse_args(argv))
    config_path = Path(parsed.pop("config"))
    config = {**load_yaml_config(config_path), **parsed}
    required = {
        "base_checkpoint",
        "output_dir",
        "data_dir",
        "batch_size",
        "num_epochs",
        "max_steps",
        "lr",
        "weight_decay",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing LoRA configuration keys: {sorted(missing)}")
    if config["max_steps"] <= 0 or config["batch_size"] <= 0:
        raise ValueError("max_steps and batch_size must be positive.")
    if config["lr"] <= 0 or config["weight_decay"] < 0:
        raise ValueError("lr must be positive and weight_decay non-negative.")
    if not isinstance(config["lora_target_modules"], list):
        raise TypeError("lora_target_modules must be a YAML list.")
    config.setdefault("base_use_ema", False)
    config.setdefault("num_workers", 0)
    config.setdefault("mixed_precision", "no")
    config.setdefault("grad_clip", 1.0)
    config.setdefault("log_every", 1)
    config.setdefault("save_every", config["max_steps"])
    config.setdefault("seed", 42)
    config.setdefault("device", "auto")
    config["config_path"] = str(config_path.resolve())
    return config


def _path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _base_state(checkpoint: dict[str, Any], use_ema: bool) -> dict[str, torch.Tensor]:
    if not use_ema:
        return checkpoint["model"]
    ema = checkpoint.get("ema")
    if not ema or "ema_model" not in ema:
        raise ValueError("base_use_ema=true but the checkpoint has no EMA state.")
    return ema["ema_model"]


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    run_config = parse_config(argv)
    torch.manual_seed(int(run_config["seed"]))
    torch.cuda.manual_seed_all(int(run_config["seed"]))
    requested_device = run_config["device"]
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    base_checkpoint_path = _path(run_config["base_checkpoint"])
    output_dir = _path(run_config["output_dir"])
    data_dir = _path(run_config["data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        base_checkpoint_path, map_location="cpu", weights_only=False
    )
    base_checkpoint_sha256 = _sha256(base_checkpoint_path)
    base_config = dict(checkpoint["config"])
    if base_config.get("model_type") != "dit_tiny":
        raise ValueError("This LoRA recipe requires a DiT-Tiny base checkpoint.")

    model = build_model(base_config)
    state_source = "ema" if run_config["base_use_ema"] else "model"
    model.load_state_dict(_base_state(checkpoint, run_config["base_use_ema"]), strict=True)
    injected_modules = inject_lora(
        model,
        target_modules=run_config["lora_target_modules"],
        rank=int(run_config["lora_rank"]),
        alpha=float(run_config["lora_alpha"]),
        dropout=float(run_config["lora_dropout"]),
    )
    assert_only_lora_trainable(model)
    total_parameters, trainable_parameters = _parameter_counts(model)

    dataset, channels, image_size, num_classes = build_dataset(
        base_config["dataset"], str(data_dir)
    )
    if channels != base_config["in_channels"] or image_size != base_config["image_size"]:
        raise ValueError("Dataset shape does not match the base checkpoint.")
    loader = DataLoader(
        dataset,
        batch_size=int(run_config["batch_size"]),
        shuffle=True,
        num_workers=int(run_config["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    objective = build_objective(base_config)
    optimizer = torch.optim.AdamW(
        lora_parameters(model),
        lr=float(run_config["lr"]),
        weight_decay=float(run_config["weight_decay"]),
    )

    effective_config = {
        **base_config,
        **run_config,
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_global_step": int(checkpoint["global_step"]),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_state_source": state_source,
        "output_dir": str(output_dir),
        "data_dir": str(data_dir),
        "device": str(device),
        "num_classes": num_classes,
        "adapter_type": "lora",
        "lora_injected_modules": injected_modules,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
    }
    trainer = LoggingTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=None,
        train_loader=loader,
        device=device,
        output_dir=output_dir,
        mixed_precision=run_config["mixed_precision"],
        grad_clip=run_config["grad_clip"],
        log_every=run_config["log_every"],
        save_every=run_config["save_every"],
        config=effective_config,
        ema=None,
        cond_drop_prob=(
            base_config["cond_drop_prob"] if base_config["class_cond"] else 0.0
        ),
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    trainer.train(
        num_epochs=int(run_config["num_epochs"]),
        max_steps=int(run_config["max_steps"]),
    )
    lora_names = set(lora_parameter_names(trainer.model))
    changed_names = [
        name
        for name, parameter in trainer.model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    ]
    changed_non_lora = sorted(set(changed_names) - lora_names)
    if not changed_names or changed_non_lora:
        raise RuntimeError(
            "LoRA update invariant failed: "
            f"changed={changed_names}, changed_non_lora={changed_non_lora}."
        )

    adapter_dir = output_dir / "adapter"
    save_lora_adapter(
        trainer.model,
        adapter_dir,
        target_modules=run_config["lora_target_modules"],
        rank=int(run_config["lora_rank"]),
        alpha=float(run_config["lora_alpha"]),
        dropout=float(run_config["lora_dropout"]),
        base_model={
            "model_type": "dit_tiny",
            "checkpoint": str(base_checkpoint_path),
            "checkpoint_global_step": int(checkpoint["global_step"]),
            "checkpoint_sha256": base_checkpoint_sha256,
            "state_source": state_source,
            "config": base_config,
        },
    )

    trainer.model.eval()
    comparison_generator = torch.Generator(device=device).manual_seed(
        int(run_config["seed"]) + 1
    )
    fixed_x = torch.randn(
        4,
        base_config["in_channels"],
        base_config["image_size"],
        base_config["image_size"],
        generator=comparison_generator,
        device=device,
    )
    fixed_t = torch.linspace(0.1, 0.9, 4, device=device)
    fixed_cond = None
    if base_config["class_cond"]:
        fixed_cond = torch.arange(4, device=device) % num_classes
    with torch.no_grad():
        expected = trainer.model(fixed_x, fixed_t, cond=fixed_cond)

    reloaded = build_model(base_config)
    reloaded.load_state_dict(_base_state(checkpoint, run_config["base_use_ema"]), strict=True)
    reloaded.to(device)
    adapter_metadata = load_lora_adapter(reloaded, adapter_dir, strict=True)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(fixed_x, fixed_t, cond=fixed_cond)
    reload_max_abs_error = float((expected - actual).abs().max().cpu())
    if not torch.equal(expected, actual):
        raise RuntimeError(
            "Reloaded adapter output is not bitwise identical; "
            f"max_abs_error={reload_max_abs_error}."
        )

    result = {
        "status": "completed",
        "base_checkpoint": str(base_checkpoint_path),
        "base_checkpoint_global_step": int(checkpoint["global_step"]),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_state_source": state_source,
        "finetune_steps": trainer.global_step,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "injected_module_count": len(injected_modules),
        "lora_tensor_count": len(adapter_metadata["tensor_names"]),
        "changed_parameter_names": sorted(changed_names),
        "changed_non_lora": changed_non_lora,
        "reload_max_abs_error": reload_max_abs_error,
        "reload_output_identical": True,
        "adapter_directory": str(adapter_dir),
        "training_checkpoint": str(output_dir / "checkpoints" / "last.pt"),
        "step_log": trainer.step_log,
    }
    with (output_dir / "lora_run.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
