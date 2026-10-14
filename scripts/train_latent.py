from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.train import build_dataset, build_objective, build_scheduler
from ugm.models.adapters.vae_adapter import FrozenAutoencoderKL
from ugm.models.build import build_model
from ugm.pipelines.latent_diffusion import LatentDiffusionPipeline
from ugm.samplers.ddim import DDIMSampler
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.samplers.ode import ODESampler
from ugm.trainers.latent_trainer import LatentDiffusionTrainer
from ugm.utils.config import DEFAULT_CONFIG, load_yaml_config, merge_configs, validate_config
from ugm.utils.ema import EMA


LATENT_DEFAULTS: dict[str, Any] = {
    **DEFAULT_CONFIG,
    "vae_model": "stabilityai/sd-vae-ft-mse",
    "vae_subfolder": None,
    "vae_revision": "31f26fdeee1355a5c34592e401dd41e45d25a493",
    "vae_cache_dir": "outputs/huggingface_cache",
    "vae_local_files_only": False,
    "vae_image_size": 32,
    "vae_resize_mode": "pad",
    "vae_scaling_factor": None,
    "sample_posterior": True,
    "sample_batch_size": 16,
    "sample_num_steps": 20,
    "sample_sampler": "ddim",
    "sample_eta": 0.0,
    "use_ema_for_sampling": True,
    "device": "auto",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a latent DDPM/FM model through a frozen pretrained VAE.",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--resume")
    parser.add_argument("--vae_model")
    parser.add_argument("--vae_cache_dir")
    parser.add_argument(
        "--vae_local_files_only", action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--sample_batch_size", type=int)
    parser.add_argument("--sample_num_steps", type=int)
    parser.add_argument(
        "--sample_sampler", choices=["ddpm", "ddim", "euler", "heun"]
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    return parser


def parse_config(argv: list[str] | None = None) -> dict[str, Any]:
    parsed = vars(build_parser().parse_args(argv))
    config_path = Path(parsed.pop("config"))
    config = merge_configs(
        defaults=LATENT_DEFAULTS,
        yaml_config=load_yaml_config(config_path),
        cli_overrides=parsed,
    )
    validate_config(config)
    if config["vae_resize_mode"] not in {"pad", "resize"}:
        raise ValueError("vae_resize_mode must be 'pad' or 'resize'.")
    if config["sample_batch_size"] <= 0 or config["sample_num_steps"] <= 0:
        raise ValueError("sample batch size and step count must be positive.")
    config["config_path"] = str(config_path.resolve())
    return config


def _resolve_path(value: str | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _build_sampler(config: dict[str, Any]):
    name = config["sample_sampler"]
    if config["objective"] == "fm":
        if name not in {"euler", "heun"}:
            raise ValueError("Flow Matching latent sampling requires euler or heun.")
        return ODESampler(
            num_steps=config["sample_num_steps"],
            t_start=config["t_min"],
            t_end=config["t_max"],
            solver=name,
        )
    if name == "ddpm":
        return DDPMAncestralSampler()
    if name == "ddim":
        return DDIMSampler(
            num_sampling_steps=config["sample_num_steps"],
            eta=config["sample_eta"],
        )
    raise ValueError("DDPM-objective latent sampling requires ddpm or ddim.")


def _save_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nrow = max(1, int(images.shape[0] ** 0.5))
    save_image((images.detach().float().cpu() + 1.0) * 0.5, path, nrow=nrow)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    config = parse_config(argv)
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    if config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = _resolve_path(
        config["output_dir"], PROJECT_ROOT / "outputs" / "latent_diffusion"
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = _resolve_path(config["data_dir"])
    cache_dir = _resolve_path(config["vae_cache_dir"])
    if data_dir is None:
        raise ValueError("data_dir is required.")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    dataset, pixel_channels, pixel_size, num_classes = build_dataset(
        config["dataset"], str(data_dir)
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    load_kwargs: dict[str, Any] = {
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "use_safetensors": True,
    }
    if config["vae_revision"] is not None:
        load_kwargs["revision"] = config["vae_revision"]
    vae = FrozenAutoencoderKL.from_pretrained(
        config["vae_model"],
        subfolder=config["vae_subfolder"],
        image_channels=pixel_channels,
        image_size=pixel_size,
        vae_image_size=config["vae_image_size"],
        resize_mode=config["vae_resize_mode"],
        scaling_factor=config["vae_scaling_factor"],
        local_files_only=config["vae_local_files_only"],
        **load_kwargs,
    ).to(device)

    # From this point onward, image_size/in_channels describe model-space
    # latents. Pixel-space metadata is retained under explicit pixel_* names.
    config.update(
        {
            "device": str(device),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "vae_cache_dir": None if cache_dir is None else str(cache_dir),
            "pixel_in_channels": pixel_channels,
            "pixel_image_size": pixel_size,
            "in_channels": vae.latent_channels,
            "out_channels": vae.latent_channels,
            "image_size": vae.latent_size,
            "num_classes": num_classes,
            "vae_resolved_scaling_factor": vae.scaling_factor,
            "vae_downsample_factor": vae.downsample_factor,
            "vae_frozen": True,
        }
    )

    model = build_model(config).to(device)
    objective = build_objective(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids.intersection(id(p) for p in vae.parameters()):
        raise RuntimeError("Frozen VAE parameters must not enter the optimizer.")
    scheduler = build_scheduler(config, optimizer, len(loader))
    ema = EMA(
        model,
        decay=config["ema_decay"],
        update_after_step=config["ema_update_after_step"],
        update_every=config["ema_update_every"],
    )
    trainer = LatentDiffusionTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        vae=vae,
        sample_posterior=config["sample_posterior"],
        device=device,
        output_dir=output_dir,
        mixed_precision=config["mixed_precision"],
        grad_clip=config["grad_clip"],
        log_every=config["log_every"],
        save_every=config["save_every"],
        config=config,
        ema=ema,
        cond_drop_prob=config["cond_drop_prob"] if config["class_cond"] else 0.0,
    )
    if config["resume"] is not None:
        resume_path = _resolve_path(config["resume"])
        if resume_path is None or not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        trainer.load_checkpoint(resume_path)

    raw_images, raw_labels = next(iter(loader))
    preview_count = min(int(config["sample_batch_size"]), raw_images.shape[0])
    preview_images = raw_images[:preview_count].to(device)
    with torch.no_grad():
        preview_latents = vae.encode(preview_images, sample_posterior=False)
        reconstructed = vae.decode(preview_latents)
    sample_dir = output_dir / "samples"
    _save_grid(preview_images, sample_dir / "input.png")
    _save_grid(reconstructed, sample_dir / "vae_reconstruction.png")

    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    trainer.train(config["num_epochs"], max_steps=config["max_steps"])
    parameter_max_change = max(
        float((parameter.detach() - before[name]).abs().max().cpu())
        for name, parameter in model.named_parameters()
    )

    sample_model = ema.ema_model if config["use_ema_for_sampling"] else model
    sample_model.to(device)
    pipeline = LatentDiffusionPipeline(
        vae=vae,
        model=sample_model,
        sampler=_build_sampler(config),
        objective=objective,
    )
    sample_batch_size = int(config["sample_batch_size"])
    conditions = None
    if config["class_cond"]:
        conditions = torch.arange(sample_batch_size, device=device) % num_classes
    initial_noise = torch.randn(
        sample_batch_size,
        vae.latent_channels,
        vae.latent_size,
        vae.latent_size,
        device=device,
    )
    generated = pipeline.sample(
        batch_size=sample_batch_size,
        cond=conditions,
        guidance_scale=config["guidance_scale"],
        initial_noise=initial_noise,
        num_steps=config["sample_num_steps"],
    )
    _save_grid(generated.images, sample_dir / "generated.png")
    torch.save(initial_noise.cpu(), sample_dir / "initial_noise.pt")

    vae_has_gradient = any(parameter.grad is not None for parameter in vae.parameters())
    manifest = {
        "status": "completed",
        "global_step": trainer.global_step,
        "loss": trainer.last_metrics.get("loss"),
        "parameter_max_abs_change": parameter_max_change,
        "vae_frozen": all(not p.requires_grad for p in vae.parameters()),
        "vae_has_gradient": vae_has_gradient,
        "image_shape": list(preview_images.shape),
        "latent_shape": list(preview_latents.shape),
        "generated_latent_shape": list(generated.latents.shape),
        "decoded_shape": list(generated.images.shape),
        "latent_scaling_factor": vae.scaling_factor,
        "sample_sampler": config["sample_sampler"],
        "sample_num_steps": config["sample_num_steps"],
        "checkpoint": str(output_dir / "checkpoints" / "last.pt"),
        "artifacts": {
            "input": str(sample_dir / "input.png"),
            "reconstruction": str(sample_dir / "vae_reconstruction.png"),
            "generated": str(sample_dir / "generated.png"),
            "initial_noise": str(sample_dir / "initial_noise.pt"),
        },
    }
    if not torch.isfinite(generated.images).all():
        raise FloatingPointError("Decoded generated images contain non-finite values.")
    if vae_has_gradient or parameter_max_change == 0.0:
        raise RuntimeError("Freeze/update invariant failed; see run manifest.")
    with (output_dir / "latent_run.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
