from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ugm.data.rectified_flow import (  # noqa: E402
    PAIR_DATASET_FORMAT,
    PAIR_DATASET_VERSION,
)
from ugm.models.build import build_model  # noqa: E402
from ugm.samplers.ode import ODESampler  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate noise/teacher-output pairs for Rectified Flow."
    )
    parser.add_argument("--teacher_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--shard_size", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--solver", choices=["euler", "heun"], default="heun")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--label_policy",
        choices=["balanced", "random"],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_checkpoint_config(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    saved = checkpoint.get("config")
    if isinstance(saved, dict) and saved:
        return dict(saved)
    output_dir = checkpoint_path.parent.parent
    for filename in ("config.json", "args.json"):
        path = output_dir / filename
        if path.is_file():
            with path.open("r", encoding="utf-8") as file:
                loaded = json.load(file)
            if not isinstance(loaded, dict):
                raise TypeError(f"Expected a JSON object in {path}.")
            return loaded
    raise FileNotFoundError(
        "The checkpoint has no embedded config and no config.json/args.json "
        f"was found beside it: {checkpoint_path}"
    )


def build_teacher(
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    config = load_checkpoint_config(checkpoint_path, checkpoint)
    objective = str(config.get("objective", "")).lower()
    if objective != "fm":
        raise ValueError(
            f"Rectified-flow pairs require an FM teacher, got {objective!r}."
        )
    model = build_model(config)
    if use_ema:
        ema_state = checkpoint.get("ema")
        if not isinstance(ema_state, dict) or "ema_model" not in ema_state:
            raise KeyError(
                "--use_ema was requested, but the teacher checkpoint has no EMA."
            )
        state = ema_state["ema_model"]
        state_source = "ema"
    else:
        state = checkpoint["model"]
        state_source = "online"
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "state_source": state_source,
    }
    return model, config, metadata


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("pairs-*.pt"))
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        existing.append(manifest)
    if existing and not overwrite:
        raise FileExistsError(
            f"Paired-data files already exist in {output_dir}; pass --overwrite."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def _atomic_torch_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_pairs(args: argparse.Namespace) -> Path:
    if args.num_samples <= 0 or args.batch_size <= 0 or args.shard_size <= 0:
        raise ValueError("num_samples, batch_size, and shard_size must be positive.")
    if args.num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if args.guidance_scale < 0 or not torch.isfinite(torch.tensor(args.guidance_scale)):
        raise ValueError("guidance_scale must be finite and non-negative.")

    checkpoint_path = _resolve(args.teacher_ckpt)
    output_dir = _resolve(args.output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
    _prepare_output_dir(output_dir, args.overwrite)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    teacher, config, teacher_metadata = build_teacher(
        checkpoint_path,
        device,
        args.use_ema,
    )
    channels = int(config.get("in_channels", 1))
    image_size = int(
        config.get(
            "image_size",
            28 if str(config.get("dataset", "mnist")).lower() == "mnist" else 32,
        )
    )
    conditional = bool(config.get("class_cond", False))
    num_classes = int(config.get("num_classes", 10)) if conditional else None
    sampler = ODESampler(
        num_steps=args.num_steps,
        t_start=0.0,
        t_end=1.0,
        solver=args.solver,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    pending: dict[str, list[torch.Tensor]] = {"x0": [], "x1": [], "cond": []}
    pending_count = 0
    shards: list[dict[str, Any]] = []
    generated = 0
    shard_start = 0

    def flush() -> None:
        nonlocal pending, pending_count, shard_start
        if pending_count == 0:
            return
        shard_index = len(shards)
        filename = f"pairs-{shard_index:05d}.pt"
        payload = {
            "x0": torch.cat(pending["x0"], dim=0),
            "x1": torch.cat(pending["x1"], dim=0),
        }
        if conditional:
            payload["cond"] = torch.cat(pending["cond"], dim=0)
        _atomic_torch_save(payload, output_dir / filename)
        shards.append({"file": filename, "start": shard_start, "count": pending_count})
        shard_start += pending_count
        pending = {"x0": [], "x1": [], "cond": []}
        pending_count = 0

    while generated < args.num_samples:
        current = min(args.batch_size, args.num_samples - generated)
        noise_cpu = torch.randn(
            current,
            channels,
            image_size,
            image_size,
            generator=generator,
            dtype=torch.float32,
        )
        cond_cpu: torch.Tensor | None = None
        if conditional:
            if args.label_policy == "balanced":
                cond_cpu = torch.arange(generated, generated + current) % num_classes
            else:
                cond_cpu = torch.randint(
                    num_classes,
                    (current,),
                    generator=generator,
                )
            cond_cpu = cond_cpu.long()
        with torch.inference_mode():
            sample_output = sampler.sample(
                model=teacher,
                shape=tuple(noise_cpu.shape),
                cond=None if cond_cpu is None else cond_cpu.to(device),
                guidance_scale=float(args.guidance_scale),
                device=device,
                initial_noise=noise_cpu.to(device),
            )
        teacher_output = sample_output.samples if hasattr(sample_output, "samples") else sample_output
        teacher_output = teacher_output.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(teacher_output).all():
            raise FloatingPointError("Teacher produced non-finite endpoints.")
        pending["x0"].append(noise_cpu)
        pending["x1"].append(teacher_output)
        if cond_cpu is not None:
            pending["cond"].append(cond_cpu)
        pending_count += current
        generated += current
        if pending_count >= args.shard_size or generated == args.num_samples:
            flush()
        print(f"Generated {generated}/{args.num_samples} pairs")

    manifest = {
        "format": PAIR_DATASET_FORMAT,
        "version": PAIR_DATASET_VERSION,
        "num_samples": args.num_samples,
        "sample_shape": [channels, image_size, image_size],
        "dtype": "float32",
        "conditional": conditional,
        "num_classes": num_classes,
        "dataset": config.get("dataset"),
        "teacher": teacher_metadata,
        "generation": {
            "seed": args.seed,
            "solver": args.solver,
            "num_steps": args.num_steps,
            "guidance_scale": float(args.guidance_scale),
            "label_policy": args.label_policy if conditional else None,
        },
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_json_save(manifest, manifest_path)
    print(f"Saved {args.num_samples} paired samples to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    generate_pairs(build_parser().parse_args())


if __name__ == "__main__":
    main()
