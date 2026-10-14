from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-noise 1/2/4-step consistency sample grids."
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--class_label", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument(
        "--consistency_model",
        choices=["target", "student"],
        default="target",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_config(checkpoint_path: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("config")
    if isinstance(config, dict) and config:
        return dict(config)
    for filename in ("config.json", "args.json"):
        candidate = checkpoint_path.parent.parent / filename
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as file:
                return json.load(file)
    raise FileNotFoundError(f"Cannot locate config for {checkpoint_path}.")


def _combine_grids(
    paths: list[tuple[int, int, Path]],
    output: Path,
) -> None:
    images = []
    for steps, nfe, path in paths:
        with Image.open(path) as image:
            images.append((steps, nfe, image.convert("L").copy()))
    label_height = 24
    width = max(image.width for _, _, image in images)
    height = sum(image.height + label_height for _, _, image in images)
    canvas = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(canvas)
    y = 0
    for steps, nfe, image in images:
        draw.text((5, y + 5), f"{steps}-step / NFE {nfe}", fill=0)
        y += label_height
        canvas.paste(image, (0, y))
        y += image.height
    canvas.save(output)


def main() -> None:
    args = build_parser().parse_args()
    if args.num_samples <= 0 or args.nrow <= 0:
        raise ValueError("num_samples and nrow must be positive.")
    checkpoint_path = _resolve(args.ckpt)
    output_dir = _resolve(args.output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [output_dir / f"samples_{steps}_step.png" for steps in (1, 2, 4)]
    expected.extend(
        [output_dir / "initial_noise.pt", output_dir / "manifest.json", output_dir / "comparison_grid.png"]
    )
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Consistency sample outputs already exist in {output_dir}; pass --overwrite."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint.")
    config = _load_config(checkpoint_path, checkpoint)
    if str(config.get("objective", "")) != "consistency_distillation":
        raise ValueError("sample_consistency.py requires a consistency checkpoint.")
    channels = int(config["in_channels"])
    image_size = int(config["image_size"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial_noise = torch.randn(
        args.num_samples,
        channels,
        image_size,
        image_size,
        generator=generator,
    )
    initial_noise_path = output_dir / "initial_noise.pt"
    torch.save(initial_noise, initial_noise_path)

    cfg_branches = (
        2
        if bool(config.get("class_cond", False))
        and args.guidance_scale not in {0.0, 1.0}
        else 1
    )
    outputs: list[tuple[int, int, Path]] = []
    logs: dict[str, str] = {}
    for steps in (1, 2, 4):
        output_path = output_dir / f"samples_{steps}_step.png"
        command = [
            sys.executable,
            "-B",
            str(PROJECT_ROOT / "scripts" / "sample.py"),
            "--ckpt",
            str(checkpoint_path),
            "--output",
            str(output_path),
            "--num_samples",
            str(args.num_samples),
            "--nrow",
            str(args.nrow),
            "--sampler",
            "consistency",
            "--num_steps",
            str(steps),
            "--consistency_model",
            args.consistency_model,
            "--initial_noise",
            str(initial_noise_path),
            "--seed",
            str(args.seed),
            "--guidance_scale",
            str(args.guidance_scale),
        ]
        if args.device is not None:
            command.extend(("--device", args.device))
        if args.class_label is not None:
            command.extend(("--class_label", str(args.class_label)))
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        logs[str(steps)] = completed.stdout
        outputs.append((steps, steps * cfg_branches, output_path))

    comparison_path = output_dir / "comparison_grid.png"
    _combine_grids(outputs, comparison_path)
    manifest = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "sampling_model": args.consistency_model,
        "seed": args.seed,
        "fixed_initial_noise": str(initial_noise_path),
        "num_samples": args.num_samples,
        "class_label": args.class_label,
        "guidance_scale": args.guidance_scale,
        "budgets": [
            {"steps": steps, "nfe": nfe, "output": str(path)}
            for steps, nfe, path in outputs
        ],
        "algorithm": (
            "Evaluate f(x,t)=x+(1-t)F(x,t) at t_i=i/K. Between "
            "evaluations, re-noise the previous data estimate as "
            "x_t=(1-t)z+t*x1_estimate using seeded fresh Gaussian noise."
        ),
        "logs": logs,
        "comparison_grid": str(comparison_path),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print(f"Saved 1/2/4-step consistency comparison to: {output_dir}")


if __name__ == "__main__":
    main()
