from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare an FM teacher and RF student under one sampling protocol."
    )
    parser.add_argument("--fm_ckpt", type=Path, required=True)
    parser.add_argument("--rf_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 10, 20, 50])
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--num_reference", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--metric_batch_size", type=int, default=64)
    parser.add_argument("--kid_subsets", type=int, default=20)
    parser.add_argument("--kid_subset_size", type=int, default=256)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid_samples", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow_pair_seed_overlap",
        action="store_true",
        help="Allow evaluation noise to reuse the RF pair-generation seed.",
    )
    return parser


def _resolve(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _run_evaluation(
    *,
    checkpoint: Path,
    output_dir: Path,
    steps: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result_path = output_dir / "results.json"
    if result_path.is_file() and not args.force:
        with result_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    command = [
        sys.executable,
        "-B",
        str(PROJECT_ROOT / "scripts" / "evaluate.py"),
        "--ckpt",
        str(checkpoint),
        "--output_dir",
        str(output_dir),
        "--sampler",
        "euler",
        "--num_steps",
        str(steps),
        "--num_samples",
        str(args.num_samples),
        "--num_reference",
        str(args.num_reference),
        "--batch_size",
        str(args.batch_size),
        "--metric_batch_size",
        str(args.metric_batch_size),
        "--kid_subsets",
        str(args.kid_subsets),
        "--kid_subset_size",
        str(args.kid_subset_size),
        "--warmup_runs",
        str(args.warmup_runs),
        "--repetitions",
        str(args.repetitions),
        "--benchmark_batch_size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--metrics",
        "fid",
        "kid",
        "pixel_fd",
        "--no-use_ema",
        "--overwrite",
    ]
    if args.device is not None:
        command.extend(("--device", args.device))
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluate.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    if completed.stderr:
        (output_dir / "evaluate.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
    with result_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _row(model_name: str, steps: int, result: dict[str, Any]) -> dict[str, Any]:
    performance = result["performance"]
    standard = result["metrics"]["standard"]
    kid = standard["kid"]
    return {
        "model": model_name,
        "steps": steps,
        "nfe": performance["nfe_per_sample_trajectory"],
        "nfe_verified": performance["nfe_verified"],
        "benchmark_seconds_mean": performance["elapsed_seconds"]["mean"],
        "benchmark_seconds_std": performance["elapsed_seconds"]["std_population"],
        "throughput_mean": performance["samples_per_second"]["mean"],
        "pixel_fd_proxy": result["metrics"]["pixel_fd_proxy"]["value"],
        "fid": standard["fid"]["value"],
        "kid_mean": kid["mean"],
        "kid_std": kid["std"],
        "checkpoint": result["run"]["checkpoint"],
        "checkpoint_global_step": result["run"]["checkpoint_global_step"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    header = (
        "| Model | Steps/NFE | Time mean±std (s, batch) | samples/s | "
        "FID ↓ | KID mean±std ↓ | pixel-FD proxy ↓ |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['steps']}/{row['nfe']} | "
            f"{row['benchmark_seconds_mean']:.4f}±{row['benchmark_seconds_std']:.4f} | "
            f"{row['throughput_mean']:.2f} | {row['fid']:.4f} | "
            f"{row['kid_mean']:.6f}±{row['kid_std']:.6f} | "
            f"{row['pixel_fd_proxy']:.6f} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def _create_grid(
    root: Path,
    steps: list[int],
    count: int,
    output_path: Path,
) -> None:
    labels = [(model, step) for model in ("FM", "RF") for step in steps]
    first_path = root / "fm" / f"steps_{steps[0]}" / "generated" / "000000.png"
    with Image.open(first_path) as first:
        tile_width, tile_height = first.size
    scale = 2
    label_width = 115
    row_height = tile_height * scale
    canvas = Image.new(
        "L",
        (label_width + count * tile_width * scale, len(labels) * row_height),
        color=255,
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (model, step) in enumerate(labels):
        run_dir = root / model.lower() / f"steps_{step}" / "generated"
        y = row_index * row_height
        draw.text((5, y + row_height // 3), f"{model} {step} step", fill=0)
        for sample_index in range(count):
            with Image.open(run_dir / f"{sample_index:06d}.png") as image:
                tile = image.convert("L").resize(
                    (tile_width * scale, tile_height * scale),
                    Image.Resampling.NEAREST,
                )
            canvas.paste(tile, (label_width + sample_index * tile_width * scale, y))
    canvas.save(output_path)


def _rf_pair_seed(checkpoint: Path) -> int | None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    teacher_sampling = config.get("teacher_sampling")
    if isinstance(teacher_sampling, dict) and teacher_sampling.get("seed") is not None:
        return int(teacher_sampling["seed"])
    manifest_path = config.get("pair_manifest")
    if manifest_path is not None and Path(manifest_path).is_file():
        with Path(manifest_path).open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        generation = manifest.get("generation", {})
        if generation.get("seed") is not None:
            return int(generation["seed"])
    return None


def main() -> None:
    args = build_parser().parse_args()
    if not args.steps or any(step <= 0 for step in args.steps):
        raise ValueError("All requested steps must be positive.")
    if args.grid_samples <= 0 or args.grid_samples > args.num_samples:
        raise ValueError("grid_samples must be in [1, num_samples].")
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        "FM": _resolve(args.fm_ckpt),
        "RF": _resolve(args.rf_ckpt),
    }
    pair_seed = _rf_pair_seed(checkpoints["RF"])
    if pair_seed == args.seed and not args.allow_pair_seed_overlap:
        raise ValueError(
            "Evaluation seed matches the RF pair-generation seed. This can "
            "reuse training noise endpoints; choose a disjoint --seed or pass "
            "--allow_pair_seed_overlap only for a deliberate memorization test."
        )
    rows: list[dict[str, Any]] = []
    for model_name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for step in args.steps:
            run_dir = output_dir / model_name.lower() / f"steps_{step}"
            print(f"Evaluating {model_name} at {step} Euler steps...")
            result = _run_evaluation(
                checkpoint=checkpoint,
                output_dir=run_dir,
                steps=step,
                args=args,
            )
            rows.append(_row(model_name, step, result))

    by_key = {(row["model"], row["steps"]): row for row in rows}
    deltas = []
    for step in args.steps:
        fm = by_key[("FM", step)]
        rf = by_key[("RF", step)]
        deltas.append(
            {
                "steps": step,
                "rf_minus_fm_fid": rf["fid"] - fm["fid"],
                "rf_minus_fm_kid_mean": rf["kid_mean"] - fm["kid_mean"],
                "rf_minus_fm_pixel_fd": rf["pixel_fd_proxy"] - fm["pixel_fd_proxy"],
                "rf_minus_fm_time_seconds": (
                    rf["benchmark_seconds_mean"] - fm["benchmark_seconds_mean"]
                ),
            }
        )
    comparison = {
        "schema_version": 1,
        "protocol": {
            "sampler": "euler",
            "steps": args.steps,
            "num_samples": args.num_samples,
            "num_reference": args.num_reference,
            "seed": args.seed,
            "rf_pair_generation_seed": pair_seed,
            "evaluation_noise_disjoint_from_pair_seed": pair_seed != args.seed,
            "sampling_model": "online for both checkpoints",
            "metrics": ["fid", "kid", "pixel_fd_proxy"],
            "warning": (
                "This is a small-sample first-pass comparison. FID/KID have high "
                "finite-sample uncertainty, and ImageNet features are mismatched to MNIST."
            ),
        },
        "rows": rows,
        "paired_deltas": deltas,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_csv(output_dir / "comparison.csv", rows)
    _write_markdown(output_dir / "comparison.md", rows)
    _create_grid(
        output_dir,
        args.steps,
        min(args.grid_samples, args.num_samples),
        output_dir / "comparison_grid.png",
    )
    print((output_dir / "comparison.md").read_text(encoding="utf-8"))
    print(f"Saved comparison to: {output_dir}")


if __name__ == "__main__":
    main()
