from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.sample import (  # noqa: E402
    build_condition,
    build_model_from_config,
    build_objective_from_config,
    denormalize_to_01,
    infer_sample_shape,
    load_config_from_checkpoint_dir,
)
from ugm.models.conditioning import validate_guidance_scale  # noqa: E402
from ugm.samplers.ddim import DDIMSampler  # noqa: E402
from ugm.samplers.ddpm import DDPMAncestralSampler  # noqa: E402
from ugm.utils.ema import EMA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DDPM and DDIM sampling speed and sample quality. "
            "Quality is reported as a low-resolution pixel Frechet "
            "distance proxy, not standard feature-space FID."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("outputs/ddpm_mnist/checkpoints/last.pt"),
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="DDIM step counts to benchmark.",
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_reference", type=int, default=1000)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_label", type=int, default=None)
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="CFG scale; defaults to the checkpoint's saved config.",
    )
    parser.add_argument(
        "--use_ema",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--clip_x0",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/ddpm_mnist/sampler_comparison"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, num_train_steps: int) -> None:
    if args.num_samples <= 1:
        raise ValueError("--num_samples must be greater than 1.")
    if args.num_reference <= 1:
        raise ValueError("--num_reference must be greater than 1.")
    if args.eta < 0.0:
        raise ValueError("--eta must be non-negative.")
    if not args.ddim_steps:
        raise ValueError("At least one DDIM step count is required.")
    invalid_steps = [
        step
        for step in args.ddim_steps
        if step <= 0 or step > num_train_steps
    ]
    if invalid_steps:
        raise ValueError(
            "DDIM steps must be in [1, "
            f"{num_train_steps}], got {invalid_steps}."
        )


def load_sampling_components(
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool,
):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    config = load_config_from_checkpoint_dir(ckpt_path)
    if str(config.get("objective", "")).lower() != "ddpm":
        raise ValueError("This comparison script requires a DDPM checkpoint.")

    model = build_model_from_config(config).to(device)
    objective = build_objective_from_config("ddpm", config).to(device)
    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("objective") is not None:
        objective.load_state_dict(checkpoint["objective"], strict=True)

    sampling_model = model
    if use_ema:
        ema_state = checkpoint.get("ema")
        if ema_state is None:
            raise KeyError(
                "--use_ema was requested, but the checkpoint has no EMA state."
            )
        ema = EMA(model=model)
        ema.load_state_dict(ema_state)
        sampling_model = ema.ema_model.to(device)

    sampling_model.eval()
    objective.eval()
    return config, sampling_model, objective


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def warm_up_model(
    model: torch.nn.Module,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    """Pay one-time CUDA/cuDNN setup cost before measuring samplers."""
    x = torch.randn(shape, device=device)
    t = torch.zeros(shape[0], device=device)
    model(x, t, cond=None)
    synchronize_cuda(device)


@torch.inference_mode()
def timed_sample(
    sampler,
    model: torch.nn.Module,
    objective: torch.nn.Module,
    shape: tuple[int, ...],
    device: torch.device,
    initial_noise: torch.Tensor,
    num_steps: int,
    cond: torch.Tensor | None,
    guidance_scale: float,
) -> tuple[torch.Tensor, float]:
    # CUDA kernels are asynchronous, so synchronize on both sides of the
    # sampler call to measure actual wall-clock sampling time.
    synchronize_cuda(device)
    start_time = time.perf_counter()

    samples = sampler.sample(
        model=model,
        objective=objective,
        shape=shape,
        cond=cond,
        guidance_scale=guidance_scale,
        device=device,
        initial_noise=initial_noise,
        num_steps=num_steps,
    )

    synchronize_cuda(device)
    elapsed = time.perf_counter() - start_time

    print(f"elapsed={elapsed:.3f}s")
    print(f"samples_per_second={shape[0] / elapsed:.3f}")
    return samples, elapsed


def image_features(images: torch.Tensor, feature_size: int = 7) -> torch.Tensor:
    """Create small pixel features for a lightweight quality proxy."""
    images = images.detach().float().cpu()
    if images.shape[1] != 1:
        images = images.mean(dim=1, keepdim=True)
    features = F.adaptive_avg_pool2d(images, (feature_size, feature_size))
    return features.flatten(start_dim=1).double()


def mean_and_covariance(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    return mean, covariance


def symmetric_matrix_sqrt(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.T


def pixel_frechet_distance(
    generated: torch.Tensor,
    reference_features: torch.Tensor,
) -> float:
    """Low-resolution pixel FD proxy. Lower is better; this is not FID."""
    generated_features = image_features(generated)
    generated_mean, generated_cov = mean_and_covariance(generated_features)
    reference_mean, reference_cov = mean_and_covariance(reference_features)

    eps = 1e-6
    identity = torch.eye(generated_cov.shape[0], dtype=torch.float64)
    generated_cov = generated_cov + eps * identity
    reference_cov = reference_cov + eps * identity

    sqrt_generated_cov = symmetric_matrix_sqrt(generated_cov)
    middle = sqrt_generated_cov @ reference_cov @ sqrt_generated_cov
    covariance_mean = symmetric_matrix_sqrt(
        (middle + middle.T) * 0.5
    )

    mean_difference = generated_mean - reference_mean
    distance = (
        mean_difference.dot(mean_difference)
        + torch.trace(generated_cov)
        + torch.trace(reference_cov)
        - 2.0 * torch.trace(covariance_mean)
    )
    return float(distance.clamp_min(0.0).item())


def load_mnist_reference(
    data_dir: Path,
    num_reference: int,
    seed: int,
) -> torch.Tensor:
    dataset = MNIST(root=data_dir, train=False, download=False)
    count = min(num_reference, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count]
    images = dataset.data[indices].unsqueeze(1).float() / 255.0
    return image_features(images)


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| Sampler | Sampling steps | Time (s) | Samples/s | "
        "Pixel-FD proxy (lower is better) | Sample grid |"
    )
    separator = "|---|---:|---:|---:|---:|---|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            "| {sampler} | {sampling_steps} | {time_seconds:.3f} | "
            "{samples_per_second:.3f} | {pixel_fd_proxy:.4f} | "
            "[{sample_grid}]({sample_grid}) |".format(**row)
        )
    return "\n".join(lines) + "\n"


def save_results(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_dir / "comparison.md"
    markdown_path.write_text(
        "# DDPM/DDIM sampler comparison\n\n"
        "Pixel-FD proxy is computed from 7x7 grayscale pixels against real "
        "MNIST samples. Lower is better. It is a lightweight comparison "
        "metric, not standard feature-space FID.\n\n"
        + markdown_table(rows),
        encoding="utf-8",
    )

    print("\n" + markdown_table(rows))
    print(f"CSV: {csv_path}")
    print(f"Markdown: {markdown_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    config, model, objective = load_sampling_components(
        args.ckpt,
        device,
        args.use_ema,
    )
    validate_args(args, objective.num_timesteps)

    if str(config.get("dataset", "")).lower() != "mnist":
        raise ValueError(
            "Pixel-FD reference loading currently supports MNIST only."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_features = load_mnist_reference(
        PROJECT_ROOT / str(config.get("data_dir", "data")),
        args.num_reference,
        args.seed,
    )

    shape = infer_sample_shape(config, args.num_samples)
    warm_up_model(model, shape, device)
    cond = build_condition(
        config=config,
        num_samples=args.num_samples,
        device=device,
        class_label=args.class_label,
    )
    guidance_scale = validate_guidance_scale(
        args.guidance_scale
        if args.guidance_scale is not None
        else config.get("guidance_scale", 1.0)
    )

    # Every solver starts from exactly the same x_T. Resetting the seed alone
    # is less explicit and can become unfair if setup code consumes randomness.
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    initial_noise = torch.randn(shape, device=device)

    experiments = [
        (
            "DDPM",
            objective.num_timesteps,
            DDPMAncestralSampler(clip_x0=args.clip_x0),
        )
    ]
    experiments.extend(
        (
            "DDIM",
            step_count,
            DDIMSampler(
                num_sampling_steps=step_count,
                eta=args.eta,
                clip_x0=args.clip_x0,
            ),
        )
        for step_count in dict.fromkeys(args.ddim_steps)
    )

    rows: list[dict[str, object]] = []
    for sampler_name, step_count, sampler in experiments:
        print(f"\nRunning {sampler_name} ({step_count} steps)...")
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        samples, elapsed = timed_sample(
            sampler=sampler,
            model=model,
            objective=objective,
            shape=shape,
            device=device,
            initial_noise=initial_noise,
            num_steps=step_count,
            cond=cond,
            guidance_scale=guidance_scale,
        )
        samples_01 = denormalize_to_01(samples)
        grid_name = f"{sampler_name.lower()}_{step_count}_steps.png"
        grid_path = args.output_dir / grid_name
        save_image(samples_01, grid_path, nrow=args.nrow)

        rows.append(
            {
                "sampler": sampler_name,
                "sampling_steps": step_count,
                "time_seconds": elapsed,
                "samples_per_second": args.num_samples / elapsed,
                "pixel_fd_proxy": pixel_frechet_distance(
                    samples_01,
                    reference_features,
                ),
                "sample_grid": grid_name,
            }
        )

        del samples, samples_01
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_results(rows, args.output_dir)


if __name__ == "__main__":
    main()
