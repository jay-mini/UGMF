from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.sample import (  # noqa: E402
    build_model_from_config,
    build_objective_from_config,
    denormalize_to_01,
    infer_sample_shape,
    load_config_from_checkpoint_dir,
)
from ugm.evaluation.fid import (  # noqa: E402
    calculate_pixel_fd_proxy,
    calculate_standard_metrics,
)
from ugm.evaluation.speed_quality import (  # noqa: E402
    CountingModel,
    benchmark_sampler,
    expected_nfe,
    synchronize,
)
from ugm.models.conditioning import validate_guidance_scale  # noqa: E402
from ugm.samplers.ddim import DDIMSampler  # noqa: E402
from ugm.samplers.ddpm import DDPMAncestralSampler  # noqa: E402
from ugm.samplers.ode import ODESampler  # noqa: E402
from ugm.utils.ema import EMA  # noqa: E402


SCHEMA_VERSION = 1
STANDARD_METRIC_NAMES = {"fid", "kid", "prc"}
ALL_METRIC_NAMES = STANDARD_METRIC_NAMES | {"pixel_fd"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export generated samples, benchmark sampling, and compute "
            "standard FID/KID/generative Precision-Recall plus pixel-FD."
        )
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--sampler",
        choices=("ddpm", "ddim", "euler", "heun"),
        default=None,
    )
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--num_reference", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--class_label", type=int, default=None)
    parser.add_argument(
        "--label_policy",
        choices=("balanced", "random"),
        default="balanced",
    )
    parser.add_argument(
        "--match_reference_condition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For a fixed class, filter real references to the same class.",
    )
    parser.add_argument(
        "--reference_split",
        choices=("train", "test"),
        default="test",
    )
    parser.add_argument(
        "--download_reference",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        "--metrics",
        nargs="+",
        choices=sorted(ALL_METRIC_NAMES),
        default=["fid", "kid", "prc", "pixel_fd"],
    )
    parser.add_argument("--metric_batch_size", type=int, default=64)
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--prc_neighborhood", type=int, default=3)
    parser.add_argument("--prc_batch_size", type=int, default=1000)
    parser.add_argument(
        "--metric_verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--benchmark_batch_size", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace PNG/CSV/JSON files in this evaluation run directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "num_samples",
        "batch_size",
        "metric_batch_size",
        "kid_subsets",
        "kid_subset_size",
        "prc_neighborhood",
        "prc_batch_size",
        "repetitions",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.num_reference is not None and args.num_reference <= 0:
        raise ValueError("--num_reference must be positive.")
    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("--num_steps must be positive.")
    if args.benchmark_batch_size is not None and args.benchmark_batch_size <= 0:
        raise ValueError("--benchmark_batch_size must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup_runs must be non-negative.")
    if args.eta < 0.0:
        raise ValueError("--eta must be non-negative.")
    if len(set(args.metrics)) != len(args.metrics):
        raise ValueError("--metrics must not contain duplicates.")


def load_sampling_components(
    ckpt_path: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[dict[str, Any], nn.Module, nn.Module, dict[str, Any]]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config")
    config = (
        dict(saved_config)
        if isinstance(saved_config, dict) and saved_config
        else load_config_from_checkpoint_dir(ckpt_path)
    )
    objective_name = str(config.get("objective", "")).lower()
    if objective_name not in {"ddpm", "fm", "rectified_flow"}:
        raise ValueError(f"Unsupported objective: {objective_name!r}.")

    model = build_model_from_config(config).to(device)
    objective = build_objective_from_config(objective_name, config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("objective") is not None:
        objective.load_state_dict(checkpoint["objective"], strict=True)

    sampling_model = model
    if use_ema:
        ema_state = checkpoint.get("ema")
        if ema_state is None:
            raise KeyError("EMA sampling requested, but checkpoint has no EMA state.")
        ema = EMA(model=model)
        ema.load_state_dict(ema_state)
        sampling_model = ema.ema_model.to(device)

    sampling_model.eval()
    objective.eval()
    return config, sampling_model, objective, checkpoint


def resolve_sampler(
    *,
    objective_name: str,
    requested_sampler: str | None,
    requested_steps: int | None,
    objective: nn.Module,
    eta: float,
    clip_x0: bool,
) -> tuple[str, int, object]:
    if objective_name == "ddpm":
        sampler_name = requested_sampler or "ddim"
        if sampler_name not in {"ddpm", "ddim"}:
            raise ValueError("DDPM objectives require the ddpm or ddim sampler.")
        train_steps = int(getattr(objective, "num_timesteps"))
        if sampler_name == "ddpm":
            steps = train_steps if requested_steps is None else requested_steps
            if steps != train_steps:
                raise ValueError(
                    "DDPM ancestral evaluation must use all training steps."
                )
            return sampler_name, steps, DDPMAncestralSampler(clip_x0=clip_x0)
        steps = 50 if requested_steps is None else requested_steps
        if steps > train_steps:
            raise ValueError("DDIM steps cannot exceed training timesteps.")
        return sampler_name, steps, DDIMSampler(
            num_sampling_steps=steps,
            eta=eta,
            clip_x0=clip_x0,
        )

    sampler_name = requested_sampler or "euler"
    if sampler_name not in {"euler", "heun"}:
        raise ValueError("Flow Matching requires the euler or heun sampler.")
    steps = 50 if requested_steps is None else requested_steps
    return sampler_name, steps, ODESampler(
        num_steps=steps,
        solver=sampler_name,
    )


def build_label_sequence(
    *,
    config: dict[str, Any],
    count: int,
    class_label: int | None,
    policy: str,
    seed: int,
) -> torch.Tensor | None:
    if not bool(config.get("class_cond", False)):
        if class_label is not None:
            raise ValueError("--class_label requires a class-conditional model.")
        return None

    num_classes = int(config["num_classes"])
    if class_label is not None:
        if not 0 <= class_label < num_classes:
            raise ValueError(
                f"class_label must be in [0, {num_classes - 1}]."
            )
        return torch.full((count,), class_label, dtype=torch.long)
    if policy == "balanced":
        return torch.arange(count, dtype=torch.long) % num_classes
    if policy == "random":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randint(
            0,
            num_classes,
            (count,),
            generator=generator,
        )
    raise ValueError(f"Unknown label policy: {policy}")


def prepare_export_directory(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [*path.glob("*.png"), *path.glob("*.csv")]
    if existing and not overwrite:
        raise FileExistsError(
            f"Export directory is not empty: {path}. Pass --overwrite to "
            "replace its PNG/CSV files."
        )
    if overwrite:
        for item in existing:
            item.unlink()


def _write_labels(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "label"))
        writer.writerows(rows)


def _seed_sampling(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def export_generated_samples(
    *,
    sampler: object,
    model: CountingModel,
    objective: nn.Module,
    config: dict[str, Any],
    output_dir: Path,
    labels: torch.Tensor | None,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    guidance_scale: float,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    _, channels, height, width = infer_sample_shape(config, num_samples=1)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    _seed_sampling(seed, device)
    label_rows: list[tuple[str, int]] = []
    sampling_seconds = 0.0
    before_calls = model.forward_calls
    wall_start = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for start in range(0, num_samples, batch_size):
        current = min(batch_size, num_samples - start)
        shape = (current, channels, height, width)
        initial_noise = torch.randn(
            shape,
            generator=noise_generator,
            device=device,
        )
        cond = (
            None
            if labels is None
            else labels[start : start + current].to(device)
        )
        synchronize(device)
        sample_start = time.perf_counter()
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
        synchronize(device)
        sampling_seconds += time.perf_counter() - sample_start
        samples = denormalize_to_01(samples).cpu()

        for offset, image in enumerate(samples):
            index = start + offset
            filename = f"{index:06d}.png"
            to_pil_image(image).save(output_dir / filename)
            if labels is not None:
                label_rows.append((filename, int(labels[index].item())))

    wall_seconds = time.perf_counter() - wall_start
    if labels is not None:
        _write_labels(output_dir / "labels.csv", label_rows)
    memory = {"available": device.type == "cuda", "unit": "bytes"}
    if device.type == "cuda":
        memory.update(
            {
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return {
        "directory": str(output_dir),
        "format": "lossless PNG",
        "value_range": "uint8 [0, 255] converted from model range [-1, 1]",
        "count": num_samples,
        "sampling_seconds": sampling_seconds,
        "wall_seconds_including_png_writes": wall_seconds,
        "samples_per_second_sampling_only": num_samples / sampling_seconds,
        "model_forward_calls": model.forward_calls - before_calls,
        "cuda_memory": memory,
        "labels_csv": (
            str(output_dir / "labels.csv") if labels is not None else None
        ),
    }


def _dataset_targets(dataset: object) -> torch.Tensor:
    targets = getattr(dataset, "targets")
    if isinstance(targets, torch.Tensor):
        return targets.long()
    return torch.as_tensor(targets, dtype=torch.long)


def export_reference_samples(
    *,
    dataset_name: str,
    data_dir: Path,
    split: str,
    output_dir: Path,
    requested_count: int,
    seed: int,
    class_label: int | None,
    download: bool,
) -> dict[str, Any]:
    train = split == "train"
    if dataset_name == "mnist":
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=download,
        )
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=download,
        )
    else:
        raise ValueError(
            "Reference export currently supports MNIST and CIFAR10."
        )

    targets = _dataset_targets(dataset)
    eligible = torch.arange(len(dataset))
    if class_label is not None:
        eligible = eligible[targets == class_label]
    actual_count = min(requested_count, int(eligible.numel()))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(eligible.numel(), generator=generator)[:actual_count]
    selected = eligible[order]

    label_rows: list[tuple[str, int]] = []
    for output_index, dataset_index in enumerate(selected.tolist()):
        image, label = dataset[dataset_index]
        filename = f"{output_index:06d}.png"
        image.save(output_dir / filename)
        label_rows.append((filename, int(label)))
    _write_labels(output_dir / "labels.csv", label_rows)
    return {
        "directory": str(output_dir),
        "dataset": dataset_name,
        "split": split,
        "requested_count": requested_count,
        "count": actual_count,
        "class_filter": class_label,
        "selection": "seeded random subset without replacement",
        "labels_csv": str(output_dir / "labels.csv"),
    }


def _default_output_dir(
    ckpt_path: Path,
    sampler_name: str,
    num_steps: int,
) -> Path:
    return ckpt_path.parent.parent / "evaluation" / (
        f"{sampler_name}_{num_steps}_steps"
    )


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        result.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_total_memory_bytes": int(
                    torch.cuda.get_device_properties(device).total_memory
                ),
            }
        )
    return result


def _warnings(
    *,
    dataset_name: str,
    generated_count: int,
    reference_count: int,
    class_label: int | None,
) -> list[str]:
    warnings = []
    if min(generated_count, reference_count) < 10_000:
        warnings.append(
            "Fewer than 10,000 samples were used. Standard distribution "
            "metrics are implemented correctly but this estimate has high "
            "finite-sample uncertainty; publication comparisons commonly "
            "use 10,000-50,000 samples under one fixed protocol."
        )
    if dataset_name == "mnist":
        warnings.append(
            "Standard FID/KID use ImageNet InceptionV3 features and PR uses "
            "VGG16 features. MNIST is 28x28 grayscale, so these are standard "
            "implementations but the feature domain is not semantically ideal."
        )
    if class_label is not None:
        warnings.append(
            "This is a class-filtered evaluation; compare it only with runs "
            "using the same generated and reference class protocol."
        )
    return warnings


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    ckpt_path = args.ckpt.resolve()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config, sampling_model, objective, checkpoint = load_sampling_components(
        ckpt_path,
        device,
        args.use_ema,
    )
    objective_name = str(config["objective"]).lower()
    sampler_name, num_steps, sampler = resolve_sampler(
        objective_name=objective_name,
        requested_sampler=args.sampler,
        requested_steps=args.num_steps,
        objective=objective,
        eta=args.eta,
        clip_x0=args.clip_x0,
    )
    guidance_scale = validate_guidance_scale(
        args.guidance_scale
        if args.guidance_scale is not None
        else config.get("guidance_scale", 1.0)
    )
    labels = build_label_sequence(
        config=config,
        count=args.num_samples,
        class_label=args.class_label,
        policy=args.label_policy,
        seed=args.seed,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(ckpt_path, sampler_name, num_steps).resolve()
    )
    generated_dir = output_dir / "generated"
    reference_dir = output_dir / "reference"
    prepare_export_directory(generated_dir, args.overwrite)
    prepare_export_directory(reference_dir, args.overwrite)

    counted_model = CountingModel(sampling_model).to(device).eval()
    benchmark_batch_size = min(
        args.benchmark_batch_size or args.batch_size,
        args.num_samples,
    )
    benchmark_shape = infer_sample_shape(config, benchmark_batch_size)
    benchmark_generator = torch.Generator(device=device).manual_seed(args.seed)
    benchmark_noise = torch.randn(
        benchmark_shape,
        generator=benchmark_generator,
        device=device,
    )
    benchmark_cond = (
        None
        if labels is None
        else labels[:benchmark_batch_size].to(device)
    )
    nfe = expected_nfe(
        sampler_name=sampler_name,
        num_steps=num_steps,
        has_condition=benchmark_cond is not None,
        guidance_scale=guidance_scale,
    )

    def run_benchmark_once() -> torch.Tensor:
        _seed_sampling(args.seed + 1, device)
        return sampler.sample(
            model=counted_model,
            objective=objective,
            shape=benchmark_shape,
            cond=benchmark_cond,
            guidance_scale=guidance_scale,
            device=device,
            initial_noise=benchmark_noise,
            num_steps=num_steps,
        )

    performance = benchmark_sampler(
        run_once=run_benchmark_once,
        counting_model=counted_model,
        device=device,
        batch_size=benchmark_batch_size,
        warmup_runs=args.warmup_runs,
        repetitions=args.repetitions,
        expected_forward_calls=nfe,
    )

    generated_export = export_generated_samples(
        sampler=sampler,
        model=counted_model,
        objective=objective,
        config=config,
        output_dir=generated_dir,
        labels=labels,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        device=device,
        seed=args.seed,
    )
    requested_reference_count = args.num_reference or args.num_samples
    reference_class = (
        args.class_label
        if args.class_label is not None and args.match_reference_condition
        else None
    )
    data_dir = Path(str(config.get("data_dir", "data")))
    if not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    reference_export = export_reference_samples(
        dataset_name=str(config["dataset"]).lower(),
        data_dir=data_dir,
        split=args.reference_split,
        output_dir=reference_dir,
        requested_count=requested_reference_count,
        seed=args.seed,
        class_label=reference_class,
        download=args.download_reference,
    )

    metric_results: dict[str, Any] = {}
    if "pixel_fd" in args.metrics:
        start = time.perf_counter()
        metric_results["pixel_fd_proxy"] = calculate_pixel_fd_proxy(
            generated_dir,
            reference_dir,
        )
        metric_results["pixel_fd_proxy"]["elapsed_seconds"] = (
            time.perf_counter() - start
        )

    requested_standard = set(args.metrics) & STANDARD_METRIC_NAMES
    if requested_standard:
        synchronize(device)
        start = time.perf_counter()
        standard = calculate_standard_metrics(
            generated_dir=generated_dir,
            reference_dir=reference_dir,
            device=device,
            fid="fid" in requested_standard,
            kid="kid" in requested_standard,
            prc="prc" in requested_standard,
            batch_size=args.metric_batch_size,
            kid_subsets=args.kid_subsets,
            kid_subset_size=args.kid_subset_size,
            prc_neighborhood=args.prc_neighborhood,
            prc_batch_size=args.prc_batch_size,
            seed=args.seed,
            verbose=args.metric_verbose,
        )
        synchronize(device)
        standard["elapsed_seconds"] = time.perf_counter() - start
        metric_results["standard"] = standard

    generated_count = int(generated_export["count"])
    reference_count = int(reference_export["count"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "checkpoint": str(ckpt_path),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "sampling_model": "ema" if args.use_ema else "online",
            "seed": args.seed,
            "output_dir": str(output_dir),
        },
        "sampling": {
            "objective": objective_name,
            "sampler": sampler_name,
            "num_steps": num_steps,
            "eta": args.eta if sampler_name == "ddim" else None,
            "clip_x0": args.clip_x0 if objective_name == "ddpm" else None,
            "guidance_scale": guidance_scale,
            "label_policy": (
                "unconditional" if labels is None else args.label_policy
            ),
            "fixed_class_label": args.class_label,
            "nfe_definition": (
                "Number of neural-network forward evaluations along one "
                "sample trajectory; CFG dual branches and Heun stages count "
                "separately."
            ),
        },
        "exports": {
            "generated": generated_export,
            "reference": reference_export,
        },
        "performance": performance,
        "metrics": metric_results,
        "metric_families": {
            "standard": (
                "torch-fidelity reference-compatible FID/KID and generative "
                "manifold Precision-Recall"
            ),
            "proxy": (
                "project-local 7x7 grayscale pixel-FD; never report as FID"
            ),
        },
        "warnings": _warnings(
            dataset_name=str(config["dataset"]).lower(),
            generated_count=generated_count,
            reference_count=reference_count,
            class_label=args.class_label,
        ),
        "environment": _environment(device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.json"
    result["run"]["results_json"] = str(result_path)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
