from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.compare_sampler import load_sampling_components  # noqa: E402
from scripts.sample import denormalize_to_01, infer_sample_shape  # noqa: E402
from ugm.models.conditioning import validate_guidance_scale  # noqa: E402
from ugm.samplers.ddim import DDIMSampler  # noqa: E402
from ugm.samplers.ddpm import DDPMAncestralSampler  # noqa: E402


class ConditionAuditModel(nn.Module):
    """Forward to a model while recording the conditions it actually sees."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.unconditional_calls = 0
        self.conditional_calls = 0
        self.conditional_labels: Counter[int] = Counter()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            self.unconditional_calls += 1
        else:
            self.conditional_calls += 1
            self.conditional_labels.update(
                int(label) for label in cond.detach().cpu().tolist()
            )
        return self.model(x, t, cond=cond)

    def audit(self) -> dict[str, object]:
        return {
            "unconditional_calls": self.unconditional_calls,
            "conditional_calls": self.conditional_calls,
            "conditional_labels_seen": {
                str(label): count
                for label, count in sorted(self.conditional_labels.items())
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible class-by-noise grids at multiple CFG scales."
        )
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(
            "outputs/ddpm_mnist_conditional/checkpoints/last.pt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "outputs/ddpm_mnist_conditional/conditional_evaluation"
        ),
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument("--samples_per_class", type=int, default=4)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
    return parser


def validate_args(
    args: argparse.Namespace,
    config: dict,
) -> tuple[list[int], list[float]]:
    if not bool(config.get("class_cond", False)):
        raise ValueError("The checkpoint is not class conditional.")
    if args.samples_per_class <= 0:
        raise ValueError("--samples_per_class must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num_steps must be positive.")
    if args.eta < 0.0:
        raise ValueError("--eta must be non-negative.")
    if not args.classes:
        raise ValueError("At least one class is required.")
    if not args.scales:
        raise ValueError("At least one guidance scale is required.")

    classes = [int(label) for label in args.classes]
    if len(set(classes)) != len(classes):
        raise ValueError("--classes must not contain duplicates.")
    num_classes = int(config["num_classes"])
    invalid_classes = [
        label for label in classes if not 0 <= label < num_classes
    ]
    if invalid_classes:
        raise ValueError(
            f"Class labels must be in [0, {num_classes - 1}], "
            f"got {invalid_classes}."
        )

    scales = [validate_guidance_scale(scale) for scale in args.scales]
    if len(set(scales)) != len(scales):
        raise ValueError("--scales must not contain duplicates.")
    return classes, scales


def build_balanced_inputs(
    classes: list[int],
    samples_per_class: int,
    sample_shape: tuple[int, int, int],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each latent column for every class row.

    The item at row ``class_index`` and column ``sample_index`` therefore has
    exactly the same initial noise as the same column in every other row.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    base_noise = torch.randn(
        (samples_per_class, *sample_shape),
        generator=generator,
    )
    initial_noise = base_noise.repeat(len(classes), 1, 1, 1)
    cond = torch.tensor(classes, dtype=torch.long).repeat_interleave(
        samples_per_class
    )
    return initial_noise, cond


def _scale_slug(scale: float) -> str:
    return f"{scale:g}".replace("-", "minus_").replace(".", "p")


def save_labeled_grid(
    images: torch.Tensor,
    row_labels: list[str],
    column_labels: list[str],
    nrow: int,
    output_path: Path,
) -> None:
    if images.shape[0] != len(row_labels) * nrow:
        raise ValueError("The number of images does not match the grid labels.")
    if len(column_labels) != nrow:
        raise ValueError("The number of column labels must equal nrow.")

    padding = 2
    grid = make_grid(images.cpu(), nrow=nrow, padding=padding, pad_value=1.0)
    grid_image = to_pil_image(grid).convert("RGB")
    # MNIST cells are only 28x28 pixels. Enlarge with nearest-neighbour
    # interpolation so labels remain readable without changing image values.
    display_scale = 3
    grid_image = grid_image.resize(
        (
            grid_image.width * display_scale,
            grid_image.height * display_scale,
        ),
        resample=Image.Resampling.NEAREST,
    )
    left_margin = 68
    top_margin = 28
    canvas = Image.new(
        "RGB",
        (grid_image.width + left_margin, grid_image.height + top_margin),
        "white",
    )
    canvas.paste(grid_image, (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas)

    cell_width = grid_image.width / nrow
    cell_height = grid_image.height / len(row_labels)
    for column_index, label in enumerate(column_labels):
        x = left_margin + (column_index + 0.5) * cell_width
        draw.text((x, 4), label, fill="black", anchor="ma")
    for row_index, label in enumerate(row_labels):
        y = top_margin + (row_index + 0.5) * cell_height
        draw.text((left_margin - 6, y), label, fill="black", anchor="rm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def compute_scale_metrics(
    samples_by_scale: dict[float, torch.Tensor],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    unconditional = samples_by_scale.get(0.0)

    for scale, samples in samples_by_scale.items():
        class_mean = samples.mean(dim=0, keepdim=True)
        scale_metrics = {
            "mean_absolute_label_spread": float(
                (samples - class_mean).abs().mean().item()
            )
        }
        if unconditional is not None:
            scale_metrics["mean_absolute_shift_from_scale_0"] = float(
                (samples - unconditional).abs().mean().item()
            )
        metrics[f"{scale:g}"] = scale_metrics

    if unconditional is not None:
        metrics["0"]["max_label_difference_for_fixed_noise"] = float(
            (unconditional - unconditional[:1]).abs().max().item()
        )
    return metrics


def reset_sampling_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def generate_class_grids(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt_path = args.ckpt.resolve()
    output_dir = args.output_dir.resolve()
    config, model, objective = load_sampling_components(
        ckpt_path=ckpt_path,
        device=device,
        use_ema=args.use_ema,
    )
    classes, scales = validate_args(args, config)

    _, channels, height, width = infer_sample_shape(config, num_samples=1)
    initial_noise, cond = build_balanced_inputs(
        classes=classes,
        samples_per_class=args.samples_per_class,
        sample_shape=(channels, height, width),
        seed=args.seed,
    )
    initial_noise = initial_noise.to(device)
    cond = cond.to(device)
    shape = tuple(initial_noise.shape)

    if args.sampler == "ddim":
        sampler = DDIMSampler(
            num_sampling_steps=args.num_steps,
            eta=args.eta,
            clip_x0=args.clip_x0,
        )
    else:
        sampler = DDPMAncestralSampler(clip_x0=args.clip_x0)

    samples_by_scale: dict[float, torch.Tensor] = {}
    audits: dict[str, dict[str, object]] = {}
    grid_files: dict[str, str] = {}

    for scale in scales:
        reset_sampling_seed(args.seed, device)
        audited_model = ConditionAuditModel(model).to(device).eval()
        sampled = sampler.sample(
            model=audited_model,
            objective=objective,
            shape=shape,
            cond=cond,
            guidance_scale=scale,
            device=device,
            initial_noise=initial_noise,
            num_steps=args.num_steps,
        )
        sampled = denormalize_to_01(sampled).cpu().reshape(
            len(classes),
            args.samples_per_class,
            channels,
            height,
            width,
        )
        samples_by_scale[scale] = sampled
        audits[f"{scale:g}"] = audited_model.audit()

        grid_path = output_dir / f"class_grid_scale_{_scale_slug(scale)}.png"
        save_labeled_grid(
            images=sampled.flatten(0, 1),
            row_labels=[f"class {label}" for label in classes],
            column_labels=[
                f"noise {index}" for index in range(args.samples_per_class)
            ],
            nrow=args.samples_per_class,
            output_path=grid_path,
        )
        grid_files[f"{scale:g}"] = str(grid_path)

    comparison = torch.stack(
        [
            samples_by_scale[scale][class_index, 0]
            for class_index in range(len(classes))
            for scale in scales
        ]
    )
    comparison_path = output_dir / "guidance_comparison.png"
    save_labeled_grid(
        images=comparison,
        row_labels=[f"class {label}" for label in classes],
        column_labels=[f"scale {scale:g}" for scale in scales],
        nrow=len(scales),
        output_path=comparison_path,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    metrics = compute_scale_metrics(samples_by_scale)
    exact_scale_zero_expected = args.sampler == "ddim" and args.eta == 0.0
    scale_zero_check = None
    if 0.0 in samples_by_scale and exact_scale_zero_expected:
        difference = metrics["0"]["max_label_difference_for_fixed_noise"]
        scale_zero_check = math.isclose(difference, 0.0, abs_tol=1e-7)
        if not scale_zero_check:
            raise AssertionError(
                "CFG scale 0 should be label invariant for deterministic DDIM."
            )

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, object] = {
        "checkpoint": str(ckpt_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "sampling_model": "ema" if args.use_ema else "online",
        "device": str(device),
        "sampler": args.sampler,
        "num_steps": args.num_steps,
        "eta": args.eta,
        "seed": args.seed,
        "classes": classes,
        "guidance_scales": scales,
        "samples_per_class": args.samples_per_class,
        "layout": (
            "Rows are class labels. Noise columns are fixed across rows. "
            "guidance_comparison.png uses noise 0 for every cell."
        ),
        "condition_forward_audit": audits,
        "metrics": metrics,
        "deterministic_scale_zero_label_invariance_passed": scale_zero_check,
        "metric_note": (
            "Label spread measures pixel sensitivity, not semantic class "
            "accuracy; visual inspection or a classifier is still required."
        ),
        "files": {
            "manifest": str(manifest_path),
            "guidance_comparison": str(comparison_path),
            "class_grids": grid_files,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_class_grids(args)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
