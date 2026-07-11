from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Make src/ importable when runing:
#   python scripts/sample.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ugm.models.unet import SimpleUNet
from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.samplers.ode import EulerODESampler


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained DDPM or Flow Matching model."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. outputs/fm_mnist/checkpoints/last.pt"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. If None, it is saved next to the checkpoint.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--nrow",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda, cpu, or None for auto.",
    )

    parser.add_argument(
        "--objective",
        type=str,
        default=None,
        choices=["ddpm", "fm"],
        help="Override objective. If None, read from args.json.",
    )

    parser.add_argument(
        "--ode_steps",
        type=int,
        default=50,
        help="Number of Euler ODE steps for Flow Matching sampling.",
    )

    parser.add_argument(
        "--clip_x0",
        action="store_true",
        help="Clip predicted x0 to [-1, 1] during DDPM sampling.",
    )

    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help=(
            "Optional fixed class label for class-conditional sampling. "
            "If not provided and the model is class-conditional, random labels are used."
        ),
    )

    return parser.parse_args()


def load_config_from_checkpoint_dir(ckpt_path: Path) -> dict:
    """
    Locate args.json or config.json from the directory.

    Expected checkpoint path:

        output/fm_mnist/checkpoint/last.pt

    Then the config should usually be:
        
        outputs/fm_mnist/args.json
        or
        outputs/fm_mnist/config.json
    """

    output_dir = ckpt_path.parent.parent

    args_json = output_dir / "args.json"
    config_json = output_dir / "config.json"

    if args_json.exists():
        config_path = args_json
    elif config_json.exists():
        config_path = config_json
    else:
        raise FileNotFoundError(
            f"Cannot find args.json or config.json under {output_dir}"
        )
    
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def build_model_from_config(config:dict) -> SimpleUNet:
    in_channels = int(config["in_channels"])
    model = SimpleUNet(
        in_channels=in_channels,
        out_channels=int(config.get("out_channels", in_channels)),
        base_channels=int(config.get("base_channels", 64)),
        time_dim=int(config.get("time_dim", 256)),
        num_classes=(
            int(config["num_classes"])
            if bool(config.get("class_cond", False))
            else None
        )
    )

    return model

def build_objective_from_config(
    objective_name: str,
    config: dict,
):
    if objective_name == "ddpm":
        objective = DDPMobjective(
            num_timesteps=int(config.get("num_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 1e-4)),
            beta_end=float(config.get("beta_end", 2e-2)),
            prediction_type="epsilon",
        )
        return objective

    if objective_name == "fm":
        objective = FlowMatchingObjective(
            t_min=0.0,
            t_max=1.0,
        )
        return objective

    raise ValueError(f"Unknown objective: {objective_name}")


def infer_sample_shape(
    config: dict,
    num_samples: int,
) -> tuple[int, int, int, int]:
    in_channels = int(config["in_channels"])

    if "image_size" not in config:
        dataset = str(config.get("dataset", "")).lower()
        if dataset == "mnist":
            image_size = 28
        elif dataset == "cifar10":
            image_size = 32
        else:
            raise ValueError(
                "Cannot infer image_size. Please make sure image_size is saved "
                "in args.json."
            )
    else:
        image_size = int(config["image_size"])

    return num_samples, in_channels, image_size, image_size


def build_condition(
    config: dict,
    num_samples: int,
    device: torch.device,
    class_label: int | None,
) -> torch.Tensor | None:
    class_cond = bool(config.get("class_cond", False))

    if not class_cond:
        return None

    num_classes = int(config["num_classes"])

    if class_label is None:
        cond = torch.randint(
            low=0,
            high=num_classes,
            size=(num_samples,),
            device=device,
        )
    else:
        if not (0 <= class_label < num_classes):
            raise ValueError(
                f"class_label must be in [0, {num_classes - 1}], "
                f"got {class_label}."
            )

        cond = torch.full(
            size=(num_samples,),
            fill_value=class_label,
            device=device,
            dtype=torch.long,
        )

    return cond


def denormalize_to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert samples from [-1, 1] to [0, 1] for image saving.
    """
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def main() -> None:
    args = parser_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config = load_config_from_checkpoint_dir(ckpt_path)

    objective_name = args.objective
    if objective_name is None:
        objective_name = str(config.get("objective", "")).lower()

    if objective_name not in {"ddpm", "fm"}:
        raise ValueError(
            f"Cannot infer objective from config. Got: {objective_name}. "
            "Please pass --objective ddpm or --objective fm."
        )

    model = build_model_from_config(config)
    objective = build_objective_from_config(objective_name, config)

    checkpoint = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(checkpoint["model"], strict=True)
    if "objective" in checkpoint:
        objective.load_state_dict(checkpoint["objective"], strict=True)

    model = model.to(device)
    objective = objective.to(device)

    shape = infer_sample_shape(
        config=config,
        num_samples=args.num_samples,
    )

    cond = build_condition(
        config=config,
        num_samples=args.num_samples,
        device=device,
        class_label=args.class_label,
    )

    if objective_name == "ddpm":
        sampler = DDPMAncestralSampler(
            clip_x0=args.clip_x0,
        )
        samples = sampler.sample(
            model=model,
            objective=objective,
            shape=shape,
            cond=cond,
            device=device,
        )

    elif objective_name == "fm":
        sampler = EulerODESampler(
            num_steps=args.ode_steps,
            t_start=0.0,
            t_end=1.0,
        )
        samples = sampler.sample(
            model=model,
            shape=shape,
            cond=cond,
            device=device,
        )

    else:
        raise RuntimeError("Invalid objective.")

    samples = denormalize_to_01(samples)

    if args.output is None:
        output_path = ckpt_path.parent.parent / f"samples_{objective_name}.png"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        samples,
        output_path,
        nrow=args.nrow,
    )

    print(f"Saved samples to: {output_path}")


if __name__ == "__main__":
    main()
  