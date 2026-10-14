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

from ugm.models.conditioning import validate_guidance_scale
from ugm.models.build import build_model
from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.objectives.rectified_flow import RectifiedFlowObjective
from ugm.samplers.ddim import DDIMSampler
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.samplers.consistency import ConsistencySampler
from ugm.samplers.base import SamplingResult
from ugm.samplers.ode import ODESampler
from ugm.utils.ema import EMA


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample from a trained DDPM, Flow Matching, Rectified Flow, or "
            "consistency-distillation model."
        )
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
        choices=["ddpm", "fm", "rectified_flow", "consistency_distillation"],
        help="Override objective. If None, read from the saved config.",
    )

    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=["ddpm", "ddim", "euler", "heun", "consistency"],
        help=(
            "Sampling algorithm. Defaults to ddpm for a DDPM objective "
            "and euler for a Flow Matching objective."
        ),
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help=(
            "Common runtime sampling-step override. The legacy "
            "--num_sampling_steps and --ode_steps flags remain supported."
        ),
    )

    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps.",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help=(
            "DDIM stochasticity parameter. "
            "eta=0 gives deterministic DDIM sampling."
        ),
    )

    parser.add_argument(
        "--ode_steps",
        type=int,
        default=50,
        help="Legacy default number of ODE steps for Flow Matching sampling.",
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

    parser.add_argument(
        "--use_ema",
        action="store_true",
        help=(
            "Use the MEA model stored in the checkpoint "
            "for sampling."
        ),
    )

    parser.add_argument(
        "--consistency_model",
        choices=["target", "student"],
        default="target",
        help=(
            "For consistency checkpoints, sample the checkpointed EMA target "
            "(default) or the online student."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initial and intermediate consistency noise.",
    )

    parser.add_argument(
        "--initial_noise",
        type=Path,
        default=None,
        help=(
            "Optional .pt file containing a tensor with the requested sample "
            "shape. Reuse it for fair solver comparisons."
        ),
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help=(
            "Classifier-free guidance scale. Defaults to the saved config; "
            "0 is unconditional and 1 is the ordinary conditional model."
        ),
    )

    parser.add_argument(
        "--trajectory_output",
        type=Path,
        default=None,
        help=(
            "Optional .pt path for the complete trajectory and timesteps. "
            "The initial state and final sample are included."
        ),
    )

    return parser.parse_args()


def load_config_from_checkpoint_dir(ckpt_path: Path) -> dict:
    """
    Locate the canonical config.json or a legacy args.json.

    Expected checkpoint path:

        output/fm_mnist/checkpoint/last.pt

    Then the config should usually be:
        
        outputs/fm_mnist/config.json
        or, for older runs, outputs/fm_mnist/args.json
    """

    output_dir = ckpt_path.parent.parent

    args_json = output_dir / "args.json"
    config_json = output_dir / "config.json"

    if config_json.exists():
        config_path = config_json
    elif args_json.exists():
        config_path = args_json
    else:
        raise FileNotFoundError(
            f"Cannot find args.json or config.json under {output_dir}"
        )
    
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def build_model_from_config(config: dict) -> torch.nn.Module:
    return build_model(config)

def build_objective_from_config(
    objective_name: str,
    config: dict,
):
    if objective_name == "ddpm":
        objective = DDPMobjective(
            num_timesteps=int(config.get("num_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 1e-4)),
            beta_end=float(config.get("beta_end", 2e-2)),
            prediction_type=str(config.get("prediction_type", "epsilon")),
        )
        return objective

    if objective_name == "fm":
        objective = FlowMatchingObjective(
            t_min=float(config.get("t_min", 0.0)),
            t_max=float(config.get("t_max", 1.0)),
        )
        return objective

    if objective_name == "rectified_flow":
        return RectifiedFlowObjective(
            t_min=float(config.get("t_min", 0.0)),
            t_max=float(config.get("t_max", 1.0)),
        )

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
                "in config.json."
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


def load_initial_noise(
    path: Path | None,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Initial noise file not found: {path}")

    initial_noise = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(initial_noise, torch.Tensor):
        raise TypeError("--initial_noise must point to a serialized tensor.")
    if tuple(initial_noise.shape) != tuple(shape):
        raise ValueError(
            "Loaded initial noise has shape "
            f"{tuple(initial_noise.shape)}, expected {tuple(shape)}."
        )
    return initial_noise


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
    guidance_scale = validate_guidance_scale(
        args.guidance_scale
        if args.guidance_scale is not None
        else config.get("guidance_scale", 1.0)
    )

    objective_name = args.objective
    if objective_name is None:
        objective_name = str(config.get("objective", "")).lower()

    if objective_name not in {
        "ddpm",
        "fm",
        "rectified_flow",
        "consistency_distillation",
    }:
        raise ValueError(
            f"Cannot infer objective from config. Got: {objective_name}. "
            "Please pass --objective ddpm, fm, rectified_flow, or "
            "consistency_distillation."
        )
    if args.num_sampling_steps <= 0:
        raise ValueError(
            "--num_sampling_steps must be positive."
        )

    if args.num_steps is not None and args.num_steps <= 0:
        raise ValueError("--num_steps must be positive.")

    if args.ode_steps <= 0:
        raise ValueError("--ode_steps must be positive.")

    if args.eta < 0.0:
        raise ValueError(
            "--eta must be non-negative."
        )

    if objective_name in {"fm", "rectified_flow"} and args.sampler in {"ddpm", "ddim"}:
        raise ValueError(
            "--sampler ddpm/ddim can only be used "
            "with the DDPM objective."
        )
    if objective_name == "ddpm" and args.sampler in {"euler", "heun"}:
        raise ValueError(
            "--sampler euler/heun can only be used with the FM objective."
        )

    if objective_name == "consistency_distillation":
        if args.sampler not in {None, "consistency"}:
            raise ValueError(
                "Consistency checkpoints require --sampler consistency."
            )
        effective_steps = 1 if args.num_steps is None else args.num_steps
        if effective_steps not in ConsistencySampler.SUPPORTED_STEPS:
            raise ValueError("Consistency --num_steps must be 1, 2, or 4.")
        if args.use_ema:
            raise ValueError(
                "Consistency checkpoints store their EMA target inside the "
                "objective; use --consistency_model target, not --use_ema."
            )

    model = build_model_from_config(config)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False,)

    model_state = checkpoint.get("model")

    if model_state is None:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain "
            "a valid 'model' state."
        )

    objective = None
    if objective_name == "consistency_distillation":
        if args.consistency_model == "student":
            sampling_state = checkpoint["model"]
        else:
            objective_state = checkpoint.get("objective")
            if not isinstance(objective_state, dict):
                raise KeyError(
                    "Consistency checkpoint has no objective/EMA target state."
                )
            prefix = "target."
            sampling_state = {
                key[len(prefix):]: value
                for key, value in objective_state.items()
                if key.startswith(prefix)
            }
            if not sampling_state:
                raise KeyError(
                    "Consistency checkpoint objective has no target.* weights."
                )
        model.load_state_dict(sampling_state, strict=True)
    else:
        objective = build_objective_from_config(objective_name, config)
        model.load_state_dict(checkpoint["model"], strict=True)
        objective_state = checkpoint.get("objective")
        if objective_state is not None:
            objective.load_state_dict(objective_state, strict=True)

    model = model.to(device)
    if objective is not None:
        objective = objective.to(device)
        objective.eval()

    sampling_model = model

    if args.use_ema:
        ema_state = checkpoint.get("ema")

        if ema_state is None:
            raise KeyError(
                "--use_ema was specified, but the checkpoint "
                "does not contain a valid 'ema' state."
            )
        
        ema = EMA(
            model=model,
        )

        ema.load_state_dict(ema_state)

        sampling_model = ema.ema_model

    sampling_model = sampling_model.to(device=device)
    sampling_model.eval()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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

    initial_noise = load_initial_noise(
        path=args.initial_noise,
        shape=shape,
        device=device,
    )

    with torch.inference_mode():
        if objective_name == "ddpm":
            sampler_name = args.sampler or "ddpm"

            if sampler_name == "ddpm":
                sampler = DDPMAncestralSampler(
                    clip_x0=args.clip_x0,
                )
            elif sampler_name == "ddim":
                sampler = DDIMSampler(
                    num_sampling_steps=args.num_sampling_steps,
                    eta=args.eta,
                    clip_x0=args.clip_x0,
                )
            else:
                raise ValueError(
                    f"Unsupported DDPM sampler: {sampler_name}"
                )

            samples = sampler.sample(
                model=sampling_model,
                objective=objective,
                shape=shape,
                cond=cond,
                guidance_scale=guidance_scale,
                device=device,
                initial_noise=initial_noise,
                num_steps=args.num_steps,
                return_trajectory=args.trajectory_output is not None,
            )
        elif objective_name in {"fm", "rectified_flow"}:
            sampler_name = args.sampler or "euler"
            sampler = ODESampler(
                num_steps=args.ode_steps,
                t_start=0.0,
                t_end=1.0,
                solver=sampler_name,
            )

            samples = sampler.sample(
                model=sampling_model,
                objective=objective,
                shape=shape,
                cond=cond,
                guidance_scale=guidance_scale,
                device=device,
                initial_noise=initial_noise,
                num_steps=args.num_steps,
                return_trajectory=args.trajectory_output is not None,
            )
        elif objective_name == "consistency_distillation":
            sampler_name = "consistency"
            effective_steps = 1 if args.num_steps is None else args.num_steps
            sampler = ConsistencySampler(num_steps=effective_steps)
            samples = sampler.sample(
                model=sampling_model,
                objective=None,
                shape=shape,
                cond=cond,
                guidance_scale=guidance_scale,
                device=device,
                initial_noise=initial_noise,
                num_steps=effective_steps,
                return_trajectory=args.trajectory_output is not None,
            )
        else:
            raise RuntimeError(
                f"Invalid objective: {objective_name}"
            )

    sampling_result = samples
    if isinstance(sampling_result, SamplingResult):
        samples = sampling_result.samples

        assert args.trajectory_output is not None
        args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trajectory": sampling_result.trajectory.cpu(),
                "timesteps": sampling_result.timesteps.cpu(),
            },
            args.trajectory_output,
        )

    samples = denormalize_to_01(samples)

    if args.output is None:
        if objective_name == "consistency_distillation":
            model_type = args.consistency_model
        else:
            model_type = "ema" if args.use_ema else "online"
        output_path = ckpt_path.parent.parent / f"samples_{objective_name}_{model_type}.png"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_image(
        samples,
        output_path,
        nrow=args.nrow,
    )

    model_label = (
        args.consistency_model
        if objective_name == "consistency_distillation"
        else ("EMA" if args.use_ema else "online")
    )
    print(f"Sampling model: {model_label}")

    if objective_name == "ddpm":
        print(f"Sampler: {args.sampler or 'ddpm'}")

        if (args.sampler or "ddpm") == "ddim":
            effective_steps = (
                args.num_steps
                if args.num_steps is not None
                else args.num_sampling_steps
            )
            print(
                f"DDIM steps: {effective_steps}, "
                f"eta: {args.eta}"
            )
    elif objective_name in {"fm", "rectified_flow"}:
        print(f"Sampler: {args.sampler or 'euler'}")
        effective_steps = (
            args.num_steps
            if args.num_steps is not None
            else args.ode_steps
        )
        print(f"ODE steps: {effective_steps}")
    else:
        print("Sampler: consistency")
        cfg_branches = (
            2
            if cond is not None and guidance_scale not in {0.0, 1.0}
            else 1
        )
        print(f"Consistency evaluations: {effective_steps}")
        print(f"NFE: {effective_steps * cfg_branches}")

    print(f"Seed: {args.seed}")

    print(f"Guidance scale: {guidance_scale}")

    if args.trajectory_output is not None:
        print(f"Saved trajectory to: {args.trajectory_output}")

    print(f"Saved samples to: {output_path}")


if __name__ == "__main__":
    main()
