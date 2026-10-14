from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ugm.models.adapters.diffusers_adapter import (  # noqa: E402
    DiffusersUNet2DAdapter,
    create_diffusers_unet2d,
)
from ugm.pipelines.diffusers_pipeline import (  # noqa: E402
    DiffusersComponentPipeline,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load or create a Diffusers UNet2DModel, call it through the "
            "UGMF adapter, save it, reload it, and verify matching outputs."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/diffusers/unet2d_roundtrip"),
    )
    parser.add_argument("--source_model", type=str, default=None)
    parser.add_argument("--subfolder", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument(
        "--safe_serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _resolve(path: Path) -> Path:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_roundtrip(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0 or args.image_size <= 0:
        raise ValueError("batch_size and image_size must be positive.")
    if args.num_train_timesteps <= 1:
        raise ValueError("num_train_timesteps must be greater than one.")
    if args.atol < 0:
        raise ValueError("atol must be non-negative.")

    _seed_all(args.seed)
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"

    if args.source_model is None:
        source_dir = output_dir / "source_component"
        component = create_diffusers_unet2d(
            {
                "in_channels": args.in_channels,
                "out_channels": args.out_channels,
                "image_size": args.image_size,
                "num_timesteps": args.num_train_timesteps,
                "class_cond": False,
            }
        ).component
        component.save_pretrained(
            source_dir,
            safe_serialization=args.safe_serialization,
        )
        source = str(source_dir)
        adapter = DiffusersUNet2DAdapter.from_diffusers_pretrained(
            source_dir,
            timestep_scale=args.num_train_timesteps - 1,
            local_files_only=True,
        )
    else:
        source = args.source_model
        adapter = DiffusersUNet2DAdapter.from_diffusers_pretrained(
            args.source_model,
            subfolder=args.subfolder,
            timestep_scale=args.num_train_timesteps - 1,
        )

    pipeline = DiffusersComponentPipeline(adapter.to(device).eval())
    x = torch.randn(
        args.batch_size,
        args.in_channels,
        args.image_size,
        args.image_size,
        device=device,
    )
    t = torch.linspace(0.0, 1.0, args.batch_size, device=device)
    before = pipeline(x, t).cpu()
    pipeline.save_pretrained(
        adapter_dir,
        safe_serialization=args.safe_serialization,
    )

    reloaded = DiffusersComponentPipeline.from_pretrained(adapter_dir)
    reloaded.model.to(device).eval()
    after = reloaded(x, t).cpu()
    max_abs_error = float((before - after).abs().max())
    outputs_close = bool(torch.allclose(before, after, rtol=0.0, atol=args.atol))
    result: dict[str, object] = {
        "schema_version": 1,
        "component": "diffusers.UNet2DModel",
        "diffusers_source": source,
        "adapter_directory": str(adapter_dir),
        "device": str(device),
        "seed": args.seed,
        "input_shape": list(x.shape),
        "output_shape": list(before.shape),
        "timestep_scale": args.num_train_timesteps - 1,
        "safe_serialization": args.safe_serialization,
        "finite_before": bool(torch.isfinite(before).all()),
        "finite_after": bool(torch.isfinite(after).all()),
        "max_abs_error": max_abs_error,
        "atol": args.atol,
        "outputs_close": outputs_close,
    }
    with (output_dir / "roundtrip.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    if not outputs_close:
        raise RuntimeError(
            f"Reloaded output differs by {max_abs_error}, exceeding {args.atol}."
        )
    return result


def main() -> None:
    result = run_roundtrip(build_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
