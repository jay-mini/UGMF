from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make src/importable when running:
#       python scripts/train.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from ugm.models.unet import SimpleUNet
from ugm.objectives.ddpm import DDPMobjective
from ugm.objectives.flow_matching import FlowMatchingObjective
from ugm.trainers.base_trainer import BaseTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DDPM or Flow Matching using a unified trainer."
    )

    parser.add_argument(
        "--objective",
        type=str,
        default="fm",
        choices=["ddpm", "fm"],
        help="Training objective.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset name."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory for datasets."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. If None, constructed automatically.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum number of training steps.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--base_channels",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--time_dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--class_cond",
        action="store_true",
        help="Use class labels as conditioning.",
    )

    # DDPM-specific arguments
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--beta_start",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--beta_end",
        type=float,
        default=2e-2,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_dataset(
        dataset_name: str,
        data_dir: str,
):
    """
    Build normalized dataset.

    Images are normalized to [-1, 1], which is standard for diffusion/FM imaging training.
    """

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        in_channels = 1
        image_size = 28
        num_classes = 10
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        in_channels = 3
        image_size = 32
        num_classes = 10

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, in_channels, image_size, num_classes


def build_objective(args: argparse.Namespace):
    if args.objective == "ddpm":
        return DDPMobjective(
            num_timesteps=args.num_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            prediction_type="epsilon",
        )
    
    if args.objective == 'fm':
        return FlowMatchingObjective(
            t_min=0.0,
            t_max=1.0,
        )
    
    raise ValueError(f"Unknown objective: {args.objective}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, in_channels, image_size, num_classes = build_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    model = SimpleUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_classes=num_classes if args.class_cond else None,
    )

    objective = build_objective(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "outputs"
            / f"{args.objective}_{args.dataset}"
        )
    else:
        output_dir = Path(args.output_dir)

    config = vars(args)
    config.update(
        {
            "device": str(device),
            "in_channels": in_channels,
            "image_size": image_size,
            "num_classes": num_classes,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    trainer = BaseTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
        mixed_precision=args.mixed_precision,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        save_every=args.save_every,
        config=config,
    )

    trainer.train(
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()




