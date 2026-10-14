from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ugm.trainers.base_trainer import BaseTrainer


@dataclass(frozen=True)
class InitializationState:
    checkpoint_path: Path
    source: str
    checkpoint_global_step: int


def initialize_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    use_ema: bool = False,
    strict: bool = True,
) -> InitializationState:
    """Load only model weights for a new fine-tuning run.

    Optimizer, scheduler, EMA counters, RNG, epoch and global step are
    intentionally not restored.  Full state restoration belongs to
    ``BaseTrainer.load_checkpoint`` and is exposed separately as ``resume``.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid training checkpoint: {checkpoint_path}")
    if use_ema:
        ema_state = checkpoint.get("ema")
        if not isinstance(ema_state, dict) or "ema_model" not in ema_state:
            raise KeyError(
                "EMA initialization was requested, but the checkpoint has no EMA."
            )
        model_state = ema_state["ema_model"]
        source = "ema"
    else:
        model_state = checkpoint["model"]
        source = "online"
    model.load_state_dict(model_state, strict=strict)
    return InitializationState(
        checkpoint_path=checkpoint_path.resolve(),
        source=source,
        checkpoint_global_step=int(checkpoint.get("global_step", 0)),
    )


class RectifiedFlowTrainer(BaseTrainer):
    """Semantic trainer name for RF fine-tuning.

    All optimization/checkpoint behavior is inherited from ``BaseTrainer``;
    the RF-specific behavior lives in the paired dataset and objective.
    """

    pass
