from __future__ import annotations

import math

import torch
from torch import nn


UNCONDITIONAL_CLASS_LABEL = -1


def validate_guidance_scale(guidance_scale: float) -> float:
    """Validate the CFG convention used throughout the project."""

    scale = float(guidance_scale)
    if not math.isfinite(scale) or scale < 0.0:
        raise ValueError("guidance_scale must be finite and non-negative.")
    return scale


def drop_class_condition(
    cond: torch.Tensor | None,
    drop_probability: float,
) -> torch.Tensor | None:
    """Independently replace class labels with the null label for CFG.

    The mask is sampled per item rather than once per batch. A probability of
    zero returns the original labels; a probability of one avoids consuming
    RNG and returns an all-null label tensor.
    """

    probability = float(drop_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("drop_probability must be in [0, 1].")
    if cond is None or probability == 0.0:
        return cond
    if cond.ndim != 1:
        raise ValueError("Class conditions must have shape [batch_size].")
    if cond.is_floating_point() or cond.is_complex():
        raise TypeError("Class conditions must use an integer tensor dtype.")
    if probability == 1.0:
        return torch.full_like(cond, UNCONDITIONAL_CLASS_LABEL)

    drop_mask = torch.rand(cond.shape, device=cond.device) < probability
    return cond.masked_fill(drop_mask, UNCONDITIONAL_CLASS_LABEL)


def guided_prediction(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor | None,
    guidance_scale: float = 1.0,
) -> torch.Tensor:
    """Evaluate a model with classifier-free guidance.

    The project defines CFG as

        prediction = uncond + scale * (conditional - uncond).

    Consequently scale=0 is exactly unconditional and scale=1 is exactly
    conditional. Those boundary cases use only one model evaluation.
    """

    scale = validate_guidance_scale(guidance_scale)

    if cond is None or scale == 0.0:
        return model(x, t, cond=None)
    if scale == 1.0:
        return model(x, t, cond=cond)

    unconditional = model(x, t, cond=None)
    conditional = model(x, t, cond=cond)
    return unconditional + scale * (conditional - unconditional)
