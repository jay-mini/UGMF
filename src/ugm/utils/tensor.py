from __future__ import annotations
import torch

def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    x_shape: tuple[int, ...],
) -> torch.Tensor:
    """
    Extract timestep-dependent coefficients and reshaoe for broadcasting.

    Parameters
    ----------
    values:
        Tensor of shape [T].
    timesteps:
    I   Integer tensor of shape [B]
    x_shape:
        Target tensor shape, usually x.shape

    Returns
    -------
    torch.Tensor
        Tensor of shape [B, 1, 1, 1] for broadcasting.
    """

    if values.ndim != 1:
        raise ValueError(f"Expected values to be 1D, but got shape {values.shape}")
    
    if timesteps.ndim != 1:
        raise ValueError(f"Expected timesteps to be 1D, but got shape {timesteps.shape}")
    
    if timesteps.device != values.device:
        raise ValueError(f"Expected timesteps and values to be on the same device, but got {timesteps.device} and {values.device}")
    
    extracted = values.gather(0, timesteps)

    return extracted.reshape(
        timesteps.shape[0], *((1,) * (len(x_shape) - 1))
    )
