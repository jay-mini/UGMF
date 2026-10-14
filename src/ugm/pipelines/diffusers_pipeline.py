from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ugm.models.adapters.diffusers_adapter import DiffusersUNet2DAdapter


class DiffusersComponentPipeline:
    """Small facade for calling and serializing an adapted component."""

    def __init__(self, model: DiffusersUNet2DAdapter) -> None:
        self.model = model

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.model.eval()
        return self.model(x, t, cond=cond)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        safe_serialization: bool = True,
    ) -> Path:
        return self.model.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_directory: str | Path,
        **load_kwargs: Any,
    ) -> "DiffusersComponentPipeline":
        return cls(
            DiffusersUNet2DAdapter.from_pretrained(
                pretrained_directory,
                **load_kwargs,
            )
        )


__all__ = ["DiffusersComponentPipeline"]
