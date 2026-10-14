from __future__ import annotations

import math

import torch
from torch import nn

from ugm.models.conditioning import UNCONDITIONAL_CLASS_LABEL
from ugm.models.unet import SinusoidalTimeEmbedding


def _sincos_1d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Return the standard sine/cosine embedding for scalar positions."""

    if dim % 2 != 0:
        raise ValueError("A 1D sine/cosine embedding dimension must be even.")
    frequencies = torch.arange(dim // 2, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0) * frequencies / max(dim // 2 - 1, 1)
    )
    angles = positions.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


def build_2d_sincos_position_embedding(
    grid_height: int,
    grid_width: int,
    hidden_size: int,
) -> torch.Tensor:
    """Build a fixed [1, H*W, D] 2D sine/cosine position embedding."""

    if hidden_size % 4 != 0:
        raise ValueError("DiT hidden_size must be divisible by 4.")
    y, x = torch.meshgrid(
        torch.arange(grid_height, dtype=torch.float32),
        torch.arange(grid_width, dtype=torch.float32),
        indexing="ij",
    )
    half = hidden_size // 2
    embedding = torch.cat(
        [_sincos_1d(y.reshape(-1), half), _sincos_1d(x.reshape(-1), half)],
        dim=1,
    )
    return embedding.unsqueeze(0)


class PatchEmbed(nn.Module):
    """Convert an image into non-overlapping patch tokens."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class DiTBlock(nn.Module):
    """Pre-norm Transformer block with adaLN-Zero conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    @staticmethod
    def _modulate(
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return x * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=1)
        )
        attention_input = self._modulate(self.norm1(x), shift_attn, scale_attn)
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        x = x + gate_attn[:, None, :] * attention_output
        mlp_input = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + gate_mlp[:, None, :] * self.mlp(mlp_input)


class DiTFinalLayer(nn.Module):
    """Condition, normalize, and project tokens back to pixel patches."""

    def __init__(self, hidden_size: int, patch_output_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        self.projection = nn.Linear(hidden_size, patch_output_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=1)
        x = self.norm(x) * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.projection(x)


class DiTTiny(nn.Module):
    """Small image Diffusion Transformer with the project's unified interface.

    The model predicts a tensor with the same spatial shape expected by DDPM,
    Flow Matching, Rectified Flow, and consistency-distillation objectives::

        prediction = model(x, t, cond=None)
    """

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int = 1,
        out_channels: int | None = None,
        patch_size: int = 4,
        hidden_size: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if image_size <= 0 or patch_size <= 0:
            raise ValueError("image_size and patch_size must be positive.")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if hidden_size <= 0 or hidden_size % 4 != 0:
            raise ValueError("hidden_size must be positive and divisible by 4.")
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")
        if depth <= 0 or mlp_ratio <= 0:
            raise ValueError("depth and mlp_ratio must be positive.")
        if num_classes is not None and num_classes <= 0:
            raise ValueError("num_classes must be positive when provided.")

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.num_classes = num_classes
        self.grid_size = self.image_size // self.patch_size

        self.patch_embed = PatchEmbed(
            self.in_channels,
            self.hidden_size,
            self.patch_size,
        )
        self.register_buffer(
            "position_embedding",
            build_2d_sincos_position_embedding(
                self.grid_size,
                self.grid_size,
                self.hidden_size,
            ),
        )
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.class_embed = (
            nn.Embedding(num_classes, self.hidden_size)
            if num_classes is not None
            else None
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.hidden_size, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTFinalLayer(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        def initialize(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize)
        nn.init.xavier_uniform_(
            self.patch_embed.projection.weight.flatten(1)
        )
        nn.init.zeros_(self.patch_embed.projection.bias)

        # adaLN-Zero starts every residual Transformer branch as an identity.
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.modulation[-1].weight)
        nn.init.zeros_(self.final_layer.modulation[-1].bias)
        nn.init.zeros_(self.final_layer.projection.weight)
        nn.init.zeros_(self.final_layer.projection.bias)

    def _conditioning(
        self,
        t: torch.Tensor,
        cond: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if t.ndim != 1 or t.shape[0] != batch_size:
            raise ValueError("t must have shape [batch_size].")
        if not torch.isfinite(t).all():
            raise ValueError("t contains non-finite values.")
        condition = self.time_embed(t)

        if self.class_embed is None:
            # Datasets may still expose labels while class_cond=False. Match
            # SimpleUNet and ignore them so the trainer stays model-agnostic.
            return condition
        if cond is None:
            return condition
        if cond.ndim != 1 or cond.shape[0] != batch_size:
            raise ValueError("Class conditions must have shape [batch_size].")
        if cond.is_floating_point() or cond.is_complex():
            raise TypeError("Class conditions must use an integer tensor dtype.")

        cond = cond.long()
        invalid = (cond < UNCONDITIONAL_CLASS_LABEL) | (cond >= self.num_classes)
        if torch.any(invalid):
            raise ValueError(
                "Class labels must be -1 (unconditional) or in "
                f"[0, {self.num_classes - 1}]."
            )
        null_mask = cond == UNCONDITIONAL_CLASS_LABEL
        class_condition = self.class_embed(cond.clamp_min(0))
        class_condition = class_condition.masked_fill(null_mask[:, None], 0.0)
        return condition + class_condition

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = patches.shape
        expected_tokens = self.grid_size * self.grid_size
        if token_count != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens, got {token_count}."
            )
        p = self.patch_size
        patches = patches.reshape(
            batch_size,
            self.grid_size,
            self.grid_size,
            p,
            p,
            self.out_channels,
        )
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.reshape(
            batch_size,
            self.out_channels,
            self.image_size,
            self.image_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, channels, height, width].")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}."
            )
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size} inputs, "
                f"got {tuple(x.shape[-2:])}."
            )

        tokens = self.patch_embed(x)
        tokens = tokens + self.position_embedding.to(dtype=tokens.dtype)
        condition = self._conditioning(t, cond, x.shape[0])
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.unpatchify(self.final_layer(tokens, condition))


__all__ = [
    "DiTBlock",
    "DiTFinalLayer",
    "DiTTiny",
    "PatchEmbed",
    "build_2d_sincos_position_embedding",
]
