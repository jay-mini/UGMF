from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


def _valid_group_count(num_channels: int, max_groups: int = 8) -> int:
    """
    Find a valid GroupNorm group count.
    """

    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding.

    Input:
        t: shape[B], usually normalized to [0, 1].

    Output:
        embedding: shape[B, dim].
    """

    def __init__(
            self,
            dim: int,
            scale: float = 1000.0,
    ):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]

        t = t.float() * self.scale

        half_dim = self.dim // 2
        if half_dim <= 1:
            raise ValueError(f"Time embedding dimension is too small.")
        
        exponent = -math.log(10000.0) * torch.arange(
            half_dim,
            device=t.device,
            dtype=torch.float32,
        ) / float(half_dim - 1)

        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None, :]

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb


class ResBlock(nn.Module):
    """
    Residual block with time conditioning.

    The time embedding is projected and added after the first convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(
            num_groups = _valid_group_count(in_channels),
            num_channels = in_channels,
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = 3,
            padding = 1,
        )

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(
            num_groups = _valid_group_count(out_channels),
            num_channels = out_channels,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size = 3,
            padding = 1,
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size = 1,
            )

    def forward(
            self,
            x: torch.Tensor,
            time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        time_bias = self.time_proj(F.silu(time_emb))
        h = h + time_bias[:, :, None, None]

        h = self.conv2(F.silu(self.norm2(h)))

        return h + self.skip(x)
    

class Downsample(nn.Module):
    """
    Downsampling layer using strided convolution.
    """

    def __init__(
            self,
            channels: int,
            out_channels: int,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            out_channels,
            kernel_size = 4,
            stride = 2,
            padding = 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    

class Upsample(nn.Module):
    """
    Upsampling layer using transposed convolution.
    """

    def __init__(
            self,
            channels: int,
            out_channels: int,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            channels,
            out_channels,
            kernel_size = 4,
            stride = 2,
            padding = 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    

class SimpleUNet(nn.Module):
    """
    A compact U-Net for MNIST/CIFAR-10 scale experiments.

    Unified interface:

        pred = model(x, t, cond=None)

    For DDPM:
        pred means noise epsilong.

    For Flow Matching:
        pred means velocity field v_t.

        
    For Consistency Models:
        pred can later mean the consistency function output.
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int | None = None,
            base_channels: int = 64,
            time_dim: int = 256,
            num_classes: int | None = None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels
        
        self.inchannels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.num_classes = num_classes

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_dim)
        else:
            self.class_emb = None

        c = base_channels

        self.input_conv = nn.Conv2d(
            in_channels,
            c,
            kernel_size = 3,
            padding = 1,
        )

        self.down_res1 = ResBlock(c, c, time_dim)
        self.downsample1 = Downsample(c, c * 2)

        self.down_res2 = ResBlock(c * 2, c * 2, time_dim)
        self.downsample2 = Downsample(c * 2, c * 4)

        self.mid_res1 = ResBlock(c * 4, c * 4, time_dim)
        self.mid_res2 = ResBlock(c * 4, c * 4, time_dim)

        self.upsample2 = Upsample(c * 4, c * 2)
        self.up_res2 = ResBlock(c * 4, c * 2, time_dim)

        self.upsample1 = Upsample(c * 2, c)
        self.up_res1 = ResBlock(c * 2, c, time_dim)

        self.output_norm = nn.GroupNorm(
            num_groups = _valid_group_count(c),
            num_channels = c,
        )

        self.output_conv = nn.Conv2d(
            c,
            out_channels,
            kernel_size = 3,
            padding = 1,
        )

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: torch.Tensor | None = None,
        ) -> torch.Tensor:

        time_emb = self.time_mlp(t)

        if self.class_emb is not None and cond is not None:
            time_emb = time_emb + self.class_emb(cond.long())

        h0 = self.input_conv(x)

        h1 = self.down_res1(h0, time_emb)
        d1 = self.downsample1(h1)

        h2 = self.down_res2(d1, time_emb)
        d2 = self.downsample2(h2)

        h = self.mid_res1(d2, time_emb)
        h = self.mid_res2(h, time_emb)

        u2 = self.upsample2(h)
        u2 = self._match_spatial_size(u2, h2)
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.up_res2(u2, time_emb)

        u1 = self.upsample1(u2)
        u1 = self._match_spatial_size(u1, h1)
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.up_res1(u1, time_emb)

        out = self.output_conv(F.silu(self.output_norm(u1)))
        return out


    @staticmethod
    def _match_spatial_size(
        x: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make x have the same H, W as ref.

        This avoids shape mismatch when image sizes are not perfectly divisible by powers of 2.
        """

        if x.shape[-2:] == ref.shape[-2:]:
            return x
        
        return F.interpolate(
            x,
            size = ref.shape[-2:],
            mode = 'nearest',
        )

 