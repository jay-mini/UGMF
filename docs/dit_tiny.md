# DiT-Tiny

## Architecture

For an image `x` with shape `[B, C, H, W]`, a convolution with kernel and
stride `P` creates a grid of non-overlapping patches. The token count is

\[
N=(H/P)(W/P).
\]

Each token has `hidden_size` channels. A fixed two-dimensional sine/cosine
position embedding is added before the Transformer blocks. Continuous model
time is embedded with the same sinusoidal convention as the U-Net followed by
an MLP. For class-conditional models the class embedding is added to the time
condition; label `-1` contributes exactly zero and is the CFG null condition.

Every block uses adaLN-Zero conditioning:

\[
x' = x + g_a(c)\,\operatorname{Attention}
  (\operatorname{LN}(x)(1+s_a(c))+b_a(c)),
\]

\[
x'' = x' + g_m(c)\,\operatorname{MLP}
  (\operatorname{LN}(x')(1+s_m(c))+b_m(c)).
\]

The modulation and final projection are zero-initialized. Consequently the
model initially predicts zero, the first update trains the output head, and
later updates progressively open the conditioned residual branches. This is
intentional stabilization, not a broken gradient path.

The final linear layer emits `P*P*out_channels` values per token. Unpatchify
rearranges `[B, N, P*P*C_out]` back to `[B, C_out, H, W]`, preserving the
common `model(x, t, cond=None)` interface used by every objective and trainer.

## Configuration

```yaml
model_type: dit_tiny
dit_patch_size: 4
dit_hidden_size: 192
dit_depth: 6
dit_num_heads: 3
dit_mlp_ratio: 4.0
```

`image_size` must be divisible by `dit_patch_size`; `dit_hidden_size` must be
divisible by both four (2D sine/cosine positions) and `dit_num_heads`.
Configurations and checkpoints created before this model factory default to
`model_type: unet` when the key is absent.

## Verified smoke run

```powershell
python scripts/train.py `
  --config configs/experiment/fm_mnist_dit_tiny_smoke.yaml
```

The verified CUDA run used 4x4 patches, 49 tokens, hidden size 192, six blocks,
three heads, 4,153,168 parameters, and five FM optimizer steps. Losses were:

```text
step 1: 1.911495
step 2: 1.900342
step 3: 1.840768
step 4: 1.797079
step 5: 1.746748
```

All checkpoint tensors were finite, 70 of 71 floating state tensors differed
from initialization, and model, optimizer, scheduler, EMA, global step, and
effective configuration were saved. A five-step run verifies integration and
optimization only; its generated images are expected to remain noise-like.

## Checks

1. Confirm patch tokens have shape `[B, (H/P)*(W/P), D]`.
2. Confirm the positional tensor has exactly the same token count and width.
3. Confirm time and optional class embeddings both have width `D`.
4. Confirm `-1` class labels match the unconditional branch.
5. Confirm the output is exactly `[B, C_out, H, W]`.
6. Run at least two updates when auditing early gradients because adaLN-Zero
   intentionally exposes the output head first.
7. Confirm `model_type` and all `dit_*` fields survive checkpoint round-trip.
8. Treat a finite short-run loss as an integration test, not evidence of
   sample quality or convergence.
