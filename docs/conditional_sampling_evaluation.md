# Conditional MNIST sampling evaluation

Verified on 2026-09-14 with PyTorch 2.5.1 and an NVIDIA GeForce RTX 3050 4GB
Laptop GPU.

## Evaluation design

The comparison must isolate the condition and guidance scale from random
variation. `scripts/sample_class_grid.py` therefore creates `K` latent-noise
samples and repeats the same `K` samples for every requested class. It also
reuses that entire batch at every guidance scale. With deterministic DDIM
(`eta=0`), cell `(class, noise column)` starts from exactly the same state in
all comparisons.

The CFG convention is

```text
epsilon_cfg = epsilon_uncond
              + scale * (epsilon_cond - epsilon_uncond).
```

Thus scale 0 must be label invariant, scale 1 is the normal conditional
prediction, and scales above 1 extrapolate along the learned class direction.

The script records a forward audit around the actual checkpoint model. This
distinguishes a real end-to-end condition path from a CLI that accepts labels
but silently discards them.

## Failed smoke baseline and diagnosis

The five-step smoke checkpoint was first continued to global step 500. Its
loss reached 0.122094, but both EMA and online-model samples remained
noise-like. The condition-forward audit passed and pixel differences increased
with guidance, but no digit was visually recognizable. This is recorded in:

```text
outputs/ddpm_mnist_conditional_smoke/eval_step_5/manifest.json
outputs/ddpm_mnist_conditional_smoke/eval_step_500/manifest.json
outputs/ddpm_mnist_conditional_smoke/eval_step_500_online/manifest.json
```

The reason is a terminal-distribution mismatch, not a missing label. With the
smoke configuration's 100-step linear beta schedule,

```text
alpha_bar_T = product_t (1 - beta_t) = 0.363563.
```

Training at the largest timestep therefore still retains
`sqrt(alpha_bar_T) ~= 0.603` times the clean-image signal. Sampling starts from
pure Gaussian noise, which does not match that training terminal distribution.
For the formal 1000-step schedule, `alpha_bar_T = 4.0358e-5`, which is close to
zero and makes the pure-noise initialization appropriate. Changing the
schedule changes the objective buffers and learned problem, so the formal run
was correctly started from scratch rather than resuming the smoke checkpoint.

## Formal training run

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe -B scripts/train.py `
  --config configs/experiment/ddpm_mnist_conditional.yaml
```

The run completed two epochs and 1874 optimizer updates. Representative logged
losses were 0.026365 at step 1000 and 0.022034 at step 1800. The final
checkpoint is:

```text
outputs/ddpm_mnist_conditional/checkpoints/last.pt
```

Checkpoint checks:

| Check | Observed value |
|---|---:|
| Global step | 1874 |
| Completed epoch count | 2 |
| Scheduler `last_epoch` | 1874 |
| Final learning rate | 0.0 |
| EMA updates | 1775 |
| GradScaler scale | 65536.0 |
| Model tensors finite | true |
| `config.json == checkpoint["config"]` | true |
| Checkpoint size | 20,237,794 bytes |

## Reproducible class and guidance comparison

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe -B scripts/sample_class_grid.py `
  --ckpt outputs/ddpm_mnist_conditional/checkpoints/last.pt `
  --output_dir outputs/ddpm_mnist_conditional/eval_step_1874 `
  --scales 0 1 3 5 `
  --samples_per_class 4 `
  --sampler ddim `
  --num_steps 50 `
  --eta 0 `
  --use_ema
```

The comparison grid uses the first fixed latent as columns `scale 0/1/3/5`;
the per-scale grids contain four fixed latent columns for every class row.

Observed pixel metrics:

| Guidance scale | Mean absolute label spread | Shift from scale 0 |
|---:|---:|---:|
| 0 | ~0 | 0 |
| 1 | 0.07057 | 0.06804 |
| 3 | 0.13732 | 0.13650 |
| 5 | 0.16462 | 0.17291 |

At scale 0, the maximum difference between class rows for a fixed latent was
exactly zero. The forward audit observed 50 unconditional and zero conditional
calls. At scale 1 it observed 50 conditional and zero unconditional calls. At
scales 3 and 5 it observed 50 calls to each branch, and every class 0--9 was
seen 200 times per scale (50 timesteps times four samples).

Visual inspection shows recognizable row-level class control after the formal
run. Scale 1 is conditional but can be ambiguous. Scale 3 gives the clearest
overall class identity in this small run. Scale 5 strengthens class-specific
strokes but sometimes over-saturates or distorts them. This is why guidance
strength must be tuned as a quality/diversity trade-off rather than maximized.

Pixel spread is a sensitivity diagnostic, not semantic accuracy. For a
production evaluation, add an independently trained MNIST classifier and
report conditional accuracy/confusion matrix together with FID/KID and visual
grids. Do not train and evaluate class correctness with the same generative
network.

## Files to inspect

```text
outputs/ddpm_mnist_conditional/eval_step_1874/guidance_comparison.png
outputs/ddpm_mnist_conditional/eval_step_1874/class_grid_scale_0.png
outputs/ddpm_mnist_conditional/eval_step_1874/class_grid_scale_1.png
outputs/ddpm_mnist_conditional/eval_step_1874/class_grid_scale_3.png
outputs/ddpm_mnist_conditional/eval_step_1874/class_grid_scale_5.png
outputs/ddpm_mnist_conditional/eval_step_1874/manifest.json
```
