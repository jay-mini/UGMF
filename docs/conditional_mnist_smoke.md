# Class-conditional MNIST AMP smoke verification

Verified on 2026-09-14 with PyTorch 2.5.1 and an NVIDIA GeForce RTX 3050 4GB
Laptop GPU.

## Reproducible command

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe -B scripts/train.py `
  --config configs/experiment/ddpm_mnist_conditional_smoke.yaml
```

The effective configuration enables:

```yaml
class_cond: true
cond_drop_prob: 0.1
guidance_scale: 3.0
mixed_precision: fp16
max_steps: 5
ema_update_after_step: 0
```

This is a pipeline and checkpoint smoke run, not a converged generative-model
quality result. Its 100-step linear beta schedule has
`alpha_bar_T ~= 0.3636`, so the final forward state is not close to pure
Gaussian noise and does not match the sampler's initial distribution. Do not
continue this checkpoint for quality evaluation; use
`configs/experiment/ddpm_mnist_conditional.yaml`, which uses 1000 diffusion
steps, instead. The diagnosis and successful formal run are recorded in
`conditional_sampling_evaluation.md`.

## Observed training log

| Global step | Loss | DDPM loss |
|---:|---:|---:|
| 1 | 1.121064 | 1.121064 |
| 2 | 1.081898 | 1.081898 |
| 3 | 1.054424 | 1.054424 |
| 4 | 1.002034 | 1.002034 |
| 5 | 0.979422 | 0.979422 |

Every logged loss was finite and the process exited successfully. A decreasing
loss over only five batches is encouraging but is not treated as evidence of
convergence.

## Checkpoint verification

The checkpoint is available at:

```text
outputs/ddpm_mnist_conditional_smoke/checkpoints/last.pt
```

Observed fields:

| Check | Value |
|---|---:|
| File size | 5,186,466 bytes |
| Global step | 5 |
| Saved epoch index | 0 (mid-epoch interruption) |
| Scheduler `last_epoch` | 5 |
| Current learning rate | 0.0001999964908278481 |
| EMA updates | 5 |
| GradScaler scale | 65536.0 |
| All saved model tensors finite | true |
| `config.json == checkpoint["config"]` | true |

The epoch index is zero because `max_steps=5` stops inside epoch zero. As
documented in `training_flow.md`, resuming this checkpoint restarts the current
epoch's DataLoader rather than restoring its batch cursor.

## Load-and-sample verification

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe -B scripts/sample.py `
  --ckpt outputs/ddpm_mnist_conditional_smoke/checkpoints/last.pt `
  --sampler ddim `
  --num_steps 10 `
  --num_samples 16 `
  --nrow 4 `
  --class_label 7 `
  --guidance_scale 3.0 `
  --use_ema `
  --output outputs/ddpm_mnist_conditional_smoke/class_7_smoke.png
```

The checkpoint loaded strictly, the EMA model sampled successfully, and the
script reported the effective DDIM step count of 10 and guidance scale of 3.0.
