# Frozen-VAE Latent Diffusion

## What is implemented

The latent pipeline uses the pretrained Diffusers `AutoencoderKL` checkpoint
`stabilityai/sd-vae-ft-mse`, pinned to Hub revision
`31f26fdeee1355a5c34592e401dd41e45d25a493`. The VAE is a fixed
representation component:

- every VAE parameter has `requires_grad=False`;
- the adapter remains in `eval()` even when training starts;
- VAE encode/decode runs under `torch.no_grad()`;
- the optimizer, EMA, and training checkpoint contain only the latent
  generation model;
- the effective config records the VAE identifier, image conversion, resolved
  scale, downsample factor, and latent shape needed to reconstruct the run.

The relevant implementation is split by responsibility:

- `src/ugm/models/adapters/vae_adapter.py`: frozen VAE, image conversion,
  latent scaling, encode/decode;
- `src/ugm/trainers/latent_trainer.py`: encodes each pixel batch before the
  existing objective and optimizer step;
- `src/ugm/pipelines/latent_diffusion.py`: unified latent sampling and final
  decode;
- `scripts/train_latent.py`: configuration, training, checkpoint, fixed-noise
  sampling, artifacts, and invariant report.

## Mathematical contract

For a normalized image `x in [-1, 1]`, the pretrained encoder produces a
diagonal Gaussian posterior:

```text
q_phi(z_raw | x) = N(mu_phi(x), diag(sigma_phi(x)^2))
z_raw = mu_phi(x) + sigma_phi(x) * epsilon,  epsilon ~ N(0, I)
```

Training uses the VAE checkpoint's scale `s`, not the unscaled posterior:

```text
z_data = s * z_raw
```

For the verified checkpoint, `s = 0.18215`. With a DDPM epsilon objective:

```text
z_t = sqrt(alpha_bar_t) * z_data
    + sqrt(1 - alpha_bar_t) * epsilon_t

L = E ||epsilon_t - epsilon_theta(z_t, t, c)||^2
```

Sampling produces a model-space latent `z_generated`. Decoding must undo the
same scale exactly once:

```text
x_generated = D_phi(z_generated / s)
```

Omitting the encode multiplication or decode division changes the distribution
seen by the denoiser or the VAE and commonly causes unstable training, contrast
or saturation errors, and poor samples.

## MNIST shape conversion

The selected VAE expects RGB inputs and downsamples by 8, while MNIST is
`1x28x28`. The adapter applies this reversible fixed conversion:

```text
1x28x28 --repeat RGB--> 3x28x28 --pad background--> 3x32x32
         --VAE encoder--> 4x4x4
```

After decode it crops `32x32` back to `28x28`, averages RGB to one channel,
and clamps to `[-1, 1]`. Padding uses `-1`, which is black in the project's
normalization. This preserves digit geometry better than stretching 28 pixels
to 32 pixels.

This VAE was trained on natural RGB images, not MNIST. Its reconstruction is
usable for a systems check, but this domain mismatch can cap downstream sample
quality. For serious MNIST quality, validate reconstruction error first and
consider a grayscale VAE trained on MNIST while retaining the same interface.

## Run

Install the optional dependency and run the one-update real-checkpoint smoke
test:

```powershell
pip install -e ".[diffusers]"
python scripts/train_latent.py `
  --config configs/experiment/latent_ddpm_mnist_smoke.yaml
```

The first run downloads the pretrained VAE into
`outputs/huggingface_cache`. Later offline runs can set:

```yaml
vae_local_files_only: true
```

The run writes:

- `config.json`: complete effective configuration;
- `checkpoints/last.pt`: latent model, optimizer, objective, scheduler, scaler,
  EMA, RNG, counters, and config;
- `latent_run.json`: shapes, loss, parameter change, freeze checks, and paths;
- `samples/input.png`: source images;
- `samples/vae_reconstruction.png`: deterministic posterior-mean VAE roundtrip;
- `samples/generated.png`: sampled latent decoded to pixels;
- `samples/initial_noise.pt`: reproducible initial latent noise.

## Verified smoke result

The real pretrained-VAE run on 2026-09-29 produced:

| Check | Result |
|---|---:|
| pixel batch | `[4, 1, 28, 28]` |
| scaled latent batch | `[4, 4, 4, 4]` |
| training loss | `0.9234538` |
| generator max parameter change | `2.00033e-4` |
| VAE frozen | `true` |
| any VAE gradient | `false` |
| sampler | DDIM, 5 steps |
| decoded output | `[4, 1, 28, 28]`, finite |

This proves the software loop and gradient boundary. It does **not** prove
generation quality: a one-update denoiser is effectively untrained, and the
retained generated grid correctly looks noisy. Quality claims require a longer
run plus reconstruction, fixed-noise sample, FID/KID, and class-control checks.

## Review checklist

1. Confirm `z_raw * scaling_factor` occurs only after encode and division by
   the same value occurs only before decode.
2. Confirm image tensors are `[-1, 1]`, not `[0, 1]`, before the VAE.
3. Confirm checkpoint `in_channels=4` and `image_size=4` refer to latent space;
   use `pixel_in_channels=1` and `pixel_image_size=28` for pixel space.
4. Confirm the VAE parameter IDs are absent from every optimizer group and all
   VAE gradients remain `None`.
5. Confirm stochastic posterior samples are used for training, while the
   posterior mode is used for deterministic reconstruction inspection.
6. Compare source and VAE-reconstruction grids before blaming the diffusion
   model for information already lost by the autoencoder.
7. Compare samplers with the saved `initial_noise.pt`; otherwise solver changes
   are confounded with different noise.
8. Treat the saved Hub model identifier/revision and preprocessing as part of
   the model specification, not as incidental data-loader settings.

Mid-epoch resume has the same documented limitation as pixel-space training:
state is restored exactly, but the DataLoader cursor is not saved, so the
current epoch restarts from its first batch.
