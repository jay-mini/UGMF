# Reproducible generative-model evaluation

`scripts/evaluate.py` is the unified evaluation entry for DDPM/DDIM and Flow
Matching Euler/Heun checkpoints. It exports lossless samples, benchmarks the
sampler, computes distribution metrics, and writes one structured JSON record.

## Installation

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe -m pip install -e ".[eval]"
```

The evaluation extra installs
[`torch-fidelity`](https://github.com/toshas/torch-fidelity). Version 0.4.0 was
used for the verified run. Its implementation is designed to match reference
FID/KID implementations to floating-point precision and includes generative
manifold Precision-Recall. The first standard-metric run downloads the official
InceptionV3 and VGG16 weights into PyTorch's model cache.

## Full evaluation command

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe -B scripts/evaluate.py `
  --ckpt outputs/ddpm_mnist_conditional/checkpoints/last.pt `
  --output_dir outputs/ddpm_mnist_conditional/evaluation_10k `
  --sampler ddim `
  --num_steps 50 `
  --num_samples 10000 `
  --num_reference 10000 `
  --batch_size 64 `
  --guidance_scale 3 `
  --metrics fid kid prc pixel_fd `
  --metric_batch_size 64 `
  --kid_subsets 100 `
  --kid_subset_size 1000 `
  --prc_neighborhood 3 `
  --prc_batch_size 1000 `
  --warmup_runs 2 `
  --repetitions 5 `
  --benchmark_batch_size 64 `
  --use_ema `
  --clip_x0
```

Use a new `output_dir` for each protocol. Existing PNG/CSV exports are rejected
to prevent stale samples from changing the metric population. `--overwrite`
explicitly replaces only the PNG/CSV files in the selected generated and
reference export directories.

## Export protocol

Generated files are stored as individual lossless PNG images in
`generated/`. Samples are converted from the model range `[-1, 1]` to uint8
`[0, 255]`. Class-conditional runs also save `generated/labels.csv`.

The default class policy cycles evenly through all classes. `--class_label N`
evaluates one class; by default the real reference set is filtered to the same
class. `--no-match_reference_condition` disables that filtering, but then the
conditional and unconditional protocols must not be compared as though they
were the same distribution.

Real MNIST/CIFAR10 images are selected without replacement using the recorded
seed and exported to `reference/` with their own `labels.csv`. The default is
the test split. Generated and real directories are both passed to the same
metric backend.

## Standard metrics

### FID

FID fits Gaussians to Inception feature distributions:

```text
FID = ||mu_r - mu_g||^2
      + Tr(C_r + C_g - 2 (C_r C_g)^(1/2)).
```

Lower is better. The implementation and `inception-v3-compat` feature
extractor come from torch-fidelity rather than local matrix code.

### KID

KID is the subset-averaged, unbiased squared MMD over Inception features using
the standard polynomial kernel. The output contains both the subset mean and
standard deviation. If a smoke run contains fewer images than the requested
subset size, the effective size is reduced and both requested/effective values
are recorded.

### Generative Precision-Recall

This is distribution Precision-Recall, not classifier Precision-Recall.
torch-fidelity constructs k-nearest-neighbour feature manifolds using VGG16:

- precision: fraction of generated samples covered by the real manifold;
- recall: fraction of real samples covered by the generated manifold;
- F-score: harmonic mean of these two values.

The JSON records the feature extractor, neighbourhood size, and all three
values. Precision is mainly a fidelity indicator; recall is mainly a coverage
or diversity indicator.

## Pixel-FD is a separate proxy

`pixel_fd_proxy` applies the Fréchet formula to 7x7 adaptive-average-pooled
grayscale pixels. It is cheap and useful for development regressions, but it
is not standard Inception FID. The result schema enforces the distinction:

```json
{
  "metrics": {
    "standard": {
      "standard_metrics": true,
      "fid": {},
      "kid": {},
      "generative_precision_recall": {}
    },
    "pixel_fd_proxy": {
      "standard_metric": false,
      "feature_definition": "7x7 ... grayscale pixels"
    }
  }
}
```

## Timing, NFE, and CUDA memory

Timing follows this order:

1. Run the requested number of full-sampler warm-ups.
2. Reuse one fixed initial-noise batch for every repetition.
3. Synchronize CUDA immediately before and after `perf_counter()` timing.
4. Record every repetition, plus mean, population standard deviation, median,
   minimum, maximum, and samples/second.
5. Reset CUDA peak-memory counters before every measured repetition.

The report distinguishes:

- peak allocated memory: live tensor memory including resident model state;
- peak reserved memory: CUDA allocator reservation;
- incremental peak allocated memory: measured peak minus the pre-run baseline.

NFE means neural-network forward evaluations along one sample trajectory. It
does not multiply by batch size:

```text
DDPM/DDIM NFE = steps * CFG branches
Euler NFE     = steps * CFG branches
Heun NFE      = 2 * steps * CFG branches
```

The CFG branch factor is one for unconditional sampling and for the optimized
boundary scales 0 and 1; it is two for conditional CFG at other scales. A
transparent model wrapper counts actual calls, and `nfe_verified` compares the
observed count with the formula on every repetition.

## Verified 256-sample smoke run

The end-to-end run used the class-conditional MNIST EMA checkpoint at global
step 1874, deterministic 50-step DDIM, CFG 3, batch size 64, one warm-up and
three repetitions on an NVIDIA GeForce RTX 3050 4GB Laptop GPU.

| Field | Observed value |
|---|---:|
| Standard FID | 154.14568 |
| Standard KID mean | 0.137739 |
| Standard KID std | 0.008893 |
| Generative precision | 0.003906 |
| Generative recall | 0.015625 |
| Pixel-FD proxy | 0.714247 |
| Mean time for 64 samples | 2.03836 s |
| Mean throughput | 31.4017 samples/s |
| NFE | 100 |
| Observed forward calls | 100, 100, 100 |
| Peak allocated CUDA memory | 99,095,552 bytes |
| Incremental peak allocated memory | 64,924,160 bytes |
| Peak reserved CUDA memory | 144,703,488 bytes |

The result is stored at:

```text
outputs/ddpm_mnist_conditional/evaluation_smoke_256/results.json
```

This validates the complete path but is not a publication-quality estimate.
Only 256 samples were used. In addition, standard Inception/VGG features were
trained on natural RGB images, while MNIST contains 28x28 grayscale digits.
Use at least 10,000 samples for a stable project comparison, keep every
sampling and metric setting fixed, and consider an independently trained
MNIST feature extractor/classifier for a domain-specific complementary metric.
