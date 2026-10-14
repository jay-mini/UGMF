# Rectified Flow reflow workflow

## Definition

Let a trained Flow Matching teacher define the numerical flow map

\[
x_1 = \Phi_{\mathrm{teacher}}(z, c), \qquad z \sim \mathcal N(0, I).
\]

Pair generation stores the exact coupling `(x0=z, x1=teacher_output, cond)`.
The Rectified Flow student then samples `t ~ Uniform(t_min, t_max)` and uses

\[
x_t = (1-t)x_0 + tx_1, \qquad u_t = x_1-x_0,
\]

\[
\mathcal L_{RF}(\theta)
= \mathbb E\|v_\theta(x_t,t,c)-(x_1-x_0)\|_2^2.
\]

The endpoint pairing is the defining difference from the independent-coupling
FM objective. Do not replace the stored `x0` with newly sampled noise during
fine-tuning.

## 1. Generate paired data from an FM teacher

```powershell
python scripts/generate_rf_pairs.py `
  --teacher_ckpt outputs/fm_mnist/checkpoints/last.pt `
  --output_dir outputs/rectified_flow_mnist/pairs `
  --num_samples 60000 `
  --batch_size 128 `
  --shard_size 1024 `
  --solver heun `
  --num_steps 100 `
  --seed 42
```

For a modern checkpoint with validated EMA, add `--use_ema`. Conditional
teachers reuse their saved architecture and accept `--guidance_scale` and
`--label_policy balanced|random`.

The output contains a `manifest.json` and one or more `pairs-xxxxx.pt` files.
Every shard contains `x0`, `x1`, and (for a conditional teacher) `cond`.
The manifest records the teacher path and step, online/EMA state, solver,
sampling steps, guidance scale, label policy, seed, shape, and shard index.
Fine-tuning shuffles shard order and the samples inside each shard, preserving
stochastic training order without pathological random disk reads.

Generation refuses to mix with existing shards unless `--overwrite` is
explicitly supplied. Shards and the manifest are written atomically.

## 2. Fine-tune the RF velocity field

```powershell
python scripts/finetune_rf.py `
  --config configs/experiment/rectified_flow_mnist.yaml
```

CLI options override YAML values. The final effective configuration is stored
in `config.json` and in every checkpoint. `init_checkpoint` loads model weights
only and starts a new optimizer/scheduler/EMA/global-step history. `resume`
restores a prior RF run in full. These operations are intentionally separate.

The RF checkpoint can use the ordinary ODE sampling and evaluation entry
points:

```powershell
python scripts/sample.py `
  --ckpt outputs/rectified_flow_mnist/finetune/checkpoints/last.pt `
  --sampler euler `
  --num_steps 10 `
  --use_ema `
  --output outputs/rectified_flow_mnist/finetune/samples_10_steps.png
```

## 3. Validation checklist

1. Construct `RectifiedFlowPairDataset(pairs_dir)` and check its length and
   keys. Conditional data must contain `cond`.
2. Compare `manifest.json` teacher provenance with the intended checkpoint.
3. Before fine-tuning, verify that student tensors exactly equal the selected
   online/EMA initialization state.
4. Run one optimizer step and require a finite `loss_rf`, global step 1, and
   at least one changed parameter tensor.
5. Inspect the RF checkpoint for model, optimizer, scheduler, scaler, EMA,
   objective, effective config, and global step.
6. Sample the checkpoint with fixed noise at several step counts. Compare
   quality and NFE against the same FM teacher using the evaluation entry.

## Verified smoke run

The repository's existing `outputs/fm_mnist/checkpoints/last.pt` was used to
generate 64 pairs with 20-step Heun integration. RF fine-tuning ran for two
optimizer steps:

```text
loss_rf step 1: 0.100811
loss_rf step 2: 0.098810
teacher global step: 9360
RF global step: 2
changed state tensors: 82
all saved model tensors finite: true
scheduler saved: true
EMA saved: true
```

Artifacts are under `outputs/rectified_flow_mnist/smoke_pairs` and
`outputs/rectified_flow_mnist/smoke_finetune`. This is an integration smoke
test, not evidence that two RF updates improve sample quality.

## Practical caveats

- Pair quality is capped by teacher quality and numerical solver accuracy.
- Reuse the exact initial noise as `x0`; regenerated noise destroys coupling.
- The generated endpoint may be outside `[-1, 1]`; do not clip training pairs
  unless clipping is explicitly part of the teacher flow definition.
- Conditional pairs must preserve their labels. CFG scale used to create the
  endpoints is part of the dataset definition and must be recorded.
- A straighter learned path can permit fewer ODE steps, but this must be shown
  with fixed-noise quality comparisons and standard metrics, not assumed from
  training loss alone.
- Repeating reflow requires generating a new paired dataset from the latest RF
  model; it is a new dataset/provenance stage, not merely more epochs.
