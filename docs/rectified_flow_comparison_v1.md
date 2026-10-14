# Rectified Flow small-scale comparison v1

## Outcome

The current RF smoke checkpoint did **not** improve over its original FM
teacher at 4, 10, 20, or 50 Euler steps. This negative result is retained as
the first auditable baseline rather than being filtered or relabeled.

The result is not evidence against Rectified Flow as a method. The evaluated
RF checkpoint was trained on only 64 teacher pairs for two optimizer updates.
It was originally created to validate data loading, initialization, finite
loss, parameter updates, and checkpoint output.

## Clean comparison protocol

- FM checkpoint: `outputs/fm_mnist/checkpoints/last.pt`, global step 9360
- RF checkpoint: `outputs/rectified_flow_mnist/smoke_finetune/checkpoints/last.pt`, global step 2
- RF pair data: 64 endpoints, generated with seed 42 and 20-step Heun
- Evaluation seed: 20260920, deliberately different from pair seed 42
- Sampler: Euler for both models
- Sampling budgets: 4, 10, 20, and 50 steps; NFE equals steps
- Generated/reference samples: 512/512 for every row
- Same generated-noise sequence and reference subset for both models
- Online model weights for both checkpoints
- Benchmark: one warm-up and three synchronized repetitions, batch size 64
- Metrics: standard torch-fidelity FID/KID and separately named pixel-FD proxy

FID/KID estimates from 512 MNIST samples have high finite-sample uncertainty,
and their ImageNet Inception feature space is not ideal for MNIST. The table is
a regression-quality first pass, not a publication result.

## Results

| Model | Steps/NFE | Time mean±std (s, batch=64) | samples/s | FID ↓ | KID mean±std ↓ | pixel-FD proxy ↓ |
|---|---:|---:|---:|---:|---:|---:|
| FM | 4/4 | 0.1855±0.0012 | 345.07 | 42.4623 | 0.021921±0.002596 | 0.202605 |
| FM | 10/10 | 0.5205±0.0147 | 123.07 | 27.3850 | 0.006665±0.001477 | 0.059512 |
| FM | 20/20 | 1.0981±0.0118 | 58.29 | 26.5075 | 0.005914±0.001222 | 0.049570 |
| FM | 50/50 | 2.9043±0.0113 | 22.04 | 25.9413 | 0.005503±0.001005 | 0.049415 |
| RF | 4/4 | 0.2009±0.0022 | 318.54 | 119.9869 | 0.137901±0.004748 | 0.327617 |
| RF | 10/10 | 0.4854±0.0067 | 131.88 | 109.9804 | 0.126584±0.005159 | 0.161374 |
| RF | 20/20 | 1.0274±0.0085 | 62.30 | 100.9466 | 0.112855±0.004960 | 0.133031 |
| RF | 50/50 | 2.5817±0.0069 | 24.79 | 91.2723 | 0.098595±0.004285 | 0.114528 |

At matched steps, `RF - FM` FID is +77.52, +82.60, +74.44, and +65.33
for 4/10/20/50 steps respectively. Every quality delta has the wrong sign.
RF quality improves as its integration step count increases, but it does not
cross the FM curve even at 50 steps.

Wall-clock differences should not be interpreted as an RF speed advantage in
this run: both checkpoints use the same U-Net architecture and matched NFE.
The small timing differences are system/run-order variance and need more
interleaved repetitions before a performance claim.

## Visual inspection

`comparison_grid.png` uses the same eight noise inputs for every row. FM and RF
both produce recognizable digits, but RF shows stronger stroke/shape drift.
This agrees directionally with all three quantitative quality measures.

## Reproduction

```powershell
python scripts/compare_reflow.py `
  --fm_ckpt outputs/fm_mnist/checkpoints/last.pt `
  --rf_ckpt outputs/rectified_flow_mnist/smoke_finetune/checkpoints/last.pt `
  --output_dir outputs/rectified_flow_mnist/comparison_v1_clean `
  --steps 4 10 20 50 `
  --num_samples 512 `
  --num_reference 512 `
  --batch_size 64 `
  --kid_subsets 20 `
  --kid_subset_size 256 `
  --warmup_runs 1 `
  --repetitions 3 `
  --seed 20260920 `
  --device cuda
```

The script refuses to use the RF pair-generation seed for evaluation unless
the explicit `--allow_pair_seed_overlap` diagnostic override is supplied.

## Next experiment

Before drawing a method-level conclusion:

1. Generate at least several thousand disjoint teacher pairs.
2. Train until held-out pair loss stabilizes, not for a fixed two updates.
3. Use a smaller RF fine-tuning learning rate and monitor deviation from the
   initialized FM model.
4. Keep a held-out noise-pair validation split.
5. Repeat this exact 4/10/20/50-step protocol first at 512 samples, then run a
   10k standard-metric evaluation only after a candidate improves consistently.
