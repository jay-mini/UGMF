# FM consistency distillation

## Fixed scheme

This implementation deliberately fixes one scheme before adding variants:

- teacher: a frozen Flow Matching velocity model;
- time convention: noise at `t=0`, data at `t=1`;
- time grid: `t_n = n / N`, for `n = 0, ..., N`;
- teacher transition: one Euler step between adjacent times;
- online model: trainable student initialized from teacher weights;
- target model: frozen EMA copy of the student;
- loss: mean-squared consistency loss;
- boundary: exact identity at the data endpoint.

## Adjacent states and time direction

For data `x_1`, Gaussian noise `x_0`, and a sampled index
`n in {0, ..., N-1}`, construct

\[
x_{t_n}=(1-t_n)x_0+t_nx_1,
\qquad
t_n=\frac{n}{N},
\qquad
t_{n+1}=\frac{n+1}{N}.
\]

The frozen teacher takes one step toward the data endpoint:

\[
\hat x_{t_{n+1}}
=x_{t_n}+(t_{n+1}-t_n)
v_{\mathrm{teacher}}(x_{t_n},t_n,c).
\]

The strict inequality `t_next > t_current` is validated. Reversing it would
silently apply the diffusion convention to this repository's opposite FM time
orientation.

## Boundary parameterization

The raw student retains the FM velocity-network architecture. Its consistency
function is

\[
f_\theta(x,t,c)=x+(1-t)F_\theta(x,t,c).
\]

Therefore

\[
f_\theta(x,1,c)=x
\]

for every network parameter value. This form is also compatible with FM
initialization: if the path has constant velocity, `x_t + (1-t)v = x_1`.

## Distillation loss

Let `theta` be the online student and `bar(theta)` the EMA target. The loss is

\[
\mathcal L_{CD}
=\left\|
f_\theta(x_{t_n},t_n,c)
-\operatorname{stopgrad}\left[
f_{\bar\theta}(\hat x_{t_{n+1}},t_{n+1},c)
\right]
\right\|_2^2.
\]

The teacher transition and target branch execute under `torch.no_grad()`. Both
teacher and target parameters have `requires_grad=False`; only the online
student is passed to the optimizer.

After a successful optimizer update, and only on configured update steps,

\[
\bar\theta\leftarrow
\mu\bar\theta+(1-\mu)\theta.
\]

Target buffers are copied from the online student. The target state and update
counter are stored in the objective checkpoint. The frozen teacher is rebuilt
from its recorded source checkpoint rather than duplicated into every output
checkpoint.

## Run

```powershell
python scripts/distill.py `
  --config configs/experiment/consistency_distill.yaml
```

Short smoke run:

```powershell
python scripts/distill.py `
  --config configs/experiment/consistency_distill.yaml `
  --output_dir outputs/consistency_distill_mnist/smoke `
  --batch_size 16 `
  --num_epochs 1 `
  --max_steps 2 `
  --num_workers 0 `
  --log_every 1 `
  --scheduler none
```

Resume to a total global step of 3:

```powershell
python scripts/distill.py `
  --config configs/experiment/consistency_distill.yaml `
  --output_dir outputs/consistency_distill_mnist/smoke `
  --batch_size 16 `
  --num_epochs 1 `
  --max_steps 3 `
  --num_workers 0 `
  --scheduler none `
  --resume outputs/consistency_distill_mnist/smoke/checkpoints/last.pt
```

`max_steps` remains a total-step boundary. Mid-epoch resume restores the model,
optimizer, scaler, target state, target-update counter, and RNG, but restarts
the DataLoader epoch because its cursor is not saved.

## Verified smoke result

Teacher: `outputs/fm_mnist/checkpoints/last.pt`, online weights at global step
9360. The legacy checkpoint has no EMA.

```text
student step 1 loss:       0.002049
student step 2 loss:       0.004268
resumed step 3 loss:       0.003348
student changed tensors:   82
target changed tensors:    76
target updates after step3: 3
all student/target finite: true
```

The artifact is at
`outputs/consistency_distill_mnist/smoke/checkpoints/last.pt`.

This verifies the distillation mechanics; three optimizer steps do not
establish one-step generation quality.

## One/few-step sampling

The implemented sampler supports exactly `K in {1, 2, 4}` network evaluations.
It uses the same time direction and boundary function as training. Evaluation
times are

\[
t_i=\frac{i}{K},\qquad i=0,\ldots,K-1.
\]

The first input is fixed Gaussian noise. Each evaluation predicts a data-end
estimate

\[
\hat x_1^{(i)}=f_\theta(x_{t_i},t_i).
\]

Before a later evaluation, the previous estimate is re-noised consistently
with the FM interpolation:

\[
x_{t_{i+1}}
=(1-t_{i+1})z_{i+1}
+t_{i+1}\hat x_1^{(i)},
\qquad z_{i+1}\sim\mathcal N(0,I).
\]

For unconditional sampling, or CFG scales 0/1, one step has NFE 1, two steps
NFE 2, and four steps NFE 4. Conditional CFG at other scales evaluates both
unconditional and conditional branches, so NFE is `2K`. The default sampling
model is the checkpointed EMA target; the online student is also selectable.
Initial noise and every fresh re-noising draw are controlled by the saved seed.

Single-grid command:

```powershell
python scripts/sample.py `
  --ckpt outputs/consistency_distill_mnist/short_run/checkpoints/last.pt `
  --sampler consistency `
  --num_steps 4 `
  --consistency_model target `
  --seed 20260927 `
  --output outputs/consistency_distill_mnist/short_run/samples_4_step.png
```

Fixed-noise 1/2/4 comparison:

```powershell
python scripts/sample_consistency.py `
  --ckpt outputs/consistency_distill_mnist/short_run/checkpoints/last.pt `
  --output_dir outputs/consistency_distill_mnist/short_run/samples_1_2_4 `
  --num_samples 64 `
  --seed 20260927 `
  --consistency_model target `
  --device cuda
```

This writes the shared `initial_noise.pt`, individual 1/2/4-step grids, a
combined `comparison_grid.png`, and `manifest.json` containing checkpoint,
global step, model source, seed, algorithm, outputs, and NFE.

## Verified 20-step short training and sampling loop

The reproducible short configuration is
`configs/experiment/consistency_distill_mnist_short.yaml`.

```powershell
python scripts/distill.py `
  --config configs/experiment/consistency_distill_mnist_short.yaml
```

Observed training checkpoints were saved at student steps 10 and 20, plus
`last.pt`. The logged consistency loss was approximately:

```text
step 1:  0.002330
step 5:  0.000874
step 10: 0.000757
step 15: 0.000763
step 20: 0.000424
target updates at step 20: 20
```

Both the EMA target and online student successfully produced 1/2/4-step
grids. The honest short-run result is that one-step samples remain strongly
collapsed, while two- and four-step re-noising restores visible diversity and
four steps produces the clearest digits. This establishes the training-to-
sampling software loop, not convergence or a quality improvement claim.

## Inspection checklist

1. Confirm teacher parameters use `requires_grad=False` and have no `.grad`.
2. Confirm the optimizer contains only `student.parameters()`.
3. Confirm all sampled pairs satisfy `0 <= t_current < t_next <= 1`.
4. Test the final interval: `t_next == 1` and target boundary output equals its
   input exactly.
5. Confirm target output is detached and target parameters receive no gradient.
6. Update target only after a successful optimizer update; skipped AMP steps
   must not advance it.
7. Verify `num_target_updates`, student global step, optimizer, scheduler, and
   target parameters survive checkpoint resume.
8. Keep the teacher checkpoint path, global step, online/EMA source, time grid,
   target decay, and teacher guidance scale in the effective config.
