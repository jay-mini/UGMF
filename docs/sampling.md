# Unified sampling interface

All samplers expose the same public call:

```python
result = sampler.sample(
    model=model,
    objective=objective,
    shape=(batch_size, channels, height, width),
    cond=condition,
    guidance_scale=guidance_scale,
    device=device,
    initial_noise=initial_noise,
    num_steps=num_steps,
    return_trajectory=True,
)
```

Without `return_trajectory`, the return value is the final tensor, preserving
the original API behavior. With it enabled, the return value is a
`SamplingResult` whose `trajectory[0]` is the initial noise,
`trajectory[-1]` is the final sample, and `timesteps` aligns one-to-one with
the trajectory. Consequently, an `N`-step sampler returns `N + 1` states.

`initial_noise` is validated, copied to the selected device, and cloned before
sampling. Reusing the same tensor is therefore the correct way to compare
solvers from the same initial-value problem. `cond` is checked for a matching
batch dimension and moved to the sampling device.

## Classifier-free guidance

The project uses the convention

```text
prediction_cfg = prediction_uncond
                 + guidance_scale
                 * (prediction_cond - prediction_uncond).
```

Therefore `guidance_scale=0` is exactly the unconditional branch and
`guidance_scale=1` is exactly the ordinary conditional branch. Values greater
than one extrapolate away from the unconditional prediction. The zero and one
boundary cases use one model evaluation; other scales normally require two.

During class-conditional training, each label is independently replaced by
the null label `-1` with probability `cond_drop_prob`. In `SimpleUNet`, this
null label contributes an exactly zero class embedding and is therefore
equivalent to passing `cond=None`. The class embedding table keeps its original
size, so older class-conditional checkpoints remain shape-compatible.

## ODE solvers

For a learned velocity field

```text
dx/dt = v_theta(x, t, cond),
```

Euler uses

```text
x_(n+1) = x_n + h v_theta(x_n, t_n).
```

Heun first predicts an endpoint and then averages the endpoint slopes:

```text
k_1 = v_theta(x_n, t_n)
x_predict = x_n + h k_1
k_2 = v_theta(x_predict, t_(n+1))
x_(n+1) = x_n + h (k_1 + k_2) / 2.
```

Euler is first-order accurate globally and costs one model evaluation per
step. Heun is second-order accurate globally and costs two evaluations per
step. A fair comparison must therefore report both step count and number of
function evaluations.

## Discrete diffusion step semantics

DDIM supports a runtime `num_steps` override because its update is defined for
a selected decreasing subsequence of the training schedule. Standard DDPM
ancestral sampling uses every trained transition. Passing fewer DDPM steps is
rejected rather than silently skipping indices, because a skipped transition
requires a different mean and variance derivation.

For stochastic sampling, fixed initial noise controls only the starting state.
DDPM and DDIM with `eta > 0` also draw transition noise, so exact reproducibility
additionally requires the same random-generator state. DDIM with `eta = 0` and
the ODE solvers are deterministic for fixed model, condition, steps, and
initial noise.

## Command-line use

The sampling script accepts a common `--num_steps`, `--initial_noise`,
`--guidance_scale`, and `--trajectory_output`. Select `--sampler euler` or
`--sampler heun` for Flow Matching, and `--sampler ddpm` or `--sampler ddim`
for a diffusion checkpoint.
The older `--num_sampling_steps` and `--ode_steps` flags remain as defaults for
backward compatibility; `--num_steps` overrides them for a particular run.
