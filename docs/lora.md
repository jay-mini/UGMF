# DiT-Tiny LoRA Fine-Tuning

## Scope and backbone

This implementation applies LoRA to the existing Flow-Matching DiT-Tiny
backbone. The verified base is:

```text
outputs/fm_mnist_dit_tiny_smoke/checkpoints/last.pt
base global_step = 5
state source = model (not EMA)
```

The injected modules are:

```text
blocks.*.mlp.0
blocks.*.mlp.2
final_layer.projection
```

For the six-block DiT this resolves to 13 linear modules. PyTorch's
`MultiheadAttention` stores Q/K/V as fused projection parameters rather than
ordinary child `nn.Linear` calls. This first explicit recipe therefore targets
the two MLP projections and output head; it does not claim to adapt Q/K/V.

## Mathematics

For a frozen base projection:

```text
y_base = W x + b
```

LoRA adds a rank-`r` residual:

```text
y = W x + b + (alpha / r) B A dropout(x)
```

with:

```text
W in R^(d_out x d_in)      frozen
A in R^(r x d_in)          trainable
B in R^(d_out x r)         trainable
```

Instead of updating `d_out * d_in` base weights, LoRA updates
`r * (d_in + d_out)` parameters. `A` uses Kaiming initialization and `B=0`, so
the low-rank residual is exactly zero at injection time. The injected network
therefore initially produces exactly the same output as the base network.

The smoke configuration uses:

```text
rank = 8
alpha = 16
scaling = alpha / rank = 2
dropout = 0.05 (training only)
```

## Freeze and optimizer boundary

`inject_lora()` first freezes the entire backbone and then creates only
`lora_A` and `lora_B` as trainable parameters. The optimizer is constructed
from `lora_parameters(model)`, not from `model.parameters()`.

Three independent checks enforce this boundary:

1. trainable parameter names must equal all and only `*.lora_A`/`*.lora_B`;
2. the optimizer receives only those tensors;
3. after fine-tuning, every parameter is compared byte-for-byte against its
   pre-training value, and any changed non-LoRA tensor aborts the run.

## Adapter serialization

The adapter directory contains only:

```text
adapter/
  adapter_config.json
  adapter_model.safetensors
```

The safetensors file contains 26 tensors: one `A` and one `B` for each of the
13 modules. It contains no frozen backbone weights. The JSON records:

- rank, alpha, dropout, target glob patterns;
- the exact resolved module names and tensor names;
- base checkpoint path, global step, selected state source, SHA-256, and model
  configuration.

Loading follows a strict order:

```text
construct identical DiT architecture
  -> load exact base checkpoint/state source
  -> inject modules from adapter metadata
  -> validate resolved module names
  -> load only A/B tensors
```

Using a different base checkpoint can produce valid shapes but different
outputs. The stored SHA-256 is therefore part of the adapter contract.

## Run

```powershell
D:\Anaconda\envs\brain_dynamic\python.exe scripts/finetune_lora.py `
  --config configs/experiment/lora_dit_mnist_smoke.yaml
```

The script writes:

- `config.json`: effective base and adapter configuration;
- `checkpoints/last.pt`: complete resumable training state;
- `adapter/adapter_model.safetensors`: portable adapter-only tensors;
- `adapter/adapter_config.json`: injection and base-model metadata;
- `lora_run.json`: structured parameter audit, reload result, and step log.

Programmatic reload uses:

```python
checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
model = build_model(checkpoint["config"])
model.load_state_dict(checkpoint["model"])
metadata = load_lora_adapter(model, adapter_directory)
model.eval()
```

Loading the adapter before the original base state is incorrect because
injection changes state-dict names such as `mlp.0.weight` into
`mlp.0.base_layer.weight`.

## Verified short run

The real CUDA smoke run on 2026-09-29 produced:

| Check | Result |
|---|---:|
| injected linear modules | 13 |
| total parameters after wrapping | 4,246,992 |
| trainable LoRA parameters | 93,824 |
| trainable fraction | 2.209% |
| adapter tensors | 26, only A/B |
| adapter safetensors size | 377,568 bytes |
| changed non-LoRA parameters | 0 |
| reload maximum absolute error | 0.0 |
| reload output | bitwise identical |

The three-step log is:

| LoRA step | learning rate | FM loss |
|---:|---:|---:|
| 1 | 1.0e-3 | 1.7253817 |
| 2 | 1.0e-3 | 1.6973556 |
| 3 | 1.0e-3 | 1.6976732 |

This establishes correct parameter-efficient fine-tuning and adapter
portability. Three stochastic mini-batches do not establish convergence or
sample-quality improvement.

## Review checklist

1. Confirm `B=0` makes pre/post-injection output exactly equal.
2. Inspect actual resolved module names, not just broad target patterns.
3. Confirm trainable names, optimizer tensor IDs, gradients, and changed
   tensors all identify only LoRA parameters.
4. Put both models in `eval()` before reload comparison so LoRA dropout is
   disabled.
5. Reload the same base checkpoint and the same `model` versus `ema` source.
6. Compare the base SHA-256 before treating an adapter as compatible.
7. Keep adapter-only export separate from the larger resumable optimizer
   checkpoint; they solve different operational needs.
8. Evaluate downstream fixed-noise samples before claiming that a lower
   mini-batch training loss improved generation quality.
