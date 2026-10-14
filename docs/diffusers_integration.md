# Diffusers component integration

## Scope

The first verified component is `diffusers.UNet2DModel`. The integration does
not pretend that UGMF's `SimpleUNet` state dictionary is architecture-compatible
with a Diffusers U-Net. Instead, it wraps a real Diffusers component and gives
it the same call contract used by UGMF objectives and trainers:

```python
prediction = model(x, t, cond=None)
```

## Time and output adaptation

UGMF objectives pass normalized time `t in [0, 1]`. A scheduler-based
Diffusers component commonly embeds integer-like indices in
`[0, num_train_timesteps - 1]`. The adapter therefore calls the component with

\[
t_{diffusers}=t_{ugmf}\,(T-1).
\]

`UNet2DModel` returns `UNet2DOutput`; the adapter extracts `.sample` and checks
that it has the same shape as the input tensor.

For a class-conditional component created by UGMF, one extra embedding is
reserved as the CFG null class. Project label `-1` and `cond=None` both map to
that index. Arbitrary downloaded conditional components need an explicitly
compatible null embedding before classifier-free guidance can be assumed.

## Serialization layout

```text
adapter/
├── ugmf_adapter_config.json
└── component/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

The component directory is standard Diffusers `save_pretrained` output. The
UGMF metadata records adapter version, component class, normalized-time scale,
null-class index, component subdirectory, and serialization mode.

## Offline verified round trip

```powershell
python scripts/export_diffusers.py `
  --output_dir outputs/diffusers/unet2d_roundtrip `
  --device cpu `
  --seed 20260927
```

The script performs:

1. create and save a small native `UNet2DModel` source component;
2. load it with `UNet2DModel.from_pretrained`;
3. call it through the unified adapter;
4. save the adapter and native component;
5. reload the saved adapter;
6. call it with the exact same tensors and compare outputs.

The verified result used shape `[2, 1, 28, 28]`, time scale 999, safetensors,
and produced finite outputs with maximum absolute reload error `0.0`.

## Loading an existing component

```python
from ugm.models.adapters.diffusers_adapter import DiffusersUNet2DAdapter

model = DiffusersUNet2DAdapter.from_diffusers_pretrained(
    "path/to/local/unet",
    local_files_only=True,
)
prediction = model(x, normalized_time)
model.save_pretrained("outputs/my_adapter")
reloaded = DiffusersUNet2DAdapter.from_pretrained("outputs/my_adapter")
```

Remote Hub identifiers use the same method with `local_files_only=False`.
Network download, model license, revision pinning, and cache location remain
the caller's responsibility.
