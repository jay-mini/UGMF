# UGMF

用 PyTorch 实现的图像生成实验项目。在同一套训练和采样接口下比较 DDPM、Flow Matching，以及基于 FM teacher 的 reflow 和 consistency distillation。目前实验以 MNIST 为主。

## 当前进度

| 模块 | 已实现和验证的内容 |
| --- | --- |
| DDPM / Flow Matching | epsilon/x0 预测、速度场目标、时间区间检查 |
| 采样 | DDPM、DDIM、Euler、Heun；固定初始噪声、类别条件、可选轨迹 |
| 训练 | YAML 配置与命令行覆盖、AMP、梯度裁剪、EMA、cosine scheduler、checkpoint 恢复 |
| 类别条件 | MNIST 类别标签、随机条件丢弃、CFG；引导强度 0/1 的边界测试 |
| 评估 | 样本导出，torch-fidelity FID/KID/Precision-Recall，重复计时、NFE 和峰值显存 |
| Rectified Flow | teacher 端点配对、配对数据加载、从已有模型初始化、4/10/20/50 步对比 |
| Consistency Distillation | 冻结 teacher、student、EMA target、边界参数化、相邻时间损失和 1/2/4 步采样 |
| DiT-Tiny | patch embedding、二维位置编码、时间/类别条件、adaLN-Zero、输出还原 |
| Diffusers | UNet2DModel 调用适配、本地/Hub 加载、组件保存与重载 |
| Latent Diffusion | 冻结预训练 AutoencoderKL、编码与缩放、latent 训练、采样与解码 |
| LoRA | DiT MLP 和输出层注入、仅更新 A/B、adapter-only 保存与重载 |

这些功能的验证深度不同。DDPM/FM 有训练与采样实验；DiT、VAE latent 和 LoRA 主要完成了短训练验证。RF 的第一版小样本对比没有优于 FM；consistency student 的单步结果仍有塌缩。具体结果保留在 [RF 对比记录](docs/rectified_flow_comparison_v1.md) 和 [蒸馏记录](docs/consistency_distillation.md)。

CIFAR10 数据加载已接入，但 `configs/experiment/ddpm_cifar10.yaml` 仍是空配置。文本条件、多卡训练和完整预训练生成 pipeline 尚未完成；目录中保留了一些早期占位文件。

## 安装与测试

需要 Python 3.10 或以上版本。先按机器的 CUDA 环境安装匹配的 PyTorch/torchvision，再安装项目：

```sh
python -m pip install -e ".[dev,diffusers]"
python -m pytest -q
```

`diffusers` extra 同时提供 VAE 和 LoRA adapter 保存所需的依赖。标准生成指标另装：

```sh
python -m pip install -e ".[eval]"
```

依赖以 [pyproject.toml](pyproject.toml) 为准，`requirements.txt` 目前为空。首次使用预训练 VAE 或标准指标时会下载对应权重。CUDA AMP 测试需要 GPU；CPU 环境会跳过相关测试。

以下命令均在仓库根目录执行，使用单行写法，适用于 PowerShell 和常见 Unix shell。数据和实验产物分别写入 `data/`、`outputs/`，不随源码发布。新克隆的仓库需要先生成命令所依赖的 checkpoint。

## 训练和采样

### DDPM 与类别条件

```sh
python scripts/train.py --config configs/experiment/ddpm_mnist.yaml
python scripts/sample.py --ckpt outputs/ddpm_mnist/checkpoints/last.pt --sampler ddim --num_steps 50 --use_ema --clip_x0
```

先检查类别条件与 AMP 流程，可以运行五步配置：

```sh
python scripts/train.py --config configs/experiment/ddpm_mnist_conditional_smoke.yaml
```

正式类别训练及不同引导强度的网格：

```sh
python scripts/train.py --config configs/experiment/ddpm_mnist_conditional.yaml
python scripts/sample_class_grid.py --ckpt outputs/ddpm_mnist_conditional/checkpoints/last.pt --output_dir outputs/ddpm_mnist_conditional/class_grid --scales 0 1 3 5 --sampler ddim --num_steps 50 --eta 0 --use_ema
```

CFG 采用 `uncond + scale * (cond - uncond)`，因此 scale=0 对应无条件预测，scale=1 对应条件预测。网格的标签传递和效果检查见 [条件采样记录](docs/conditional_sampling_evaluation.md)。

### Flow Matching 与 DiT

```sh
python scripts/train.py --config configs/experiment/fm_mnist.yaml
python scripts/sample.py --ckpt outputs/fm_mnist/checkpoints/last.pt --sampler heun --num_steps 50
```

同一 trainer 可通过 `model_type` 选择 U-Net、DiT-Tiny 或 Diffusers UNet2DModel。DiT 的五步配置：

```sh
python scripts/train.py --config configs/experiment/fm_mnist_dit_tiny_smoke.yaml
python scripts/sample.py --ckpt outputs/fm_mnist_dit_tiny_smoke/checkpoints/last.pt --sampler euler --num_steps 4
```

FM 使用噪声 t=0、数据 t=1 的方向，Euler/Heun 沿时间增大的方向积分。DDPM/DDIM 使用反向去噪时间表。接口约定和解析 ODE 测试见 [采样说明](docs/sampling.md)。

### 配置与恢复

`train.py` 的配置优先级为：显式命令行参数 > YAML > 默认值。最终配置保存在运行目录的 `config.json` 和 checkpoint 中。

```sh
python scripts/train.py --config configs/experiment/ddpm_mnist_conditional_smoke.yaml --resume outputs/ddpm_mnist_conditional_smoke/checkpoints/last.pt --max_steps 10
```

`max_steps` 表示总 optimizer update 数。恢复时若已达到上限，训练不会再多走一步。模型、optimizer、scheduler、scaler、EMA、RNG 和 step 会恢复；epoch 中途保存的 checkpoint 没有 DataLoader 游标，恢复时会从该 epoch 的首批数据重新开始。

## Reflow 与蒸馏

先运行上面的 FM 训练，得到 teacher checkpoint，再生成配对数据：

```sh
python scripts/generate_rf_pairs.py --teacher_ckpt outputs/fm_mnist/checkpoints/last.pt --output_dir outputs/rectified_flow_mnist/pairs --num_samples 60000 --batch_size 128 --shard_size 1024 --solver heun --num_steps 100
python scripts/finetune_rf.py --config configs/experiment/rectified_flow_mnist.yaml
python scripts/sample.py --ckpt outputs/rectified_flow_mnist/finetune/checkpoints/last.pt --sampler euler --num_steps 10 --use_ema
```

配对文件记录噪声、teacher 输出及生成参数。`scripts/compare_reflow.py` 可用固定噪声比较多个步数。第一版实验仅用 64 对样本、两次更新，4/10/20/50 步的指标均未改善，见 [reflow 实现说明](docs/rectified_flow.md) 和 [对比表](docs/rectified_flow_comparison_v1.md)。

Consistency distillation 使用同一个 FM teacher：

```sh
python scripts/distill.py --config configs/experiment/consistency_distill_mnist_short.yaml
python scripts/sample_consistency.py --ckpt outputs/consistency_distill_mnist/short_run/checkpoints/last.pt --output_dir outputs/consistency_distill_mnist/short_run/samples_1_2_4 --num_samples 64 --seed 20260927 --consistency_model target
```

当前方案固定为相邻 FM 时间上的 teacher Euler 转移，边界函数为 `f(x,t)=x+(1-t)F(x,t)`。20 步短训练及 1/2/4 步采样结果见 [蒸馏说明](docs/consistency_distillation.md)。

## VAE、LoRA 和 Diffusers 组件

VAE latent 训练：

```sh
python scripts/train_latent.py --config configs/experiment/latent_ddpm_mnist_smoke.yaml
```

使用固定 revision 的 `stabilityai/sd-vae-ft-mse`。MNIST 从灰度扩展为 RGB，28×28 填充至 32×32，编码为 4×4×4 latent；训练前乘 scaling factor，解码前除回同一个值。已完成一批图像的训练和解码，VAE 无梯度，生成网络参数有更新。该配置只用于检查流程，细节见 [Latent Diffusion](docs/latent_diffusion.md)。

LoRA 依赖前面生成的 DiT checkpoint：

```sh
python scripts/finetune_lora.py --config configs/experiment/lora_dit_mnist_smoke.yaml
```

默认在每个 DiT block 的两层 MLP 及最终投影注入 rank-8 LoRA，共 13 层；当前不适配融合 QKV 权重。三步微调中可训练参数为 93,824（2.209%），原参数无变化，adapter 重载输出误差为 0。导出文件为 `adapter_config.json` 和 `adapter_model.safetensors`，加载时需要相同的 base 权重。说明见 [LoRA](docs/lora.md)。

Diffusers 组件往返检查：

```sh
python scripts/export_diffusers.py --output_dir outputs/diffusers/unet2d_roundtrip --device cpu --seed 20260927
```

该脚本验证 UNet2DModel 的调用、保存和重载；不负责将自定义 U-Net 权重转换为另一种架构。见 [Diffusers 适配](docs/diffusers_integration.md)。

## 评估

下面命令依赖已训练的条件 DDPM checkpoint：

```sh
python scripts/evaluate.py --ckpt outputs/ddpm_mnist_conditional/checkpoints/last.pt --output_dir outputs/ddpm_mnist_conditional/evaluation_10k --sampler ddim --num_steps 50 --num_samples 10000 --num_reference 10000 --batch_size 64 --guidance_scale 3 --metrics fid kid prc pixel_fd --warmup_runs 2 --repetitions 5 --use_ema --clip_x0
```

输出包括 PNG 样本、标签 CSV 和结构化 `results.json`。FID/KID/Precision-Recall 使用 torch-fidelity；`pixel_fd_proxy` 是像素特征上的非标准距离，单独记录，不能当作 FID。

性能记录包含 warm-up、CUDA 同步、重复计时、NFE 和峰值显存。比较模型时需固定样本数、参考集、初始噪声和权重来源。当前评估入口主要用于像素空间 DDPM/FM/RF checkpoint；latent 解码后的评估和 LoRA adapter 自动装载还需另外接入。详见 [评估协议](docs/evaluation.md)。

## 代码位置

| 目录 | 内容 |
| --- | --- |
| `configs/experiment/` | 可直接传给对应入口的实验配置 |
| `scripts/` | 训练、微调、采样、评估和组件导出 |
| `src/ugm/models/` | 模型构建、U-Net、DiT 和 adapter |
| `src/ugm/objectives/` | DDPM、FM、reflow 和 consistency 损失 |
| `src/ugm/samplers/` | 离散去噪、ODE 和少步采样 |
| `src/ugm/trainers/` | 训练循环及 latent/蒸馏扩展 |
| `src/ugm/evaluation/` | 指标与性能统计 |
| `tests/` | 数学边界、参数更新、采样和保存/恢复测试 |
| `docs/` | 实现细节与实验记录 |

模型接口为 `model(x, t, cond=None)`，训练目标接口为 `objective.compute_loss(model, batch)`。多数实验共享 BaseTrainer；VAE 编码和蒸馏 target 更新由对应 trainer 扩展处理。

接下来需要补充较长训练下的 RF、蒸馏、latent 和 LoRA 质量对比，完成 CIFAR10 实验配置，并将各分支的采样结果接到统一评估入口。已有短实验的数值只反映对应运行，不能据此判断不同方法的质量上限。
