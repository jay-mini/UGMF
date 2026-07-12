# Unified Generative Models

这是一个统一的生成模型工程仓库，目标是逐步兼容 DDPM、Flow Matching、Rectified Flow、Consistency Distillation、DiT、Latent Diffusion、Classifier-Free Guidance、LoRA 以及 Diffusers 生态。

本项目的核心目标不是只实现某一个生成模型，而是构建一个统一、可扩展、工程化的生成模型训练框架，方便后续进行模型训练、采样、微调、蒸馏和评估。

---

## 1. Project Goals

本项目计划支持以下功能：

- DDPM / DDIM 图像生成
- Flow Matching 训练与 ODE 采样
- Rectified Flow 微调
- Consistency Distillation 少步生成蒸馏
- DiT / U-Net / Latent U-Net 模型结构
- Classifier-Free Guidance
- LoRA 微调
- Latent Diffusion
- Diffusers 格式导出
- FID / KID / Precision-Recall / Speed-Quality 评估

---

## 2. Project Structure

```text
unified-generative-models/
├── configs/
│   ├── experiment/
│   │   ├── ddpm_mnist.yaml
│   │   ├── ddpm_cifar10.yaml
│   │   ├── fm_mnist.yaml
│   │   ├── rectified_flow_cifar10.yaml
│   │   └── consistency_distill.yaml
│   ├── model/
│   │   ├── unet_small.yaml
│   │   ├── unet_cifar.yaml
│   │   ├── dit_tiny.yaml
│   │   └── latent_unet.yaml
│   ├── data/
│   │   ├── mnist.yaml
│   │   ├── cifar10.yaml
│   │   └── image_folder.yaml
│   ├── objective/
│   │   ├── ddpm.yaml
│   │   ├── flow_matching.yaml
│   │   ├── rectified_flow.yaml
│   │   └── consistency_distillation.yaml
│   └── train/default.yaml
│
├── scripts/
│   ├── train.py
│   ├── sample.py
│   ├── evaluate.py
│   ├── distill.py
│   ├── finetune_rf.py
│   └── export_diffusers.py
│
├── src/
│   └── ugm/
│       ├── data/
│       │   ├── build.py
│       │   ├── image_datasets.py
│       │   └── transforms.py
│       │
│       ├── models/
│       │   ├── build.py
│       │   ├── unet.py
│       │   ├── dit.py
│       │   ├── time_embedding.py
│       │   ├── conditioning.py
│       │   ├── ema.py
│       │   └── adapters/
│       │       ├── lora.py
│       │       └── diffusers_adapter.py
│       │
│       ├── objectives/
│       │   ├── base.py
│       │   ├── ddpm.py
│       │   ├── flow_matching.py
│       │   ├── rectified_flow.py
│       │   └── consistency.py
│       │
│       ├── samplers/
│       │   ├── base.py
│       │   ├── ddpm.py
│       │   ├── ddim.py
│       │   ├── ode.py
│       │   ├── rectified_flow.py
│       │   └── consistency.py
│       │
│       ├── trainers/
│       │   ├── base_trainer.py
│       │   ├── diffusion_trainer.py
│       │   ├── distill_trainer.py
│       │   └── finetune_trainer.py
│       │
│       ├── evaluation/
│       │   ├── fid.py
│       │   ├── kid.py
│       │   ├── precision_recall.py
│       │   └── speed_quality.py
│       │
│       ├── utils/
│       │   ├── config.py
│       │   ├── checkpoint.py
│       │   ├── logger.py
│       │   ├── seed.py
│       │   ├── distributed.py
│       │   └── visualization.py
│       │
│       └── pipelines/
│           ├── base_pipeline.py
│           ├── image_generation.py
│           ├── latent_diffusion.py
│           └── diffusers_pipeline.py
│
├── tests/
│   ├── test_objectives.py
│   ├── test_samplers.py
│   └── test_training_step.py
│
├── notebooks/
│   ├── 00_toy_data.ipynb
│   ├── 01_ddpm_mnist.ipynb
│   └── 02_fm_toy.ipynb
│
├── README.md
├── pyproject.toml
└── requirements.txt
```

---

## 3. Module Design

### 3.1 `configs/`

用于管理实验配置、模型配置、数据配置、目标函数配置和训练配置。

建议采用 Hydra 或 OmegaConf 管理配置，避免在代码中硬编码参数。

主要子目录：

- `configs/experiment/`: 完整实验配置
- `configs/model/`: 模型结构配置
- `configs/data/`: 数据集配置
- `configs/objective/`: 训练目标配置
- `configs/train/`: 优化器、学习率、batch size、训练轮数等通用训练配置

---

### 3.2 `scripts/`

项目入口脚本。

| Script | Function |
| --- | --- |
| `train.py` | 统一训练入口 |
| `sample.py` | 从训练好的模型中采样 |
| `evaluate.py` | 计算 FID、KID 等指标 |
| `distill.py` | Consistency Distillation 蒸馏 |
| `finetune_rf.py` | Rectified Flow 微调 |
| `export_diffusers.py` | 导出为 Diffusers 兼容格式 |

---

### 3.3 `src/ugm/data/`

数据加载与预处理模块。

| File | Function |
| --- | --- |
| `build.py` | 根据配置构建 dataset 和 dataloader |
| `image_datasets.py` | MNIST、CIFAR10、ImageFolder 等数据集封装 |
| `transforms.py` | 图像预处理、归一化、增强等 |

---

### 3.4 `src/ugm/models/`

模型结构模块。

| File | Function |
| --- | --- |
| `build.py` | 根据配置构建模型 |
| `unet.py` | DDPM / FM 常用 U-Net |
| `dit.py` | Diffusion Transformer |
| `time_embedding.py` | 时间嵌入模块 |
| `conditioning.py` | 条件生成、类别条件、文本条件接口 |
| `ema.py` | Exponential Moving Average |
| `adapters/lora.py` | LoRA 适配器 |
| `adapters/diffusers_adapter.py` | Diffusers 格式适配 |

---

### 3.5 `src/ugm/objectives/`

训练目标函数模块。

| File | Function |
| --- | --- |
| `base.py` | 统一 objective 接口 |
| `ddpm.py` | DDPM 噪声预测目标 |
| `flow_matching.py` | Flow Matching 速度场学习目标 |
| `rectified_flow.py` | Rectified Flow 训练/微调目标 |
| `consistency.py` | Consistency Training / Consistency Distillation 目标 |

统一接口建议：

```python
loss = objective.training_loss(model, batch, cond=None)
```

这样不同生成模型只需要替换 objective，而训练器可以尽量复用。

---

### 3.6 `src/ugm/samplers/`

采样器模块。

| File | Function |
| --- | --- |
| `base.py` | 统一 sampler 接口 |
| `ddpm.py` | DDPM 随机采样 |
| `ddim.py` | DDIM 确定性采样 |
| `ode.py` | ODE solver 采样 |
| `rectified_flow.py` | Rectified Flow 采样 |
| `consistency.py` | Consistency Model 少步采样 |

统一接口建议：

```python
samples = sampler.sample(model, shape, cond=None, num_steps=50)
```

---

### 3.7 `src/ugm/trainers/`

训练流程模块。

| File | Function |
| --- | --- |
| `base_trainer.py` | 通用训练循环 |
| `diffusion_trainer.py` | DDPM / FM / RF 通用训练 |
| `distill_trainer.py` | 蒸馏训练 |
| `finetune_trainer.py` | 微调训练 |

---

### 3.8 `src/ugm/evaluation/`

评估模块。

| File | Function |
| --- | --- |
| `fid.py` | FID 计算 |
| `kid.py` | KID 计算 |
| `precision_recall.py` | 生成分布覆盖度与精度 |
| `speed_quality.py` | 采样速度与质量对比 |

---

### 3.9 `src/ugm/utils/`

工具函数模块。

| File | Function |
| --- | --- |
| `config.py` | 配置读取与合并 |
| `checkpoint.py` | 模型保存与加载 |
| `logger.py` | 日志记录 |
| `seed.py` | 随机种子控制 |
| `distributed.py` | 多卡训练工具 |
| `visualization.py` | 图像可视化、采样轨迹可视化 |

---

### 3.10 `src/ugm/pipelines/`

高级推理流程模块。

| File | Function |
| --- | --- |
| `base_pipeline.py` | 统一 pipeline 接口 |
| `image_generation.py` | 像素空间图像生成 |
| `latent_diffusion.py` | Latent Diffusion 推理流程 |
| `diffusers_pipeline.py` | 与 Hugging Face Diffusers 生态兼容 |

---

## 4. Development Roadmap

### Stage 1: Minimal DDPM on MNIST

目标：先跑通最小闭环。

- [ ] 实现 MNIST dataloader
- [ ] 实现 small U-Net
- [ ] 实现 DDPM objective
- [ ] 实现 DDPM sampler
- [ ] 实现基本训练循环
- [ ] 保存 checkpoint
- [ ] 从 checkpoint 采样生成图片

对应实验：

```bash
python scripts/train.py --config configs/experiment/ddpm_mnist.yaml
python scripts/sample.py --ckpt outputs/ddpm_mnist/checkpoints/latest.pt
```

---

### Stage 2: Flow Matching on MNIST / Toy Data

目标：在同一训练框架下兼容 Flow Matching。

- [ ] 实现 Flow Matching objective
- [ ] 实现 ODE sampler
- [ ] 复用 U-Net 或 MLP velocity model
- [ ] 保存中间轨迹
- [ ] 可视化 learned velocity field

对应实验：

```bash
python scripts/train.py --config configs/experiment/fm_mnist.yaml
python scripts/sample.py --ckpt outputs/fm_mnist/checkpoints/latest.pt
```

---

### Stage 3: CIFAR10 DDPM / DDIM

目标：扩展到更标准的图像生成任务。

- [ ] 实现 CIFAR10 dataloader
- [ ] 实现更大的 U-Net
- [ ] 加入 EMA
- [ ] 加入 DDIM sampler
- [ ] 加入 FID / KID 评估

对应实验：

```bash
python scripts/train.py --config configs/experiment/ddpm_cifar10.yaml
python scripts/evaluate.py --ckpt outputs/ddpm_cifar10/checkpoints/latest.pt
```

---

### Stage 4: Rectified Flow Finetuning

目标：在已有 diffusion / flow model 基础上做 Rectified Flow 微调。

- [ ] 实现 rectified flow objective
- [ ] 支持从已有 checkpoint 初始化
- [ ] 支持 trajectory reflow
- [ ] 比较 RF 微调前后的采样步数与质量

对应实验：

```bash
python scripts/finetune_rf.py --config configs/experiment/rectified_flow_cifar10.yaml
```

---

### Stage 5: Consistency Distillation

目标：实现少步生成蒸馏。

- [ ] 加载 teacher model
- [ ] 实现 student model
- [ ] 实现 consistency distillation loss
- [ ] 支持 1-step / 2-step / 4-step 采样
- [ ] 比较 teacher 与 student 的 speed-quality tradeoff

对应实验：

```bash
python scripts/distill.py --config configs/experiment/consistency_distill.yaml
```

---

### Stage 6: DiT and Latent Diffusion

目标：扩展到更现代的生成模型结构。

- [ ] 实现 DiT-Tiny
- [ ] 实现 patch embedding
- [ ] 实现 latent U-Net
- [ ] 支持 VAE latent space
- [ ] 支持 latent diffusion training
- [ ] 支持 classifier-free guidance

---

### Stage 7: LoRA and Diffusers Compatibility

目标：提升工程兼容性。

- [ ] 实现 LoRA adapter
- [ ] 支持冻结 backbone 只训练 LoRA
- [ ] 支持导出 Hugging Face Diffusers 格式
- [ ] 支持从 Diffusers checkpoint 加载权重

---

## 5. Current Progress

| Date | Progress | Notes |
| --- | --- | --- |
| 2026-07-11 | Created project structure | Initial unified generative model framework |
| 2026-07-11 | Added README.md | Recorded project goals and roadmap |
| YYYY-MM-DD |  |  |
| YYYY-MM-DD |  |  |


| Component | Status | Notes |
|---|---|---|
| Simple U-Net | Implemented | Forward pass available |
| DDPM objective | Partial | Training objective implemented |
| DDPM ancestral sampler | Partial | Requires correctness fixes |
| Flow Matching objective | Partial | Time-range sampling requires fix |
| Euler ODE sampler | Partial | Requires numerical tests |
| Base trainer | Partial | AMP/checkpoint available; EMA missing |
| DDIM sampler | Not implemented | Planned for Day 5 |
| EMA | Not implemented | Planned for Day 4 |
| YAML configuration | Not implemented | Planned for Day 7 |
| Consistency model | Placeholder | Out of current scope |
| DiT | Placeholder | Out of current scope |

---

## 6. Notes

### 6.1 Core Design Principle

本项目的核心设计原则是：

```text
Data + Model + Objective + Sampler + Trainer + Evaluation
```

不同生成模型之间应尽量共享训练框架，只替换关键模块。

例如：

| Model Type | Objective | Sampler |
| --- | --- | --- |
| DDPM | noise prediction | DDPM / DDIM sampler |
| Flow Matching | velocity prediction | ODE sampler |
| Rectified Flow | straightened velocity prediction | RF sampler |
| Consistency Model | consistency loss | 1-step / few-step sampler |

---

### 6.2 Recommended Unified Objective Interface

```python
class BaseObjective:
    def training_loss(self, model, batch, cond=None):
        raise NotImplementedError

    def predict(self, model, x_t, t, cond=None):
        raise NotImplementedError
```

---

### 6.3 Recommended Unified Sampler Interface

```python
class BaseSampler:
    def sample(self, model, shape, cond=None, num_steps=50):
        raise NotImplementedError
```

---

### 6.4 Recommended Unified Trainer Interface

```python
class BaseTrainer:
    def train(self):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self, path):
        raise NotImplementedError
```

---

## 7. Environment Setup

建议使用 Conda 或 venv 创建独立环境。

```bash
conda create -n ugm python=3.10
conda activate ugm
pip install -r requirements.txt
```

开发模式安装：

```bash
pip install -e .
```

---

## 8. Example Commands

### Train DDPM on MNIST

```bash
python scripts/train.py --config configs/experiment/ddpm_mnist.yaml
```

### Sample Images

```bash
python scripts/sample.py --ckpt outputs/ddpm_mnist/checkpoints/latest.pt
```

### Evaluate Model

```bash
python scripts/evaluate.py --ckpt outputs/ddpm_cifar10/checkpoints/latest.pt
```

### Distill Consistency Model

```bash
python scripts/distill.py --config configs/experiment/consistency_distill.yaml
```

---

## 9. TODO

- [ ] 创建基础项目结构
- [ ] 编写 `pyproject.toml`
- [ ] 编写 `requirements.txt`
- [ ] 实现配置系统
- [ ] 实现 MNIST dataloader
- [ ] 实现 U-Net
- [ ] 实现 DDPM objective
- [ ] 实现 DDPM sampler
- [ ] 实现训练器
- [ ] 实现采样脚本
- [ ] 添加 Flow Matching objective
- [ ] 添加 ODE sampler
- [ ] 添加 Rectified Flow finetune
- [ ] 添加 Consistency Distillation
- [ ] 添加 FID / KID 评估
- [ ] 添加 DiT
- [ ] 添加 Latent Diffusion
- [ ] 添加 LoRA
- [ ] 添加 Diffusers export
