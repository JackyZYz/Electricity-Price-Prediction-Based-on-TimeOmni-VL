# Bagel-7B 获取方式与可行性调研报告

> **调研目标**：确认 TimeOmni-VL 论文中使用的 Bagel-7B backbone 是否可公开获取、如何下载、如何训练/微调，以及是否适合用户侧日前电价预测场景。  
> **调研时间**：2026-07-03  
> **结论**：Bagel-7B（ByteDance-Seed/BAGEL-7B-MoT）已完全开源，可直接从 Hugging Face 下载，代码仓库完整，支持理解与生成任务的联合微调。推荐作为复现 backbone。

---

## 1. Bagel-7B 基本信息

| 项目 | 内容 |
|---|---|
| **模型名称** | BAGEL-7B-MoT |
| **发布机构** | ByteDance Seed |
| **论文** | Emerging Properties in Unified Multimodal Pretraining (arXiv:2505.14683) |
| **开源协议** | Apache 2.0 |
| **模型规模** | 7B active parameters，14B total（MoT 架构） |
| **Hugging Face** | https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT |
| **GitHub 仓库** | https://github.com/ByteDance-Seed/Bagel |
| **Demo** | https://demo.bagel-ai.org/ |
| **项目主页** | https://bagel-ai.org/ |

---

## 2. 获取方式

### 2.1 模型权重下载

Bagel-7B 权重托管在 Hugging Face，总大小约 **29.6 GB**，包含：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors.index.json` | 123 kB | 分片索引 |
| `ema.safetensors` | 29.2 GB | EMA 权重（训练/推理推荐） |
| `ae.safetensors` | 335 MB | FLUX.1-schnell VAE |
| `tokenizer.json` / `tokenizer_config.json` | ~7 MB | Qwen2.5 分词器 |
| `config.json` / `llm_config.json` / `vit_config.json` | 数 KB | 模型配置 |

**推荐下载方式**：

```python
from huggingface_hub import snapshot_download

save_dir = "models/BAGEL-7B-MoT"
repo_id = "ByteDance-Seed/BAGEL-7B-MoT"
cache_dir = save_dir + "/cache"

snapshot_download(
    cache_dir=cache_dir,
    local_dir=save_dir,
    repo_id=repo_id,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
```

> **注意**：需要 Hugging Face 账号并配置 `HF_TOKEN` 或本地登录，但仓库为公开（public），无需特殊申请。

### 2.2 代码仓库下载

```bash
git clone https://github.com/bytedance-seed/BAGEL.git
cd BAGEL
conda create -n bagel python=3.10 -y
conda activate bagel
pip install -r requirements.txt
pip install flash_attn==2.5.8 --no-build-isolation
```

### 2.3 关键依赖

| 依赖 | 说明 |
|---|---|
| Python 3.10 | 推荐版本 |
| PyTorch + CUDA | 需支持 GPU |
| flash-attn 2.5.8 | 必须，安装较耗时 |
| transformers / accelerate | Hugging Face 生态 |
| PIL / torchvision | 图像处理 |

---

## 3. 模型架构与能力

### 3.1 架构特点

Bagel 是一个统一的解码器模型（decoder-only），支持：

- **多模态理解**：输入图像 + 文本，输出文本；
- **多模态生成**：输入文本/图像，输出生成图像；
- **图像编辑**：输入图像 + 文本指令，输出编辑后图像；
- **交错理解与生成**：同一个对话中交替输出文本和图像。

关键技术组件：

| 组件 | 来源 | 作用 |
|---|---|---|
| LLM | Qwen2.5-7B-Instruct | 文本理解与生成 |
| ViT | siglip-so400m-14-384 | 图像理解编码 |
| VAE | FLUX.1-schnell VAE | 图像生成/解码 |
| MoT | Mixture-of-Transformer-Experts | 扩展模型容量 |
| NaViT | Patch n' Pack | 任意分辨率图像处理 |

### 3.2 与 TimeOmni-VL 的关系

TimeOmni-VL 论文明确使用 Bagel 作为 backbone，其整体框架与 Bagel 原生能力高度匹配：

- **理解任务**：使用 Bagel 的 VLM 分支，输入 TS-image + 文本指令，输出文本 CoT；
- **生成任务**：使用 Bagel 的图像编辑/生成分支，输入 TS-image + 文本条件，输出生成图像；
- **交错训练**：TimeOmni-VL 的 `seq = P_sys ⊕ I_src ⊕ C_gen ⊕ R_gen ⊕ I_tgt` 正是 Bagel 的 `interleave` 数据格式。

---

## 4. 训练与微调可行性

### 4.1 官方训练入口

Bagel 官方提供完整的训练脚本：

```bash
# 预训练
bash scripts/train.sh
```

训练脚本核心文件：`train/pretrain_unified_navit.py`。

### 4.2 微调关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_path` | - | 加载 BAGEL-7B-MoT 路径 |
| `layer_module` | `Qwen2MoTDecoderLayer` | MoT 解码层 |
| `max_latent_size` | 64 | 微调时必须设为 64，否则权重加载越界 |
| `lr` | 2e-5 | 学习率 |
| `visual_gen` | True | 启用图像生成分支 |
| `visual_und` | True | 启用图像理解分支 |
| `freeze_llm` | False | 是否冻结 LLM |
| `freeze_vit` | False | 是否冻结 ViT |
| `freeze_vae` | True | 默认冻结 VAE |

### 4.3 自定义数据集接入

官方数据集配置在 `data/dataset_info.py`，支持：

- `t2i`：文本到图像（parquet 格式）；
- `editing`：图像编辑（parquet 格式）；
- `vlm`：视觉语言理解（JSONL 格式）。

TimeOmni-VL 场景下，需要自定义数据集格式：

- **理解任务**：vlm 格式，包含 TS-image 路径、问题、CoT、答案；
- **生成任务**：editing 格式，包含源 TS-image、目标 TS-image、生成指令、生成 CoT。

---

## 5. 推理方式

### 5.1 官方推理脚本

官方提供 `inferencer.py` 和 `inference.ipynb`，核心类为 `InterleaveInferencer`。

### 5.2 推理流程

```python
from inferencer import InterleaveInferencer

# 初始化
inferencer = InterleaveInferencer(model, vae_model, tokenizer, ...)

# 理解任务
output = inferencer(
    image=ts_image,
    text="Describe the price trend in this TS-image.",
    think=True,
    understanding_output=True,
)

# 生成任务（图像编辑）
output = inferencer(
    image=src_ts_image,
    text="Predict the next day electricity price.",
    think=True,
    understanding_output=False,
)
```

### 5.3 关键超参数

| 参数 | 典型值 | 说明 |
|---|---|---|
| `cfg_text_scale` | 4.0-8.0 | 文本条件强度 |
| `cfg_image_scale` | 1.0-2.0 | 输入图像保留程度 |
| `num_timesteps` | 50 | 扩散去噪步数 |
| `timestep_shift` | 3.0 | 步数分布偏移 |

---

## 6. 硬件资源需求

### 6.1 推理资源

| 场景 | 显存需求 | 说明 |
|---|---|---|
| FP16/BF16 推理 | 22-32 GB | 单张 A100/V100 32GB |
| NF4 量化推理 | 12-16 GB | 官方推荐 `--mode 2` |
| INT8 量化推理 | 16-22 GB | 不推荐 |
| CPU 推理 | 不推荐 | 极慢，仅用于验证代码结构 |

### 6.2 训练资源

| 场景 | 显存/硬件 | 说明 |
|---|---|---|
| 全参数微调 | 8×A100 40GB+ | 官方脚本使用 8 GPU |
| LoRA 微调 | 1×A100 40GB | 需自行修改脚本 |
| QLoRA 微调 | 1×A100 24GB / 1×4090 24GB | 适合个人/小团队 |

> **本项目建议**：由于本地无 GPU，代码需支持 CPU 结构验证；云平台训练时优先选择 1×A100 40GB 或 8×A100 环境。

---

## 7. 适配 TimeOmni-VL 的风险与建议

### 7.1 主要风险

| 风险 | 说明 | 建议 |
|---|---|---|
| 模型体积大 | 29.6 GB 权重 + 依赖多 | 云端下载，本地不保存 |
| flash-attn 安装困难 | 需 CUDA 环境编译 | 云平台镜像通常已预装 |
| 数据格式需自定义 | 官方为 T2I/Editing/VLM | 需重写 dataset_info.py |
| 训练成本高昂 | 全参数微调需 8×A100 | 先用 LoRA/QLoRA 验证 |
| 量化后精度下降 | NF4/INT8 可能影响生成质量 | 先用 FP16 跑通，再量化 |

### 7.2 适配建议

1. **优先复现 Bi-TSI 和 TSUMM-Suite**：在替换 Bagel 前先验证 TS-image 表示有效；
2. **使用官方 inference.ipynb 跑通单样本**：确认模型能加载、能理解、能生成；
3. **分阶段微调**：
   - 阶段 A：冻结 LLM/ViT/VAE，仅训练生成头和理解任务输出层；
   - 阶段 B：解冻 LLM，联合训练理解与生成；
4. **云端训练脚本化**：所有下载、训练、评估命令写成 shell 脚本，方便复用；
5. **本地 CPU 验证**：写 mock 版本的模型前向，验证数据流正确。

---

## 8. 备选方案

若 Bagel-7B 在部署中遇到困难，可考虑以下备选 UMM：

| 备选模型 | 优势 | 劣势 |
|---|---|---|
| **Janus-Pro-7B** | 理解+生成统一，轻量 | 生成质量弱于 Bagel |
| **Qwen2.5-VL-7B + Diffusion** | 生态成熟，文档丰富 | 非原生统一模型，需额外桥接 |
| **Emu3.5 / Chameleon** | 原生统一架构 | 社区支持相对较少 |
| **JanusFlow** | AR + Flow 混合 | 训练复杂度更高 |

**推荐**：Bagel-7B 是 TimeOmni-VL 的原生 backbone，应优先尝试；若资源不足，再考虑 Janus-Pro-7B 作为降级方案。

---

## 9. 结论与建议

1. **Bagel-7B 可完全获取**：权重、代码、训练脚本均已开源，Apache 2.0 协议可商用可修改。
2. **与 TimeOmni-VL 高度匹配**：原生支持理解+生成交错，论文方法可直接在其上实现。
3. **硬件要求较高**：本地无法训练，需云平台 GPU；本地可只做数据与 Bi-TSI 验证。
4. **建议实施路径**：
   - 第一步：本地完成 Bi-TSI + 数据流验证；
   - 第二步：云端下载 Bagel-7B 并跑通官方 inference；
   - 第三步：构造电价 TSUMM-Suite 数据集；
   - 第四步：在云端进行 LoRA/全参数微调；
   - 第五步：评估、消融、部署。

---

## 10. 参考链接

- 论文：https://arxiv.org/abs/2505.14683
- 模型：https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
- 代码：https://github.com/ByteDance-Seed/Bagel
- 主页：https://bagel-ai.org/
- Demo：https://demo.bagel-ai.org/
