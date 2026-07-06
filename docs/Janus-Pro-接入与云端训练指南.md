# Janus-Pro Backbone 接入与云端训练指南

> **文档定位**：说明如何将 `timeomni_vl` 从 Mock backbone 切换到 Janus-Pro 真实模型，以及云端 GPU 训练步骤。  
> **前提**：本地环境无 GPU，因此模型加载与训练必须在云端 GPU 实例上执行。  > **生成时间**：2026-07-03

---

## 1. 当前状态

- `MockAdapter`：已跑通数据流、训练、推理、评估全流程。
- `JanusAdapter`：已实现接口，包含：
  - `load()`
  - `understand()`
  - `generate()`（自回归 VQ token 生成）
  - `train_step()`（理解损失 + 占位生成损失）
  - `setup_optimizer()`（LoRA 支持）
  - `save_checkpoint()` / `load_checkpoint()`
- 本地 CPU 已验证 `JanusAdapter` 可以成功导入。
- 真实模型加载需要：
  - CUDA GPU
  - `transformers`, `janus` 包
  - 从 Hugging Face 下载 Janus-Pro 权重

---

## 2. 环境准备

### 2.1 安装依赖

```bash
# 在云端 GPU 服务器上执行
pip install -r requirements.txt

# Janus 需要从源码安装
git clone https://github.com/deepseek-ai/Janus.git
pip install -e ./Janus
```

### 2.2 下载模型权重

```bash
# 推荐先用 1B 做 POC
huggingface-cli download deepseek-ai/Janus-Pro-1B --local-dir ./models/Janus-Pro-1B

# 或 7B（需要更多显存）
huggingface-cli download deepseek-ai/Janus-Pro-7B --local-dir ./models/Janus-Pro-7B
```

### 2.3 验证 GPU

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

---

## 3. 切换到 Janus Backbone

使用专用配置：

```bash
python -m timeomni_vl.cli.train --config configs/train_janus.yaml
```

配置文件关键项：

```yaml
model:
  backbone: "janus"
  model_path: "deepseek-ai/Janus-Pro-1B"  # 或本地路径 ./models/Janus-Pro-1B
  device: "auto"
  dtype: "bfloat16"

bitsi:
  image_size: 384  # Janus-Pro 固定 384x384

training:
  lora_enabled: true
  lora_rank: 8
  lora_alpha: 16
  lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

---

## 4. 训练流程

### 4.1 目前代码中的调用链

```python
# timeomni_vl/cli/train.py
adapter = build_adapter(model_cfg.backbone)  # JanusAdapter
adapter.load(model_cfg.model_path, model_cfg.device)

# TSUMMDatasetBuilder 将样本转为 image + text
# TSUMMTrainer 调用 adapter.train_step(batch)
```

### 4.2 JanusAdapter.train_step 内部

当前实现：

```python
def train_step(self, batch):
    loss_und = self._understanding_loss(batch)   # CrossEntropy on CoT/answer
    loss_gen = self._generation_loss(batch)      # placeholder 0.0
    loss_total = loss_und + loss_gen
    loss_total.backward()
    self.optimizer.step()
    return {"loss_und": ..., "loss_gen": ..., "loss_total": ...}
```

**阶段一：先微调理解任务**

理解损失已经可用。运行 `train_janus.yaml` 会先验证理解 CoT 训练是否可行。

**阶段二：再添加生成损失**

需要实现：
- 用 `image_to_vq_tokens()` 将目标 TS-image 编码为离散 token
- 在 `train_step` 中计算图像 token 的 CrossEntropy 损失
- 参考 `timeomni_vl/models/janus_utils.py`

---

## 5. 关键限制

### 5.1 分辨率限制

- TimeOmni-VL 原生设计：896×896 TS-image
- Janus-Pro 理解/生成分支：固定 384×384

影响：
- TS-image 必须 resize 到 384×384
- 每个日周期 96 点，在 384 宽度下每个周期约 4 像素
- 高频电价细节可能丢失

建议：
- 在 `TS2IConverter` 中降低 image_size 到 384
- 或减少 context_days，避免过度压缩
- 后续可考虑更高分辨率理解模型（如 Qwen2.5-VL + 扩散生成）

### 5.2 生成范式差异

- Bagel：扩散生成 → MSE 损失
- Janus-Pro：自回归 VQ token → CrossEntropy 损失

因此 `JanusAdapter.generate()` 和 `train_step()` 已按自回归实现，与扩散不同。

### 5.3 显存需求

| 模型 | 推理 | LoRA 微调 |
|---|---|---|
| Janus-Pro-1B | 8-12 GB | 16-24 GB |
| Janus-Pro-7B | 16-24 GB | 24-40 GB |

云端推荐：
- POC：1×RTX 4090 24GB（1B LoRA）
- 生产：1×A100 40GB 或 8×A100（7B LoRA/全参数）

---

## 6. 云端训练脚本示例

```bash
# 云端 GPU 服务器
python -m pip install -r requirements.txt
git clone https://github.com/deepseek-ai/Janus.git
python -m pip install -e ./Janus

huggingface-cli download deepseek-ai/Janus-Pro-1B --local-dir ./models/Janus-Pro-1B

export PYTHONPATH=$(pwd)
python -m timeomni_vl.cli.train --config configs/train_janus.yaml
```

---

## 7. 验证理解推理

```bash
python -m timeomni_vl.cli.infer \
  --config configs/train_janus.yaml \
  --image outputs/test_train_ts_image.png \
  --task understand \
  --prompt "How many variables are encoded in this TS-image?" \
  --output outputs/pred.npy
```

---

## 8. 下一步工作

1. 在云端 GPU 上安装 Janus 并下载权重
2. 运行 `train_janus.yaml` 验证理解任务微调
3. 实现真实的生成损失（`janus_utils.py` 已提供工具函数）
4. 验证生成推理效果
5. 评估 I2TS 解码精度
6. 根据效果决定是否升级到 Janus-Pro-7B 或迁移到 Bagel

---

## 9. 参考链接

- Janus-Pro 论文：https://arxiv.org/abs/2501.17811
- 模型 1B：https://huggingface.co/deepseek-ai/Janus-Pro-1B
- 模型 7B：https://huggingface.co/deepseek-ai/Janus-Pro-7B
- 代码：https://github.com/deepseek-ai/Janus
