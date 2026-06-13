# LMM 训练指南

> 从零开始训练你的第一个大模型

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch transformers tokenizers pyyaml tqdm
pip install airllm  # 用于推理阶段
```

### 2. 准备数据

```bash
# 生成示例语料（用于快速测试）
python data/download_scripts/crawl_edu_corpus.py --sample --max_size_mb 50

# 训练分词器
python scripts/train_tokenizer.py --data_dir data/raw --output_dir data/tokenizer

# 对语料进行分词并保存
# （此步骤需要你自己实现或手动将文本转为 token 序列保存为 data/tokens.pt）
```

### 3. 开始训练

```bash
# 使用默认配置训练
python scripts/train.py --config configs/lmm_small.yaml

# 恢复训练
python scripts/train.py --config configs/lmm_small.yaml --resume experiments/exp_xxx/checkpoints/latest.pt
```

### 4. 转换为 HuggingFace 格式

```bash
python scripts/convert_to_hf.py \
    --checkpoint experiments/exp20260609_lmm_small_v1/checkpoints/best.pt \
    --output_dir models/lmm_small_hf \
    --tokenizer_dir data/tokenizer
```

### 5. AirLLM 推理验证

```bash
# 使用 AirLLM 低显存推理
python scripts/validate_with_airllm.py --model_dir models/lmm_small_hf

# 或使用标准 Transformers（需要更多显存）
python scripts/validate_with_airllm.py --model_dir models/lmm_small_hf --use_transformers
```

### 6. 部署服务

```bash
python scripts/serve_with_airllm.py --model_dir models/lmm_small_hf --port 8000
```

---

## 模型架构

### LMM-Small (~120M 参数)

| 参数 | 值 |
|------|-----|
| 架构 | GPT-style Decoder-only |
| 层数 | 12 |
| 注意力头数 | 12 |
| 嵌入维度 | 768 |
| FFN 维度 | 3072 |
| 上下文长度 | 1024 |
| 词表大小 | 32000 |

### 关键设计

- **SwiGLU 激活**: 现代大模型标准，训练更稳定
- **Pre-LayerNorm**: 训练更稳定的主流选择
- **因果掩码**: 确保自回归特性，不"偷看"未来
- **权重共享**: 词嵌入和输出层共享，减少参数量

---

## 训练配置

### 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 32 | 每批样本数 |
| 梯度累积 | 4 | 等效 batch_size = 128 |
| 学习率 | 3e-4 | 初始学习率 |
| Warmup | 1000 步 | 线性预热 |
| 训练步数 | 50000 | 总计 |
| 验证间隔 | 500 步 | |
| Checkpoint | 2000 步 | |

### 硬件要求

| 阶段 | 最低配置 | 推荐配置 |
|------|----------|----------|
| 训练 | 8GB VRAM | RTX 3060 12GB |
| 推理 (AirLLM) | 4GB VRAM | 任意 GPU |
| 推理 (标准) | 6GB VRAM | RTX 3060 |

---

## 训练监控

### 日志输出

```
Step   100 | Loss: 8.2341 | PPL: 3775.23 | LR: 3.00e-04
Step   200 | Loss: 6.1234 | PPL:  456.78 | LR: 3.00e-04
...
Validation | Step   500 | Val Loss: 5.4321 | Val PPL: 228.45
Checkpoint saved: experiments/.../step_2000.pt
```

### 预期指标

| 阶段 | Train Loss | Val PPL | 说明 |
|------|-----------|---------|------|
| 初始 | ~10.0 | ~22000 | 随机初始化 |
| 1000 步 | ~5.0 | ~150 | 快速下降 |
| 10000 步 | ~3.5 | ~33 | 稳定学习 |
| 50000 步 | ~2.5 | ~12 | 收敛 |

---

## 常见问题

### Q: 显存不足 (OOM)

**A**: 尝试以下方法:
1. 减小 batch_size（如 16 → 8）
2. 增大 gradient_accumulation_steps 保持等效 batch
3. 使用更小的模型配置（LMM-Tiny: 30M 参数）
4. 启用混合精度训练（默认已启用）

### Q: 训练 loss 不下降

**A**: 检查:
1. 学习率是否过大/过小
2. 数据是否正确加载（检查 token 文件）
3. 梯度是否正常（检查 grad_norm）
4. 模型初始化是否正确

### Q: AirLLM 无法加载模型

**A**: 
1. 确保已转换为 HuggingFace 格式
2. 检查 config.json 中的 model_type 是否为 "gpt2"
3. 尝试使用 `--use_transformers` 参数用标准方式加载

---

## 文件结构

```
LMM/
├── models/              # 模型实现
│   ├── config.py
│   ├── attention.py
│   ├── mlp.py
│   ├── block.py
│   └── model.py
├── data/                # 数据处理
│   ├── dataset.py
│   ├── dataloader.py
│   └── download_scripts/
├── trainer/             # 训练工具
│   ├── optimizer.py
│   ├── checkpoint.py
│   └── logger.py
├── scripts/             # 脚本
│   ├── train.py
│   ├── train_tokenizer.py
│   ├── convert_to_hf.py
│   ├── validate_with_airllm.py
│   ├── serve_with_airllm.py
│   └── evaluate.py
├── configs/             # 配置
│   └── lmm_small.yaml
└── docs/                # 文档
    └── TRAINING_GUIDE.md
```

---

## 进阶调优

### 扩大模型规模

编辑 `configs/lmm_small.yaml`:

```yaml
model:
  n_layer: 24      # 从 12 增加到 24
  n_head: 16       # 从 12 增加到 16
  n_embd: 1024     # 从 768 增加到 1024
```

注意: 参数量会相应增加，需要更多显存。

### 使用更多数据

1. 收集真实教育语料（教材、百科、试题）
2. 放入 `data/raw/` 目录
3. 重新训练分词器
4. 重新训练模型

### 微调 (Fine-tuning)

在预训练完成后，可以用特定任务数据微调:

```python
# 加载预训练模型
model = LMMModel(config)
checkpoint = torch.load("checkpoints/best.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# 在特定任务数据上继续训练
# ...
```

---

**祝训练顺利！** 🚀