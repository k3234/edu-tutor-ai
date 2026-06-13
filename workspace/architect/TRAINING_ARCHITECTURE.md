# TRAINING_ARCHITECTURE.md — 从零训练大模型架构设计

> Architect 产出 | TASK-LLM-001 | 2026-06-09

---

## 一、关键概念澄清

### AirLLM 的定位

| 维度 | 说明 |
|------|------|
| **本质** | 推理优化工具（非训练框架） |
| **核心技术** | 分层加载（Layer-by-Layer）：模型存硬盘，按需加载当前层到显存 |
| **适用场景** | 低显存推理、模型验证、部署服务 |
| **不适用场景** | 预训练（需反向传播，要求全部参数在显存） |

### 正确路线

```
数据准备 → Tokenizer训练 → PyTorch手写Transformer → 预训练 → AirLLM推理验证 → 部署
    ↑________________________训练阶段________________________↑  ↑___AirLLM阶段___↑
```

---

## 二、模型架构设计

### 2.1 模型规格：LMM-Small（教育版）

| 参数 | 值 | 说明 |
|------|-----|------|
| 架构类型 | GPT-style Decoder-only | 自回归语言模型 |
| 总参数量 | ~120M | 可在8GB VRAM上训练 |
| 层数 (n_layer) | 12 | Transformer Block 数量 |
| 注意力头数 (n_head) | 12 | 每层的多头注意力 |
| 嵌入维度 (n_embd) | 768 | Token 嵌入和隐藏层维度 |
| FFN维度 | 3072 | 前馈网络中间层 (4× n_embd) |
| 上下文长度 | 1024 | 最大序列长度 |
| 词表大小 | 32000 | 中文教育领域词表 |
| Dropout | 0.1 | 训练时正则化 |

### 2.2 模型类层级

```
LMMConfig          # 配置类（dataclass），存储所有超参数
  └── LMMBlock     # 单个 Transformer Block
       ├── CausalSelfAttention   # 因果自注意力（带Flash Attention）
       └── MLP                   # 前馈网络（SwiGLU激活）
  └── LMMModel     # 完整模型
       ├── TokenEmbedding        # Token嵌入层
       ├── PositionalEmbedding   # 可学习位置编码
       ├── [LMMBlock × 12]       # 12层Transformer
       ├── LayerNorm             # 最终层归一化
       └── LMHead               # 语言建模头（线性层 → 词表大小）
```

### 2.3 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 激活函数 | SwiGLU | 训练更稳定，效果优于ReLU/GELU |
| 归一化位置 | Pre-LayerNorm | 训练更稳定，主流选择 |
| 位置编码 | 可学习位置嵌入 | 简化实现，小模型效果可接受 |
| 注意力实现 | Flash Attention (可选) | 减少显存，加速训练 |
| 权重初始化 | 正态分布 N(0, 0.02) | GPT标准初始化方式 |

### 2.4 Python 代码结构（供 Coder 实现）

```python
# models/config.py
@dataclass
class LMMConfig:
    vocab_size: int = 32000
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    ffn_dim: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    # ...

# models/attention.py
class CausalSelfAttention(nn.Module): ...

# models/mlp.py
class SwiGLUMLP(nn.Module): ...

# models/block.py
class LMMBlock(nn.Module): ...

# models/model.py
class LMMModel(nn.Module):
    """完整的 LMM 语言模型"""
    def forward(self, idx, targets=None): ...
    def generate(self, idx, max_new_tokens, temperature=1.0): ...
```

---

## 三、数据管道设计

### 3.1 数据来源（教育垂直领域）

| 来源 | 预估规模 | 优先级 |
|------|----------|--------|
| 教材文本（数理化语英） | ~50MB | P0 |
| 百科问答（百度百科/维基） | ~200MB | P0 |
| 教学对话（师生问答） | ~30MB | P1 |
| 考试题库 | ~20MB | P1 |

**总目标**：收集 ~300MB 清洗后中文教育语料（约1-2亿 token）

### 3.2 数据处理流程

```
原始文本 → 分句 → 去重去噪 → 拼接成文档 → BPE分词 → 序列化存储
```

### 3.3 Tokenizer 设计

- **算法**：BPE（Byte-Pair Encoding）
- **工具**：HuggingFace `tokenizers` 库
- **词表大小**：32000
- **特殊Token**：`<pad>`, `<unk>`, `<bos>`, `<eos>`

### 3.4 DataLoader 设计

```python
# data/dataset.py
class LMMDataset(Dataset):
    """将文本切分为固定长度的训练样本"""
    def __init__(self, tokens, seq_len=1024): ...

# data/dataloader.py
def create_dataloader(tokens, batch_size, seq_len, shuffle=True):
    """创建训练/验证 DataLoader"""
```

---

## 四、训练流程设计

### 4.1 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 32 | 每批样本数 |
| 梯度累积步数 | 4 | 等效 batch_size = 128 |
| 学习率 | 3e-4 | 初始学习率 |
| 学习率调度 | Cosine + Warmup | 前1000步线性预热 |
| 优化器 | AdamW | weight_decay=0.1 |
| 训练步数 | 50000 | 总计训练步数 |
| 验证间隔 | 500步 | 每500步验证一次 |
| Checkpoint 间隔 | 2000步 | 保存模型和优化器状态 |
| 混合精度 | bf16 (自动) | 减少显存，加速训练 |

### 4.2 训练脚本结构

```python
# scripts/train.py
def main():
    # 1. 加载配置
    config = load_config("configs/lmm_small.yaml")
    
    # 2. 初始化模型
    model = LMMModel(config)
    
    # 3. 加载数据
    train_loader = create_dataloader(...)
    val_loader = create_dataloader(...)
    
    # 4. 初始化优化器+调度器
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=1000)
    
    # 5. 训练循环
    for step in range(config.max_steps):
        loss = train_step(model, batch, optimizer, scheduler)
        
        if step % 500 == 0:
            val_loss = validate(model, val_loader)
            log_metrics(step, loss, val_loss)
        
        if step % 2000 == 0:
            save_checkpoint(model, optimizer, step, loss)

# configs/lmm_small.yaml
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  
training:
  batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 3.0e-4
  max_steps: 50000
  warmup_steps: 1000
  
data:
  seq_len: 1024
  data_path: "data/edu_corpus/"
```

### 4.3 训练监控

| 指标 | 频率 | 说明 |
|------|------|------|
| Train Loss | 每步 | 训练损失 |
| Val Loss | 每500步 | 验证集损失 |
| Perplexity | 每500步 | 困惑度 = exp(val_loss) |
| GPU 显存 | 每步 | 监控显存使用 |
| 学习率 | 每步 | 跟踪调度器变化 |

---

## 五、AirLLM 集成方案

### 5.1 AirLLM 在项目中的角色

```
训练 Checkpoint → [转换] → HuggingFace 格式权重 → [加载] → AirLLM 推理
                                                        ↓
                                              低显存验证 / API 部署
```

### 5.2 集成点1：训练后验证

```python
# scripts/validate_with_airllm.py
"""
目的：训练完成后，用 AirLLM 在低显存环境下验证模型生成质量
流程：
1. 加载训练好的 checkpoint
2. 转换为 HuggingFace 兼容格式
3. 用 AirLLM AutoModel 加载
4. 运行评估 prompt，检查生成质量
"""
from airllm import AutoModel

# 加载训练好的模型（AirLLM 分层加载，仅需4GB显存）
model = AutoModel.from_pretrained("./checkpoints/lmm_small_step50000_hf")
prompts = [
    "什么是勾股定理？",
    "请解释牛顿第一定律。",
    "光合作用的化学方程式是",
]
for prompt in prompts:
    output = model.generate(prompt, max_new_tokens=100)
    print(f"[Prompt] {prompt}")
    print(f"[Output] {output}\n")
```

### 5.3 集成点2：低显存推理服务

```python
# scripts/serve_with_airllm.py
"""
目的：用 AirLLM 部署 API 推理服务
优势：4GB 显存即可运行 120M 模型（未来扩展到更大模型时优势明显）
"""
```

### 5.4 Checkpoint → HuggingFace 转换

```python
# scripts/convert_to_hf.py
"""
将训练checkpoint转为HuggingFace格式：
- model_state_dict → HF safetensors
- config → HF config.json
- tokenizer → HF tokenizer files
"""
```

---

## 六、评估体系设计

### 6.1 评估维度

| 维度 | 方法 | 指标 |
|------|------|------|
| 语言建模能力 | 验证集评估 | Perplexity |
| 知识掌握 | 教育领域问答 | 人工评分（1-5分） |
| 生成质量 | 续写任务 | 流畅度、相关性 |
| 推理效率 | AirLLM 推理 | 生成速度（token/s）、显存占用 |

### 6.2 基线对比

| 模型 | 参数量 | 训练数据 | 预期 PPL |
|------|--------|----------|----------|
| LMM-Small (ours) | 120M | 教育语料 2亿token | < 30 |
| GPT-2 Small (参考) | 124M | WebText | 参考值 |
| 随机初始化 | 120M | 无 | > 100 |

---

## 七、项目目录结构

```
LMM/                                   # 项目根目录
├── models/                            # 模型相关（Coder 实现）
│   ├── __init__.py
│   ├── config.py                      # LMMConfig 配置类
│   ├── attention.py                   # 因果自注意力
│   ├── mlp.py                         # SwiGLU 前馈网络
│   ├── block.py                       # Transformer Block
│   └── model.py                       # LMMModel 完整模型
│
├── data/                              # 数据相关（Coder 实现）
│   ├── __init__.py
│   ├── dataset.py                     # LMMDataset
│   ├── dataloader.py                  # create_dataloader
│   ├── download_scripts/              # 数据采集脚本
│   │   └── crawl_edu_corpus.py        # 教育语料爬取
│   └── raw/                           # 原始数据目录
│
├── scripts/                           # 脚本（Coder 实现）
│   ├── train.py                       # 训练主脚本
│   ├── train_tokenizer.py             # 分词器训练
│   ├── convert_to_hf.py               # Checkpoint → HF 格式
│   ├── validate_with_airllm.py        # AirLLM 推理验证
│   ├── serve_with_airllm.py           # AirLLM API 部署
│   └── evaluate.py                    # 评估脚本
│
├── configs/                           # 配置文件
│   ├── lmm_small.yaml                 # 模型训练配置
│   └── data_config.yaml               # 数据配置
│
├── experiments/                       # 实验记录
│   └── exp20260609_lmm_small_v1/      # 实验目录
│       ├── config.yaml                # 实验配置
│       ├── train.log                  # 训练日志
│       ├── checkpoints/               # 模型检查点
│       └── README.md                  # 实验总结
│
├── docs/                              # 文档
│   ├── TRAINING_GUIDE.md              # 训练指南
│   └── ARCHITECTURE.md                # 本架构文档副本
│
├── trainer/                           # 训练器（Coder 实现）
│   ├── __init__.py
│   ├── optimizer.py                   # AdamW + 调度器
│   ├── checkpoint.py                  # 保存/加载 checkpoint
│   └── logger.py                      # 训练日志和监控
│
└── configs/requirements.txt           # Python 依赖
    # torch>=2.0.0
    # transformers
    # tokenizers
    # airllm
    # pyyaml
    # tqdm
    # wandb (可选)
```

---

## 八、实施顺序（给 Coder 的指令）

### Phase 1：基础设施（优先级 P0）
1. `models/config.py` — LMMConfig
2. `models/attention.py` — CausalSelfAttention
3. `models/mlp.py` — SwiGLUMLP
4. `models/block.py` — LMMBlock
5. `models/model.py` — LMMModel（含 forward + generate）

### Phase 2：数据处理（优先级 P0）
6. `data/download_scripts/crawl_edu_corpus.py` — 教育语料爬取
7. `scripts/train_tokenizer.py` — BPE 分词器训练
8. `data/dataset.py` — LMMDataset
9. `data/dataloader.py` — 数据加载器

### Phase 3：训练系统（优先级 P0）
10. `trainer/optimizer.py` — 优化器+调度器
11. `trainer/checkpoint.py` — Checkpoint 管理
12. `trainer/logger.py` — 训练日志
13. `scripts/train.py` — 训练主脚本
14. `configs/lmm_small.yaml` — 训练配置

### Phase 4：AirLLM 集成（优先级 P1）
15. `scripts/convert_to_hf.py` — 模型转换
16. `scripts/validate_with_airllm.py` — AirLLM 验证推理
17. `scripts/serve_with_airllm.py` — AirLLM 部署服务

### Phase 5：评估与文档（优先级 P1）
18. `scripts/evaluate.py` — 评估脚本
19. `docs/TRAINING_GUIDE.md` — 训练指南

---

## 九、设计原则回顾

- ✅ **简洁优先** — 每文件 < 300行，函数 < 50行
- ✅ **工程化** — PEP 8，类型注解，docstring
- ✅ **可复现** — YAML 配置，seed 固定，实验目录规范
- ✅ **可教学** — 详细注释，便于 Learner 讲解
- ✅ **分层设计** — 训练/AirLLM推理分离，互不耦合
- ✅ **消费级友好** — 8GB VRAM 可训练，4GB 可推理

---

**Architect 签名**：架构方案已完成，移交 Coder 开始编码实现。

**下一步**：Coder 收到 `shared_chat.json` 中的 `task_design` 消息后，按 Phase 1→5 顺序实现各模块。