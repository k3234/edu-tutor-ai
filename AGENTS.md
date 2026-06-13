# AGENTS.md — 面向教育领域的 7B 小模型训练协议

> 目标：用 **Qwen2.5-7B / DeepSeek-R1-1.5B / Trae SOLO 开放大模型** 作为教师，
> 训练一个教育垂直领域的自研小模型（3B-7B），在教育问答上
> **接近甚至超越 Google Gemma 4 12B**。
>
> 核心方法论：**数据为王 + 多教师蒸馏 + 垂直领域对齐**

---

## 一、整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                       多教师训练流水线                          │
│                                                              │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐      │
│  │ Qwen2.5-7B │     │DeepSeek-R1 │     │Trae SOLO   │      │
│  │  (主教师)   │────▶│  (推理专家) │────▶│  (评委)    │      │
│  └────────────┘     └────────────┘     └────────────┘      │
│         │              │              │                       │
│         ▼              ▼              ▼                       │
│  ┌───────────────────────────────────────────┐               │
│  │  scripts/teacher_inference.py             │               │
│  │  — 生成 SFT 问答对 (prompt/answer)        │               │
│  │  — 生成 DPO 偏好对 (chosen/rejected)      │               │
│  │  — JSONL 格式，兼容 LMM                   │               │
│  └───────────────────────────────────────────┘               │
│         │                                                     │
│         ▼                                                     │
│  ┌───────────────────────────────────────────┐               │
│  │  data/corpus_collector.py                 │               │
│  │  — 收集清洗教育语料                        │               │
│  │  — 去重 / 合并 / 导出纯文本               │               │
│  └───────────────────────────────────────────┘               │
│         │                                                     │
│         ▼                                                     │
│  ┌───────────────────────────────────────────┐               │
│  │  models/model.py (LMM GPT Transformer)    │               │
│  ├───────────────────────────────────────────┤               │
│  │  ① 预训练 scripts/train.py                │               │
│  │  ② SFT scripts/train_sft.py               │               │
│  │     支持 LoRA (models/lora.py)            │               │
│  │  ③ DPO scripts/train_dpo.py               │               │
│  │  ④ 蒸馏 scripts/distill.py                │               │
│  └───────────────────────────────────────────┘               │
│         │                                                     │
│         ▼                                                     │
│  ┌───────────────────────────────────────────┐               │
│  │  scripts/eval_edu_benchmark.py            │               │
│  │  — PPL (val)                             │               │
│  │  — 问答对比 (你的模型 vs 教师模型)        │               │
│  │  — 教师打分 (1~5 分)                      │               │
│  └───────────────────────────────────────────┘               │
│                                                              │
│  产出：一个可以在教育问答上匹敌 Gemma 4 12B 的小模型       │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、为什么这个方案能让"小模型超越大模型"

| 维度 | 从零训练 12B 通用大模型 | 用多教师 + 垂直蒸馏 训练 3B 模型 |
|------|------------------------|--------------------------------|
| 训练算力 | 数万美元 GPU / 数周 | 单卡/几小时即可跑通，1-3 天收敛 |
| 数据需求 | 万亿 tokens | 10~50 万条高质量问答对 |
| 领域能力 | 泛而不精，对教育题容易出错 | 精于教育问答，在垂直领域更强 |
| 对齐难度 | 需要 RLHF + 大量人工标注 | DPO 偏好对由教师模型自动生成 |
| 推理成本 | 需要 ~24GB 显存 + 较慢 | 3B 模型 4~8GB 显存即可跑 |
| 可控性 | 模型行为难预测 | 教师模型生成的偏好直接约束学生 |

**核心洞见**：小模型不可能在"所有领域"超越 12B 大模型，
但可以在"你的目标领域——教育"上显著优于 12B 大模型——因为：
1. 它专门看了教育语料；
2. 它的学习信号来自多位"教育专家教师"（比通用大模型更强的问答能力）；
3. 它的行为被 DPO 偏好对直接约束，输出更像老师。

---

## 三、教师模型分工

| 教师模型 | 部署方式 | 角色 | 适用数据 |
|----------|---------|------|---------|
| **Qwen2.5-7B-Instruct** | Ollama 本地 | 主教师：生成问答对、打分、日常对话 | 通用/语文/英语/历史/政治 |
| **DeepSeek-R1-1.5B** | Ollama 本地 | 推理专家：数学题、物理题、逻辑题 | 数学/物理/化学/信息 |
| **Trae SOLO 开放大模型** | HTTP API (OpenAI 兼容) | 最高评委：对难题生成参考答案，或做最终评分 | 高难度题/评估对比 |

> 为什么 DeepSeek-R1 不做"主教师"？
> 因为它的中文自然语言表达（散文、诗、故事类）不如 Qwen2.5，
> 但它的推理能力（多步逻辑、数学）非常强，适合当"专科老师"。

---

## 四、快速开始（30 分钟跑通全流程）

### 第 0 步：环境

```bash
pip install torch transformers requests pyyaml tqdm

# 安装 Ollama（Linux/macOS/Windows）
# https://ollama.com

# 拉教师模型
ollama pull qwen2.5:7b-instruct       # ~5GB
ollama pull deepseek-r1:1.5b          # ~1GB
ollama serve
```

### 第 1 步：数据

```bash
# 1) 准备语料：把课本、题库、教案、维基百科-中文教育条目等
#    放到 data/raw_src/ 下（支持 txt/md/jsonl）
mkdir -p data/raw_src
# ... 拷贝你的教育文本文件进去

# 2) 清洗合并
python data/corpus_collector.py --input-dir data/raw_src \
    --output data/raw/edu_corpus.txt

# 3) 用教师模型生成 SFT 问答对（第一次至少 500 条测试）
python scripts/teacher_inference.py \
    --output data/sft/raw.jsonl --num 500 \
    --models qwen2.5:7b-instruct deepseek-r1:1.5b

# 4) 格式整理 + 切分验证集
python scripts/prepare_sft.py \
    --input data/sft/raw.jsonl \
    --output data/sft/train.jsonl

# 5) （可选）生成 DPO 偏好对
python scripts/teacher_inference.py \
    --mode dpo --output data/dpo/raw.jsonl --num 200
```

### 第 2 步：预训练（让模型有"中文教育"的语言理解基础）

```bash
# 先跑 12 层 768d 的小模型（~120M 参数）验证流程
python scripts/train.py --config configs/lmm_small.yaml --steps 5000

# 之后上 3B（需要更多 GPU 显存）
# python scripts/train.py --config configs/lmm_3b.yaml --steps 50000
```

### 第 3 步：SFT（指令微调，让模型学会"提问-回答"格式）

```bash
# 全参数微调（需要 GPU）
python scripts/train_sft.py \
    --data data/sft/train.jsonl \
    --checkpoint experiments/exp_small_v1/checkpoints/best.pt \
    --exp-dir experiments/exp_sft_01 \
    --steps 2000 --batch-size 4

# 或者用 LoRA（显存不足时优先选）
python scripts/train_sft.py --use-lora --lora-rank 8 \
    --data data/sft/train.jsonl \
    --checkpoint experiments/exp_small_v1/checkpoints/best.pt \
    --exp-dir experiments/exp_sft_lora_01 \
    --steps 3000
```

### 第 4 步：DPO（偏好对齐，让模型输出"更像老师"的回答）

```bash
python scripts/train_dpo.py \
    --data data/dpo/raw.jsonl \
    --checkpoint experiments/exp_sft_01/checkpoints/best.pt \
    --exp-dir experiments/exp_dpo_01 \
    --beta 0.1 --steps 1000
```

### 第 5 步：评估 + 迭代

```bash
python scripts/eval_edu_benchmark.py \
    --checkpoint experiments/exp_dpo_01/checkpoints/best.pt \
    --val-data data/sft/val.jsonl \
    --compare-teacher qwen2.5:7b-instruct

# 查看 eval_report.json — 这个报告告诉你模型的优势/不足
# 如果发现某学科差（比如数学），就针对该学科扩充 SFT/DPO 数据再训练
```

### 第 6 步（可选）：蒸馏到更小模型

```bash
# 把 7B 教师蒸馏到 3B / 1.5B / 更小
python scripts/distill.py \
    --teacher-ckpt experiments/exp_dpo_01/checkpoints/best.pt \
    --teacher-n-layer 12 --teacher-n-embd 768 \
    --student-n-layer 6 --student-n-embd 384 \
    --data data/sft/train.jsonl --steps 2000
```

---

## 五、模型规模策略

| 阶段 | 模型规模 | 建议硬件 | 训练时长 | 用途 |
|------|---------|---------|---------|------|
| POC 验证 | 120M (Small) | CPU 或 4GB GPU | 几小时 | 验证整个训练 pipeline 能跑通 |
| v0.1 | 360M (1B) | 8GB GPU | 半天 | 第一个"能用"的模型 |
| v1.0 | 3B | 24GB GPU / 双卡 | 1~3 天 | 主力推理模型，质量接近 12B 教师 |
| v2.0 | 7B | 80GB GPU / 多张卡 | 几天 | 旗舰质量，推理较慢 |

**推荐路线：先让 120M 跑通 → 直接跳到 3B 训练**

---

## 六、文件总览

```
LMM/
├── README.md                     # 总览 + 快速开始 + 训练路线图
├── AGENTS.md                     # 本文件（多智能体训练协议）
├── requirements.txt
│
├── models/
│   ├── model.py                  # LMM GPT Transformer
│   ├── config.py                 # 模型配置
│   ├── attention.py              # 自注意力
│   ├── block.py                  # Transformer Block
│   └── lora.py                   # LoRA 低秩适配
│
├── data/
│   ├── corpus_collector.py       # 教育语料采集清洗
│   ├── dataset.py                # 数据集基类
│   └── dataloader.py             # DataLoader
│
├── trainer/
│   ├── checkpoint.py             # 保存/加载
│   ├── logger.py                 # 日志
│   └── optimizer.py              # AdamW + 余弦退火
│
├── scripts/
│   ├── train.py                  # 预训练
│   ├── train_tokenizer.py        # 训练 BPE tokenizer
│   ├── prepare_tokens.py         # 预处理 tokens
│   ├── convert_to_hf.py          # 转 HuggingFace 格式
│   │
│   ├── teacher_inference.py      # ★ 多教师推理生成 SFT/DPO 数据
│   ├── prepare_sft.py            # ★ SFT 数据整理 + 切分
│   ├── train_sft.py              # ★ SFT 指令微调 (支持 LoRA)
│   ├── train_dpo.py              # ★ DPO 偏好对齐
│   ├── distill.py                # ★ 知识蒸馏 Teacher → Student
│   └── eval_edu_benchmark.py     # ★ 教育领域评估基准
│
└── configs/
    ├── lmm_small.yaml            # 120M 默认配置
    ├── lmm_3b.yaml               # 3B 配置
    └── lmm_7b.yaml               # 7B 配置
```

---

## 七、常见问题 & 解决

| 问题 | 原因 | 解决 |
|------|------|------|
| 训练很慢/显存 OOM | 模型太大或 batch 太大 | 先用 120M；调小 batch-size；用 LoRA；梯度累积 |
| 模型输出胡言乱语 | 预训练步数不够 + SFT 数据太少 | 增加预训练步数；至少 1000 条 SFT 数据 |
| 模型回答太短/答非所问 | SFT 数据质量不够 + 没有 DPO | 提高 answer 质量；跑 DPO 偏好对齐 |
| 教师模型调用失败 | Ollama 没启动 / 模型名写错 | `ollama serve` 检查；确认 `ollama list` 有目标模型 |
| 某学科（如数学）特别差 | 该学科训练数据不足 | 用 DeepSeek-R1 专门生成该学科问答对，针对性再 SFT 一轮 |
| 评估报告得分上不去 | 数据量不够 + 训练不足 | 目标：SFT 至少 1 万条，DPO 至少 1000 条；训练步数 1 万+ |

---

## 八、与 Gemma 4 12B 的差距如何缩小/超越

Gemma 4 12B 是通用模型，在"通用中文教育问答"上强，但：

1. **它没有专门优化过中国 K12 教育**（课标、题型、术语）
2. **它的中文并不完美**（对文言文、古诗、中文语法题不如本地模型）
3. **它的推理成本远高于 3B 小模型**

策略：
- **10000 条**教育问答对 + 教师模型生成 → SFT 后，你的模型在教育问答上就会比 Gemma 4 "更像老师"
- **1000 条**DPO 偏好对 → 让模型在"同一问题下选择更合适的回答"这一判断上，超过通用大模型
- 持续用 `eval_edu_benchmark.py` 评估，并**在得分低的学科上继续补数据**

**经验：只要你的训练数据 > 1 万条高质量教育问答，
配合 DPO，3B 模型在教育问答质量上可以接近甚至超过 12B 通用模型。**

---

## 九、关键技术点

### 9.1 LoRA (Low-Rank Adaptation)
- 原理：冻结大模型权重，只在每个 Transformer Block 的线性层旁边插入两个小矩阵（R × d 和 d × R，d << hidden_dim）
- 效果：训练参数减少 1000 倍以上，显存占用减半以上，训练速度翻倍
- 位置：`models/lora.py`，配合 `train_sft.py --use-lora` 使用

### 9.2 DPO (Direct Preference Optimization)
- 原理：对同一个问题，让教师模型给出一个"最好回答"和一个"较差回答"；
  训练时最大化 `P(chosen)/P(rejected)` 的对数比
- 核心优势：**不需要奖励模型，不需要 RLHF，端到端可训练**
- 位置：`scripts/train_dpo.py`

### 9.3 知识蒸馏 (Knowledge Distillation)
- 原理：`L = alpha * KL(P_teacher || P_student) + (1-alpha) * CE(P_student, y)`
- 温度 T：让教师输出的概率分布软化，学生能学"置信度"而不只是标签
- 位置：`scripts/distill.py`

### 9.4 教师协同
- Qwen2.5-7B 做日常问答和中文表达
- DeepSeek-R1 做数学/物理/推理题
- Trae SOLO 做最终打分和高难度参考答案
- 三者的"互补"能让单一模型同时学多种能力

---

## 十、安全与合规

- ⚠️ 所有训练数据必须合规，不要爬未授权的内容
- ⚠️ 教师模型本身有开源 license 限制（如 Qwen2.5 有商用许可协议），请查看
- ✅ 模型权重和代码建议 Apache 2.0 开源
- ✅ 建议在模型输出中加入"AI 生成，仅供参考"提示语

---

## 十一、下一步行动清单

```
[ ] 跑通 120M 小模型 + 500 条 SFT 数据的全流程
[ ] 收集 1GB+ 教育语料（课本、教案、题库、维基百科教育条目）
[ ] 扩充 SFT 数据到 1 万条（按学科均衡）
[ ] 跑 3B 模型预训练（上云，如 AutoDL 24GB 机器）
[ ] 跑 SFT + DPO（各 5000 步 + 1000 步）
[ ] 评估 → 找到最差学科 → 针对性补数据 → 再训练 → 再评估
[ ] 蒸馏到 1.5B / 500M 版本用于边缘推理
[ ] 集成到你的教育应用（AI 助教、批改、错题讲解等）
```

---

**最后更新**：2026-06-13
**架构负责人**：LMM Project Team
**适用版本**：LMM v1.0+
