# LMM — Large Model Management (面向教育领域的开源大模型训练框架)

> 目标：使用 **Qwen2.5-7B / DeepSeek-R1 / Trae SOLO 大模型** 作为 **教师**，
> 训练一个教育垂直领域的 **3B-7B** 自研模型，在教育问答上达到/超过 Google Gemma 4 12B 水平。
> 架构：标准 GPT-style Decoder-only Transformer（从 scratch 实现，易于理解和扩展）。

---

## 功能概览

| 模块 | 说明 | 脚本 |
|------|------|------|
| 📝 语料采集 | 中文教育领域语料合并与清洗 | `data/corpus_collector.py` |
| 🎓 教师推理 | 用 Ollama 部署的 Qwen2.5 / DeepSeek-R1 生成高质量问答/偏好对 | `scripts/teacher_inference.py` |
| ✨ SFT 数据 | 把问答对整理为 instruction-tuning 格式 | `scripts/prepare_sft.py` |
| 🔥 SFT 训练 | 指令微调，含 LoRA 低秩适配 | `scripts/train_sft.py` |
| ⚖️ DPO 对齐 | 用教师偏好对模型做对齐 | `scripts/train_dpo.py` |
| 💧 知识蒸馏 | 大模型 → 小模型蒸馏压缩 | `scripts/distill.py` |
| 📊 评估基准 | 教育问答对比 + PPL + 教师打分 | `scripts/eval_edu_benchmark.py` |

## 快速开始（推荐 30 分钟跑通"最小可运行"流水线）

### 第 0 步：环境

```bash
# 安装依赖
pip install torch transformers requests pyyaml tqdm

# 安装 Ollama（用于拉起教师模型）
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: 去 https://ollama.com 下载安装

# 拉取教师模型（可选，按你的机器配置挑一个）
ollama pull qwen2.5:7b-instruct       # 主教师（~5GB）
ollama pull deepseek-r1:1.5b           # 思考链专家（~1GB）
ollama pull llama3.1:8b-instruct-q4_K_M # 备用（~5GB）
ollama serve
```

### 第 1 步：收集语料（数据是关键）

```bash
# 把你的课本/题库/教案等 txt/md/jsonl 文件放到 data/raw_src/
# 然后执行合并：
python data/corpus_collector.py --input-dir data/raw_src --output data/raw/edu_corpus.txt
```

### 第 2 步：用教师模型生成 SFT 问答对

```bash
# 先跑 500 条做实验
python scripts/teacher_inference.py --output data/sft/raw.jsonl --num 500
python scripts/prepare_sft.py --input data/sft/raw.jsonl --output data/sft/train.jsonl
```

### 第 3 步：预训练 + SFT

```bash
# 预训练（120M 小模型先跑通，约几分钟到几小时）
python scripts/train.py --config configs/lmm_small.yaml --steps 5000

# SFT 指令微调
python scripts/train_sft.py --data data/sft/train.jsonl \
    --checkpoint experiments/exp_small_v1/checkpoints/best.pt \
    --exp-dir experiments/exp_sft_01 --steps 1000

# 如果想省显存/时间，可以用 LoRA：
python scripts/train_sft.py --data data/sft/train.jsonl --use-lora --lora-rank 8 \
    --checkpoint experiments/exp_small_v1/checkpoints/best.pt \
    --exp-dir experiments/exp_sft_lora_01
```

### 第 4 步：生成 DPO 偏好对 + 训练（可选，但效果显著）

```bash
# 1. 同一问题让 3 个教师出答案，由主教师选出最好/最差
python scripts/teacher_inference.py --mode dpo --output data/dpo/raw.jsonl --num 200

# 2. DPO 训练
python scripts/train_dpo.py --data data/dpo/raw.jsonl \
    --checkpoint experiments/exp_sft_01/checkpoints/best.pt \
    --exp-dir experiments/exp_dpo_01 --beta 0.1 --steps 500
```

### 第 5 步：评估你的模型

```bash
python scripts/eval_edu_benchmark.py \
    --checkpoint experiments/exp_dpo_01/checkpoints/best.pt \
    --val-data data/sft/val.jsonl \
    --compare-teacher qwen2.5:7b-instruct
```

看输出的 `eval_report.json` —— 里面有每个教育学科你的模型 vs 教师模型的对比，用于下一步迭代方向。

### 第 6 步（可选）：蒸馏到 3B 小模型

```bash
python scripts/distill.py --teacher-ckpt experiments/exp_dpo_01/checkpoints/best.pt \
    --teacher-n-layer 12 --teacher-n-embd 768 \
    --student-n-layer 4  --student-n-embd 256 \
    --data data/sft/train.jsonl --steps 1000
```

---

## 项目结构

```
LMM/
├── PRODUCT.md                 # 产品目标（必读，先理解目标再改代码）
├── DESIGN.md                  # 设计规范
├── AGENTS.md                  # 多 Agent 协作协议
├── README.md                  # 本文件
├── requirements.txt
│
├── models/
│   ├── model.py               # LMM 模型（GPT-style Transformer）
│   ├── config.py              # 模型配置
│   ├── attention.py           # 自注意力实现
│   ├── block.py               # Transformer Block
│   └── lora.py                # LoRA 低秩适配
│
├── data/
│   ├── corpus_collector.py    # 语料合并 & 清洗
│   ├── dataset.py             # 通用数据集（已存在）
│   └── dataloader.py          # DataLoader（已存在）
│
├── trainer/
│   ├── checkpoint.py          # checkpoint 保存/加载（已存在）
│   ├── logger.py              # 日志
│   └── optimizer.py           # 优化器
│
├── scripts/
│   ├── train.py               # 预训练主脚本（已存在）
│   ├── train_tokenizer.py     # tokenizer 训练（已存在）
│   ├── prepare_tokens.py      # token 预处理（已存在）
│   ├── evaluate.py            # 原评估脚本（已存在）
│   ├── convert_to_hf.py       # 转 HuggingFace 格式（已存在）
│   │
│   ├── teacher_inference.py   # 教师模型生成 SFT/DPO 数据
│   ├── prepare_sft.py         # SFT 数据整理
│   ├── train_sft.py           # SFT 训练（含 LoRA）
│   ├── train_dpo.py           # DPO 偏好优化
│   ├── distill.py             # 知识蒸馏
│   └── eval_edu_benchmark.py  # 教育领域评估基准
│
└── configs/
    ├── lmm_small.yaml         # 120M 配置（默认跑通用）
    ├── lmm_3b.yaml            # 3B 配置
    └── lmm_7b.yaml            # 7B 配置
```

## 训练路线图（从 0 → 3B → 7B）

```
第一阶段：验证链路（1 天）
  120M 模型，教育语料 100MB → 预训练 1 小时
  Qwen2.5-7B 生成 500 条问答对
  SFT 训练 30 分钟 → 验证模型可以"像老师一样回答"

第二阶段：数据为王（1-2 周）
  收集 1GB+ 教育领域语料
  生成 5万-10万条问答对（人工抽查质量）
  跑通 DPO（1000 条偏好对）
  建立评估基准 → 看 PPL 和教师打分曲线

第三阶段：规模上去（1-3 周）
  1B 模型 → 跑几天
  3B 模型 → 需要云 GPU / 多张卡
  持续做蒸馏，让 3B 学生逼近教师水平

第四阶段：对齐 Gemma 4 12B（持续）
  让你的模型在中文教育问答上，比 Gemma 4 12B 更懂学生
  （通用领域 Gemma 更强，不求超过；但垂直领域你有优势）
```

## 为什么用教师模型蒸馏而不是从零训练通用模型

| 维度 | 从零训练 7B 通用模型 | 用大模型做教师 + 垂直蒸馏 |
|------|------------------------|----------------------------|
| 训练数据量 | 万亿 tokens | 千万级 tokens 即可 |
| 训练算力 | 数千张 A100 × 数月 | 单张卡跑几天 |
| 目标能力 | 什么都懂一点，但不精 | 垂直领域比小模型强很多 |
| 对齐难度 | 需要大量人工偏好数据 | 偏好数据由教师模型生成 |
| 评估 | 复杂，需要多种基准 | 简单，教师模型自己打分 |

一句话：**你的模型靠"数据质量 + 领域化"赢，不靠参数量赢**。

## 关键设计决策

| 决策 | 说明 |
|------|------|
| 从 scratch 实现 Transformer | 教学友好，可完全掌控，也方便后续蒸馏/剪枝 |
| 教师模型全部本地可跑（Ollama） | 不依赖外部 API，数据隐私可控，成本低 |
| 数据格式统一使用 JSONL | 易于扩展、易于查看、易于清洗 |
| LoRA 默认支持 | 让小显存机器也能跑通训练流程 |
| DPO 而非 RLHF | 简化训练流程，不需要 reward model，且效果相近 |

## 下一步建议（按重要性排序）

1. **先让 120M 小模型跑通全流程**（预训练 → SFT → DPO → 评估）
2. **提升数据质量**：花一天写更好的 prompt，让教师生成更好的问答
3. **升级词表**：用真正的 BPE tokenizer（不是简单 char）替换 `SimpleTokenizer`
4. **上云训练 3B 模型**：租用 GPU（如 AutoDL / 阿里云 PAI / 腾讯云黑石）
5. **做评测对比**：每次迭代都跑 eval_edu_benchmark.py 看进度

## 安全与合规

- ⚠️ 所有训练数据确保合规（不要爬未授权内容）
- ⚠️ 教师模型本身有 License 限制，请注意
- ✅ 模型权重和代码建议使用 Apache 2.0 开源协议
- ✅ 输出内容最好加"AI 生成"提示语，不误导学生

## 参考

- LLaMA 论文 & 代码架构参考
- LoRA: https://arxiv.org/abs/2106.09685
- DPO: https://arxiv.org/abs/2305.18290
- NanoGPT (Karpathy): https://github.com/karpathy/nanoGPT
- Rasbtka LLMs-from-scratch: https://github.com/rasbt/LLMs-from-scratch

---

**最后更新**：2026-06-13
