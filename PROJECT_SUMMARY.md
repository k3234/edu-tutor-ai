LMM - 多教师模型训练的教育领域小模型框架
==================================================

一个用 PyTorch 从零实现的 GPT-style 语言模型训练框架，
支持通过 Qwen2.5 / DeepSeek-R1 / Trae SOLO 大模型作为教师，
通过 SFT + DPO + 蒸馏三阶段训练，使 7B 以内小模型
在教育领域达到甚至超越 Google Gemma 4 12B 的水平。


快速启动（5 分钟跑通最小版本）
--------------------------------------------------
```bash
cd e:\学习LLM\开发者学习\09-LLM从零学习\LMM

# 1. 用示例数据先跑一次 SFT，验证完整 pipeline 能跑通
#    （无需教师模型，无需 GPU，CPU 即可）
python scripts/train_sft.py --data data/sample_sft.jsonl ^
    --steps 200 --batch-size 2 --use-lora --exp-dir experiments/quick_test

# 2. 跑教育评估基准，看看模型学到了什么
python scripts/eval_edu_benchmark.py ^
    --checkpoint experiments/quick_test/checkpoints/best.pt

# 3. 使用教师模型生成更多 SFT 数据（需要 Ollama 已启动）
python scripts/teacher_inference.py --mode sft --num 100 ^
    --models qwen2.5:7b-instruct deepseek-r1:1.5b ^
    --output data/sft/raw.jsonl

# 4. 用端到端流水线跑完整训练
python scripts/pipeline.py --run-all --sft-num 500 --dpo-num 150 ^
    --n-layer 12 --n-embd 768 --sft-steps 2000 --dpo-steps 800 ^
    --use-lora --exp-dir experiments/edu_v1
```


目录结构
--------------------------------------------------
```
LMM/
├── AGENTS.md                 # 多教师协作训练协议（架构设计）
├── models/                   # 模型实现
│   ├── model.py             # GPT-style Transformer
│   └── lora.py              # LoRA 低秩适配
├── configs/                  # 模型配置（不同规模）
│   ├── lmm_small.yaml       # 120M 参数（教学验证）
│   ├── lmm_3b.yaml          # ~3B 参数
│   └── lmm_7b.yaml          # ~7B 参数
├── data/                     # 数据相关
│   ├── sample_sft.jsonl     # 10 条示例 SFT 数据
│   ├── corpus_collector.py  # 语料收集清洗
│   └── (sft/ | dpo/ | raw/)  # 运行后自动生成
├── scripts/                  # 训练脚本
│   ├── train.py             # 预训练（Causal LM）
│   ├── train_sft.py         # SFT 指令微调（+LoRA 支持）
│   ├── train_dpo.py         # DPO 偏好对齐
│   ├── distill.py           # 知识蒸馏（大→小）
│   ├── eval_edu_benchmark.py # 教育领域评估
│   ├── prepare_sft.py       # SFT 数据格式转换
│   ├── teacher_inference.py # 多教师数据生成
│   └── pipeline.py          # 端到端训练流水线
└── experiments/              # 训练输出（运行后自动生成）
    └── exp_*/checkpoints/   # 各阶段 best.pt
```


三阶段训练路线（核心思路）
--------------------------------------------------

【阶段 1】SFT 指令微调
   用 Qwen2.5-7B 和 DeepSeek-R1-1.5B 生成高质量教育问答对，
   让学生模型学会"按教学风格回答问题"。
   特色：仅在 assistant 部分计算 loss，不浪费算力在 system/user 上。

【阶段 2】DPO 偏好对齐
   让学生模型学会"哪个回答更好"，优化输出质量和教学风格。
   数学公式：L_DPO = -log σ(β * (log p(chosen) - log p(rejected)))
   优势：不需要单独训练 reward model，直接用 preference pairs。

【阶段 3】知识蒸馏（可选）
   把阶段 2 的模型作为"教师"，蒸馏到更小模型（如 120M/300M），
   让小模型学到大模型的"软概率分布"，实现"7B 的水平，120M 的体积"。


教师模型分工
--------------------------------------------------
1. Qwen2.5:7B-instruct    — 主生成教师（通用 + 中文强）
   用途：生成 SFT 问答对、生成 chosen 回答

2. DeepSeek-R1:1.5B       — 推理专家教师（数学/逻辑强）
   用途：生成数学题解答、推理过程

3. Trae SOLO 开放大模型   — 验证/打分教师（多模型对比）
   用途：对回答打分、生成 rejected 回答


模型规模配置表
--------------------------------------------------
| 名称      | 层数 | 嵌入维度 | 头数 | 估算参数 | 适用场景           |
|-----------|------|----------|------|----------|--------------------|
| Small     | 12   | 768      | 12   | ~120M   | 教学验证 / 本地CPU |
| Medium    | 24   | 1024     | 16   | ~350M   | 中端 GPU (8GB+)    |
| Large (3B)| 32   | 2048     | 32   | ~3B     | 高性能 GPU 云训练  |
| XLarge (7B) | 48   | 3072    | 48   | ~7B     | 云端多卡训练       |


硬件建议
--------------------------------------------------
【本地快速验证】CPU + 8GB 内存即可跑 Small 模型
【本地小规模训练】RTX 3090/4090 (24GB) 可训练 Medium (~350M)
【云端训练】推荐：
  - 入门：单张 A100 (80GB) — 可训练 3B 模型
  - 进阶：4×A100 80GB — 可训练 7B 模型
  - 高端：8×A100/H100 — 可训练更大模型

云服务厂商：阿里云、腾讯云、火山引擎、百度智能云等。


下一步可以做什么？
--------------------------------------------------
1. 把课本/教案/题库 PDF 转成 txt/md，放到 data/raw_src/
2. 运行 python data/corpus_collector.py 生成训练语料
3. 用 ollama pull qwen2.5:7b-instruct deepseek-r1:1.5b
4. 运行 scripts/pipeline.py --run-all 开始训练
5. 看 experiments/exp_edu_v1/eval_report.json 决定下一轮方向
