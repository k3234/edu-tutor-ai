# -*- coding: utf-8 -*-
"""
eval_edu_benchmark.py — 教育领域模型评估基准

评估维度：
    1. 困惑度（PPL） — 在 SFT 验证集上算 PPL = exp(-avg log prob)
    2. 问答样例对比 — 对 10 道典型题，比较你的模型 vs 教师模型（如 qwen2.5）的输出
    3. "教师打分" — 让教师模型对两个答案评分 1~5，看你的模型是否接近期望质量
    4. 输出质量统计 — 答案平均长度、字符/分词统计

使用：
    python scripts/eval_edu_benchmark.py \
        --checkpoint experiments/exp_dpo_01/checkpoints/best.pt \
        --val-data data/sft/val.jsonl \
        --compare-teacher qwen2.5:7b-instruct
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel


# ============ 工具函数 ============

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def simple_char_tokenizer(text: str, max_seq_len: int, vocab_size: int = 32000) -> List[int]:
    encoded = text.encode("utf-8")
    tokens = [b for b in encoded]
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    return tokens


def simple_char_decode(token_ids: List[int]) -> str:
    """把 token ids 还原成文本"""
    try:
        return bytes([max(0, min(255, t)) for t in token_ids]).decode("utf-8", errors="replace")
    except Exception:
        return ""


# ============ 教师模型推理（Ollama） ============

def ollama_generate(prompt: str, model: str = "qwen2.5:7b-instruct",
                    host: str = "http://localhost:11434",
                    temperature: float = 0.6, max_tokens: int = 1000,
                    timeout: int = 300) -> Optional[str]:
    """调用本地 Ollama 模型生成文本（用于对比参考）"""
    url = f"{host}/api/chat"
    try:
        r = requests.post(url, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[WARN] Ollama 调用失败: {e}", file=sys.stderr)
    return None


# ============ 文本生成（用自研 LMM 模型） ============

@torch.no_grad()
def generate_text(model, prompt: str, max_new_tokens: int = 200,
                  temperature: float = 0.7, top_k: int = 40,
                  device: str = "cpu") -> str:
    """用自研模型从 prompt 生成文本"""
    model.eval()
    prompt_tokens = simple_char_tokenizer(prompt, model.config.max_seq_len - 1,
                                          model.config.vocab_size)
    if not prompt_tokens:
        return ""

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    max_input_len = model.config.max_seq_len - max_new_tokens - 1
    if input_ids.size(1) > max_input_len:
        input_ids = input_ids[:, -max_input_len:]

    # 自回归生成
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_logits = logits[:, -1, :] / temperature

        # top-k sampling
        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            topk_vals, _ = torch.topk(next_logits, k)
            min_val = topk_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < min_val] = -float("inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 如果输出太长，停
        if input_ids.size(1) >= model.config.max_seq_len - 1:
            break

    new_tokens = input_ids[0, len(prompt_tokens):].tolist()
    return simple_char_decode(new_tokens).strip()


# ============ PPL 计算 ============

@torch.no_grad()
def compute_ppl(model, data_items: List[Dict], device: str) -> float:
    """在验证集上计算困惑度 PPL = exp( - (1/N) ∑ log P(t_i | t_{<i}) )"""
    model.eval()
    total_log_prob = 0.0
    total_tokens = 0

    for item in data_items:
        prompt = item.get("user", "") or item.get("prompt", "")
        answer = item.get("assistant", "") or item.get("answer", "")
        text = f"{prompt}\n{answer}\n"
        tokens = simple_char_tokenizer(text, model.config.max_seq_len,
                                        model.config.vocab_size)
        if len(tokens) < 5:
            continue
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        logits = logits[:, :-1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        targets = input_ids[:, 1:]
        V = log_probs.size(-1)
        targets = torch.clamp(targets, 0, V - 1)
        gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        total_log_prob += gathered.sum().item()
        total_tokens += gathered.numel()

    if total_tokens == 0:
        return float("inf")
    avg_log_prob = total_log_prob / total_tokens
    ppl = float(torch.exp(torch.tensor(-avg_log_prob)).item())
    return ppl


# ============ 教师打分 ============

def teacher_score_answer(teacher_model: str, prompt: str, answer: str,
                         host: str = "http://localhost:11434") -> Optional[int]:
    """让教师模型对 answer 的质量打 1-5 分"""
    rating_prompt = (
        f"作为教育专家，请对下面这个学生问答的『回答』进行评分（1=极差，5=优秀）。\n"
        f"只输出一个数字，不要任何解释或附加文字。\n\n"
        f"【题目】{prompt}\n"
        f"【回答】{answer[:1500]}\n"
        f"评分（只输出 1-5 之间的一个整数）："
    )
    try:
        out = ollama_generate(rating_prompt, model=teacher_model, host=host,
                              temperature=0.0, max_tokens=50)
        if out:
            for ch in out:
                if ch.isdigit() and 1 <= int(ch) <= 5:
                    return int(ch)
    except Exception:
        pass
    return None


# ============ 主评估流程 ============

SUBJECT_PROMPTS = [
    ("小学数学", "请用简便方法计算： 125 × 48 = ? 并说明思路。"),
    ("初中数学", "证明：在任意三角形中，两边之和大于第三边。"),
    ("初中物理", "一辆汽车以 20 m/s 的速度匀速行驶，然后以 -2 m/s² 的加速度减速。求 5 秒后的速度？"),
    ("初中化学", "什么是氧化反应？请举例一个常见的氧化反应并写出化学方程式。"),
    ("初中语文", "请解释『不以物喜，不以己悲』的含义，并简述其表达的思想感情。"),
    ("初中英语", "Please fill in the blank: He ___ (go) to school by bus every day. 并回答为什么这样填？"),
    ("初中历史", "简述秦朝统一中国的历史意义。"),
    ("初中地理", "简述季风气候对我国农业的影响。"),
    ("初中生物", "光合作用和呼吸作用的主要区别是什么？"),
    ("信息科技", "请解释什么是递归函数，并用 Python 写一个求阶乘的递归函数。"),
]


def evaluate(args):
    device = get_device()
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 教师模型: {args.compare_teacher}")

    # 1. 加载模型
    config = LMMConfig()
    if args.n_layer:
        config.n_layer = args.n_layer
    if args.n_embd:
        config.n_embd = args.n_embd
        config.n_head = max(1, args.n_embd // 64)  # 保持 head_dim=64
        config.ffn_dim = 4 * args.n_embd

    model = LMMModel(config).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        filtered = {k: v for k, v in sd.items()
                    if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        print(f"[INFO] 加载 checkpoint: {args.checkpoint}")
    else:
        print("[WARN] 没有 checkpoint，用随机初始化模型（仅供测试脚本）")

    # 2. 加载验证集
    val_data = []
    if args.val_data and os.path.exists(args.val_data):
        with open(args.val_data, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    val_data.append(json.loads(line))
                except Exception:
                    pass
    print(f"[INFO] 验证集样本: {len(val_data)}")

    # 3. PPL
    print("\n[STEP 1/3] 计算 PPL (语言建模困惑度)...")
    if val_data:
        ppl = compute_ppl(model, val_data, device)
        print(f"  PPL (val): {ppl:.2f}")
    else:
        ppl = None
        print("  (无验证集，跳过)")

    # 4. 问答对比
    print("\n[STEP 2/3] 问答对比（学科基准）...")
    sample_comparisons: List[Dict] = []

    random.seed(42)
    samples = SUBJECT_PROMPTS
    if len(val_data) > 0:
        # 混一些验证集里的题目
        extra = [(item.get("subject", "通用"), item.get("user", "") or item.get("prompt", ""))
                 for item in val_data[:5]]
        samples = list(samples) + extra

    our_answers = []
    teacher_answers = []

    for i, (subject, prompt) in enumerate(samples):
        # 我们模型的回答
        our = generate_text(model, f"系统: 你是{subject}老师。\n用户: {prompt}\n老师: ",
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature, device=device)
        # 教师模型的回答
        teacher = ollama_generate(
            f"你是一位优秀的{subject}老师。请详细回答：{prompt}",
            model=args.compare_teacher, temperature=0.7, max_tokens=args.max_new_tokens
        )

        our_answers.append(our)
        teacher_answers.append(teacher)

        print(f"  [{i+1}/{len(samples)}] {subject}")
        print(f"    你的模型: {our[:100]}{'...' if len(our) > 100 else ''}")
        print(f"    教师模型: {teacher[:100] if teacher else '(无)'}"
              f"{'...' if teacher and len(teacher) > 100 else ''}")

        sample_comparisons.append({
            "subject": subject,
            "prompt": prompt,
            "our_answer": our,
            "teacher_answer": teacher,
            "our_len": len(our),
            "teacher_len": len(teacher) if teacher else 0,
        })

    # 5. 教师打分
    print("\n[STEP 3/3] 教师评分（1~5 分）...")
    our_scores = []
    teacher_scores = []
    for item in sample_comparisons:
        s1 = teacher_score_answer(args.compare_teacher, item["prompt"], item["our_answer"]) \
            if item["our_answer"] else None
        s2 = teacher_score_answer(args.compare_teacher, item["prompt"], item["teacher_answer"]) \
            if item["teacher_answer"] else None
        our_scores.append(s1)
        teacher_scores.append(s2)
        print(f"  {item['subject']:12s} 你的模型: {s1 if s1 else '-'}  教师模型: {s2 if s2 else '-'}")

    # 6. 汇总统计
    our_valid_scores = [s for s in our_scores if s]
    teacher_valid_scores = [s for s in teacher_scores if s]

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "teacher_model": args.compare_teacher,
        "ppl": ppl,
        "score": {
            "our_avg": (sum(our_valid_scores) / len(our_valid_scores)) if our_valid_scores else None,
            "teacher_avg": (sum(teacher_valid_scores) / len(teacher_valid_scores)) if teacher_valid_scores else None,
            "our_scores": our_scores,
            "teacher_scores": teacher_scores,
        },
        "answer_length_stats": {
            "our_avg_len": sum(len(a) for a in our_answers if a) / max(1, sum(1 for a in our_answers if a)),
            "teacher_avg_len": sum(len(a) for a in teacher_answers if a) / max(1, sum(1 for a in teacher_answers if a)),
        },
        "samples": sample_comparisons,
    }

    # 打印汇总
    print("\n" + "=" * 60)
    print("【评估汇总】")
    print(f"  PPL: {ppl:.2f}" if ppl else "  PPL: (无)")
    if our_valid_scores:
        print(f"  你的模型平均得分: {report['score']['our_avg']:.2f} / 5")
    if teacher_valid_scores:
        print(f"  教师模型平均得分: {report['score']['teacher_avg']:.2f} / 5")
    print(f"  你的模型平均回答长度: {report['answer_length_stats']['our_avg_len']:.0f} 字符")
    print(f"  教师模型平均回答长度: {report['answer_length_stats']['teacher_avg_len']:.0f} 字符")
    print("=" * 60)

    # 7. 保存
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] 报告已保存到 {args.output}")


def main():
    p = argparse.ArgumentParser(description="教育领域模型评估基准")
    p.add_argument("--checkpoint", default=None, help="要评估的模型 checkpoint")
    p.add_argument("--val-data", default=None, help="SFT 验证集 JSONL")
    p.add_argument("--compare-teacher", default="qwen2.5:7b-instruct",
                   help="用来对比打分的 Ollama 教师模型")
    p.add_argument("--output", default="eval_report.json",
                   help="评估报告输出路径 (JSON)")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="模型生成的最大 token 数")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="生成温度")
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
