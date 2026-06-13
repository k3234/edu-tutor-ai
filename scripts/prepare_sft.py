# -*- coding: utf-8 -*-
"""
prepare_sft.py
==============
把 teacher_inference.py 生成的 SFT 问答对，整理成训练用格式，
并按 95/5 切分训练集 / 验证集。

输出的每条 JSONL 格式：
    {
      "system": "你是一位精通 {subject} 的老师，会详细且有条理地解答学生问题。",
      "user":   "题目正文...",
      "assistant": "解答正文..."
    }

用法：
    python scripts/prepare_sft.py --input data/sft/raw.jsonl \
        --output data/sft/train.jsonl --val-output data/sft/val.jsonl
"""

import argparse
import json
import os
import random
import sys
from typing import List, Dict

# 简单的学科 -> 系统提示词映射
SUBJECT_SYSTEM_PROMPTS = {
    "数学": "你是一位严谨的数学老师，擅长用清晰的步骤讲解题目，并会强调常见易错点。",
    "语文": "你是一位资深的语文老师，既能讲解文学作品，也能辅导阅读理解和写作。",
    "英语": "You are an experienced English teacher. 请用中英双语讲解英语相关问题。",
    "物理": "你是一位物理老师，擅长用公式结合实例讲解物理概念，并强调单位与量纲。",
    "化学": "你是一位化学老师，会清晰写出化学反应式，并解释反应机理。",
    "生物": "你是一位生物老师，擅长用生活化例子帮助学生理解复杂的生物概念。",
    "历史": "你是一位历史老师，会按时间线和因果关系讲解历史事件。",
    "地理": "你是一位地理老师，擅长把自然地理与人文地理结合讲解。",
    "政治": "你是一位政治老师，会用规范、准确的表述讲解政治学与社会学概念。",
    "信息": "你是一位编程与信息科技老师，会给出可运行的代码示例与清晰解释。",
}


def default_system_prompt(subject: str) -> str:
    for key, prompt in SUBJECT_SYSTEM_PROMPTS.items():
        if key in subject:
            return prompt
    return f"你是一位精通{subject}的老师，会详细且有条理地解答学生问题。"


def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}", file=sys.stderr)
        return []
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def normalize(item: Dict) -> Dict:
    """把教师生成的问答对 -> 标准的 (system, user, assistant) 格式。"""
    prompt = item.get("prompt") or item.get("question") or item.get("user") or ""
    answer = item.get("answer") or item.get("assistant") or ""
    subject = item.get("subject") or ""

    # 基础质量过滤
    if not isinstance(prompt, str) or not isinstance(answer, str):
        return {}
    if len(prompt.strip()) < 5 or len(answer.strip()) < 50:
        return {}

    return {
        "system": default_system_prompt(subject),
        "user": prompt.strip(),
        "assistant": answer.strip(),
        "subject": subject,
    }


def main():
    p = argparse.ArgumentParser(description="SFT 数据整理 & 切分")
    p.add_argument("--input", required=True,
                   help="teacher_inference.py 生成的 JSONL")
    p.add_argument("--output", default="data/sft/train.jsonl",
                   help="训练集输出路径")
    p.add_argument("--val-output", default="data/sft/val.jsonl",
                   help="验证集输出路径（可留空表示不分）")
    p.add_argument("--val-ratio", type=float, default=0.05,
                   help="验证集比例")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（确保可重复）")
    args = p.parse_args()

    raw = load_jsonl(args.input)
    print(f"[INFO] 读取 {len(raw)} 条原始样本")

    normalized = [it for it in (normalize(x) for x in raw) if it]
    print(f"[INFO] 清洗后保留 {len(normalized)} 条")

    # 简单去重（按 user 前 120 字）
    seen = set()
    deduped = []
    for it in normalized:
        k = it["user"][:120]
        if k in seen:
            continue
        seen.add(k)
        deduped.append(it)
    print(f"[INFO] 去重后保留 {len(deduped)} 条")

    # 切分
    random.seed(args.seed)
    random.shuffle(deduped)
    n_val = int(len(deduped) * args.val_ratio)
    val = deduped[:n_val]
    train = deduped[n_val:]

    # 写出
    for path, items in [(args.output, train),
                        (args.val_output, val)]:
        if not path:
            continue
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"[OUT] 写出 {path}: {len(items)} 条")


if __name__ == "__main__":
    main()
