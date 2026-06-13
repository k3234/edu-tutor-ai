# -*- coding: utf-8 -*-
"""
teacher_inference.py
====================
用本地 Ollama 部署的大模型（Qwen2.5-7B、DeepSeek-R1-1.5B 等）和可选的
Trae SOLO HTTP API 作为"教师"，生成高质量的 SFT 问答对 和 DPO 偏好对。

主要功能：
  1. ollama_chat          - 调用本地 Ollama 大模型
  2. solo_chat            - 调用 Trae SOLO 的 OpenAI 兼容 API（可选）
  3. generate_qa          - 生成 SFT 问答对（{prompt, answer, subject}）
  4. generate_dpo         - 生成 DPO 偏好对（{prompt, chosen, rejected, ...}）
  5. CLI                  - 支持命令行参数，方便脚本化

典型使用：
    # 生成 500 条 SFT 问答对
    python scripts/teacher_inference.py --output data/sft/raw.jsonl --num 500

    # 生成 200 条 DPO 偏好对
    python scripts/teacher_inference.py --mode dpo --output data/dpo/raw.jsonl --num 200

    # 同时用 qwen2.5:7b-instruct 和 deepseek-r1:1.5b 交叉生成
    python scripts/teacher_inference.py --output data/sft/raw.jsonl --num 1000 \
        --models qwen2.5:7b-instruct deepseek-r1:1.5b
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict, Optional

import requests

# ========== 学科与题型池（用来让教师模型生成多样化题目） ==========
SUBJECTS = [
    "小学数学", "初中数学", "高中数学",
    "小学语文", "初中语文", "高中语文",
    "小学英语", "初中英语", "高中英语",
    "初中物理", "高中物理",
    "初中化学", "高中化学",
    "初中生物", "高中生物",
    "初中历史", "高中历史",
    "初中地理", "初中政治", "高中政治",
    "信息科技", "编程入门",
]

# 题目类型提示词，让教师有多种出题方式
QUESTION_TEMPLATES = [
    "请以 {subject} 老师的身份，给学生出一道中等难度的题目，并给出详细解答。"
    "要求：① 题目要有实际情境；② 解答要先给解题思路再给步骤；③ 用中文作答。",
    "请从 {subject} 中挑一个学生经常做错的知识点，设计一道例题并详细讲解。"
    "要求：指出易出错点和正确思路。",
    "请用 {subject} 中一个核心概念，设计一道概念辨析题，并给出完整解答。",
    "请设计一道 {subject} 的应用题，给出详细解答，突出思考过程。",
    "请从 {subject} 中选一个公式/定理，给出它的一个使用例题并详细讲解。",
]

# DPO 模式下用的提示词（让主教师在多个答案中选出最好/最差）
DPO_JUDGE_PROMPT = """你是一位资深的 {subject} 教育专家。下面是同一个问题的 {n_answers} 个不同解答，请：
1. 选出**最好的**一个回答（最清晰、逻辑最严谨、最适合学生理解）
2. 选出**最差的**一个回答（最模糊、逻辑错误、或不适合学生）

请用以下 JSON 格式回答，不要写其他文字：
{{
  "chosen_index": 0,
  "rejected_index": 1,
  "reason": "简述为什么最好的那个更好"
}}

问题：
{question}

各个解答：
{answers_text}
"""


# ========== 教师模型调用 ==========

def ollama_chat(model: str, prompt: str, host: str = "http://localhost:11434",
                temperature: float = 0.8, max_tokens: int = 1500,
                timeout: int = 300) -> Optional[str]:
    """调用本地 Ollama 模型。"""
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"[ollama_chat] 调用 {model} 失败: {e}", file=sys.stderr)
        return None


def solo_chat(prompt: str, api_url: Optional[str] = None,
              api_key: Optional[str] = None, model: str = "default",
              temperature: float = 0.7, max_tokens: int = 2000,
              timeout: int = 300) -> Optional[str]:
    """
    调用 Trae SOLO 的 OpenAI 兼容 API。
    只有当本地模型不够强时才用。配置通过环境变量读取：
        SOLO_API_URL  - API 地址
        SOLO_API_KEY  - API Key（留空则不鉴权）
    """
    api_url = api_url or os.environ.get("SOLO_API_URL")
    api_key = api_key or os.environ.get("SOLO_API_KEY", "")
    if not api_url:
        return None

    url = api_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[solo_chat] 调用失败: {e}", file=sys.stderr)
        return None


# ========== SFT 问答对生成 ==========

def _parse_qa(text: str) -> Optional[Dict]:
    """
    把教师模型输出的文本尽量解析成 {"question": ..., "answer": ...}。
    支持多种格式：
      - JSON
      - "题目：... 解答：..."
      - "问题：... 答案：..."
      - 启发式：前半段题目，后半段答案
    """
    if not text:
        return None

    # 尝试 1: JSON 格式
    json_text = text.strip()
    if json_text.startswith("```"):
        json_text = json_text.strip("`").strip()
        if json_text.lower().startswith("json"):
            json_text = json_text[4:].strip()
    try:
        if json_text.startswith("{"):
            obj = json.loads(json_text)
            q = obj.get("question") or obj.get("题目") or obj.get("prompt")
            a = obj.get("answer") or obj.get("解答") or obj.get("答案")
            if q and a:
                return {"question": str(q).strip(), "answer": str(a).strip()}
    except Exception:
        pass

    # 尝试 2: "题目：... 解答：..." 分割
    import re
    for q_key in ["题目", "问题", "Question"]:
        for a_key in ["解答", "答案", "解析", "Answer", "解答过程"]:
            pat = rf"{q_key}[：:](.+?)\n[^:]*{a_key}[：:](.+)"
            m = re.search(pat, text, re.DOTALL)
            if m:
                return {"question": m.group(1).strip(), "answer": m.group(2).strip()}

    # 尝试 3: "### 题目 ... ### 解答 ..."
    for q_h in ["## 题目", "### 题目", "**题目**", "**问题**"]:
        for a_h in ["## 解答", "### 解答", "**解答**", "**答案**"]:
            if q_h in text and a_h in text:
                q = text.split(q_h, 1)[1].split(a_h, 1)[0].strip(" -\n")
                a = text.split(a_h, 1)[1].strip()
                if len(q) > 10 and len(a) > 50:
                    return {"question": q, "answer": a}

    # 尝试 4: 启发式——第一个空行/短段落作为题目，其余为答案
    lines = text.strip().splitlines()
    if len(lines) >= 3:
        # 第一段可能是题目
        head, rest = "", []
        found_blank = False
        for line in lines:
            if not found_blank and not line.strip():
                found_blank = True
                continue
            if not found_blank:
                head += line + "\n"
            else:
                rest.append(line)
        if head.strip() and rest:
            q = head.strip()
            a = "\n".join(rest).strip()
            if 10 <= len(q) <= 500 and len(a) >= 50:
                return {"question": q, "answer": a}

    return None


def generate_qa(models: List[str], subject: str,
                host: str = "http://localhost:11434",
                temperature: float = 0.8) -> Optional[Dict]:
    """生成一条 SFT 问答对。失败返回 None。"""
    tmpl = random.choice(QUESTION_TEMPLATES)
    prompt = tmpl.format(subject=subject)

    # 随机挑一个模型
    model = random.choice(models)
    text = ollama_chat(model, prompt, host=host, temperature=temperature)
    if not text:
        # 失败，尝试下一个模型
        for m in models:
            if m == model:
                continue
            text = ollama_chat(m, prompt, host=host, temperature=temperature)
            if text:
                model = m
                break
    if not text:
        return None

    parsed = _parse_qa(text)
    if not parsed:
        # 解析失败 — 让教师把它"整理成 JSON"再试一次
        fix_prompt = (
            f"请把下面这段内容整理成严格的 JSON 格式，"
            f"包含 question（题目）和 answer（解答）两个字段：\n\n{text}"
        )
        text2 = ollama_chat(model, fix_prompt, host=host, temperature=0.3)
        if text2:
            parsed = _parse_qa(text2)

    if not parsed:
        return None

    return {
        "prompt": parsed["question"],
        "answer": parsed["answer"],
        "subject": subject,
        "teacher_model": model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ========== DPO 偏好对生成 ==========

def generate_dpo(models: List[str], subject: str,
                 host: str = "http://localhost:11434",
                 n_answers: int = 3) -> Optional[Dict]:
    """
    生成一条 DPO 偏好对：
      - 同一个问题让 n 个模型（或同一个模型不同温度）各答一次
      - 让主教师（默认第一个模型）选出最好 / 最差
    """
    # 1. 先出一个问题
    tmpl = random.choice(QUESTION_TEMPLATES)
    prompt_text = tmpl.format(subject=subject)
    question_parsed = generate_qa(models[:1], subject, host=host)
    if not question_parsed:
        return None
    question = question_parsed["prompt"]

    # 2. 用不同模型/不同温度多答几次
    answers = []
    used_models = []
    tried = set()
    attempts = 0
    while len(answers) < n_answers and attempts < n_answers * 3:
        attempts += 1
        model = random.choice(models)
        temp = random.choice([0.5, 0.7, 0.9, 1.1])
        key = (model, temp, len(answers))
        if key in tried:
            continue
        tried.add(key)

        a = ollama_chat(model, f"请回答下面的 {subject} 题目，给出详细解答：\n\n{question}",
                        host=host, temperature=temp, max_tokens=1500)
        if a and len(a) >= 80:
            answers.append(a)
            used_models.append(model)

    if len(answers) < 2:
        return None

    # 3. 让主教师挑出最好/最差
    answers_text = "\n\n".join(f"【解答 {i}】（来自 {used_models[i]}）\n{a}"
                               for i, a in enumerate(answers))
    judge_prompt = DPO_JUDGE_PROMPT.format(
        subject=subject, n_answers=len(answers),
        question=question, answers_text=answers_text,
    )
    judge_model = models[0]
    judge_out = ollama_chat(judge_model, judge_prompt,
                            host=host, temperature=0.3, max_tokens=800)
    if not judge_out:
        return None

    # 解析 JSON
    import re
    try:
        # 清理 ```json ... ```
        t = judge_out.strip()
        if t.startswith("```"):
            t = t.strip("`").strip()
            if t.lower().startswith("json"):
                t = t[4:].strip()
        obj = json.loads(t)
    except Exception:
        # 用正则兜底
        m_chosen = re.search(r'chosen["\s:：]+(\d+)', judge_out)
        m_reject = re.search(r'reject(?:ed)?["\s:：]+(\d+)', judge_out)
        if not (m_chosen and m_reject):
            return None
        obj = {
            "chosen_index": int(m_chosen.group(1)),
            "rejected_index": int(m_reject.group(1)),
            "reason": judge_out[:500],
        }

    chosen_idx = int(obj.get("chosen_index", 0))
    rejected_idx = int(obj.get("rejected_index", 0))
    if chosen_idx == rejected_idx or \
       not (0 <= chosen_idx < len(answers)) or \
       not (0 <= rejected_idx < len(answers)):
        return None

    return {
        "prompt": question,
        "chosen": answers[chosen_idx],
        "rejected": answers[rejected_idx],
        "all_answers": answers,
        "subject": subject,
        "judge_model": judge_model,
        "answerer_models": used_models,
        "judge_reason": obj.get("reason", ""),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ========== CLI ==========

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load_existing(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items


def main():
    parser = argparse.ArgumentParser(
        description="用 Ollama / Trae SOLO 大模型生成 SFT/DPO 训练数据")
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft",
                        help="生成模式：sft（问答对）或 dpo（偏好对）")
    parser.add_argument("--output", required=True,
                        help="输出 JSONL 文件路径（追加写入）")
    parser.add_argument("--num", type=int, default=500,
                        help="要生成的样本数量")
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5:7b-instruct", "deepseek-r1:1.5b"],
                        help="教师模型列表（Ollama model tag）")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama 服务地址")
    parser.add_argument("--subjects", nargs="*", default=None,
                        help="学科列表（留空则用内置的完整列表）")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="教师模型生成温度，越高越有多样性")
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="每条样本之间的间隔秒数（避免打崩 Ollama）")
    parser.add_argument("--resume", action="store_true",
                        help="继续追加到已有的 output 文件")
    args = parser.parse_args()

    subjects = args.subjects or SUBJECTS

    # 加载已有（用于去重 & 续跑）
    existing = _load_existing(args.output) if args.resume else []
    seen_prompts = {item.get("prompt", "")[:120] for item in existing}
    target = args.num
    generated = 0
    failed = 0

    _ensure_dir(args.output)

    mode_fn = generate_dpo if args.mode == "dpo" else generate_qa
    print(f"[INFO] 模式={args.mode}  目标={target}  教师模型={args.models}")
    print(f"[INFO] 已有样本={len(existing)}  输出={args.output}")

    # 打开文件，追加写入
    with open(args.output, "a", encoding="utf-8") as f:
        while generated < target and failed < target * 5:
            subject = random.choice(subjects)
            try:
                item = mode_fn(args.models, subject,
                               host=args.host, temperature=args.temperature)
            except KeyboardInterrupt:
                print("\n[中断] 用户中断，已保存当前进度。")
                break
            except Exception as e:
                print(f"[异常] {e}", file=sys.stderr)
                failed += 1
                time.sleep(args.sleep)
                continue

            if not item:
                failed += 1
                time.sleep(args.sleep)
                continue

            # 去重
            key = item.get("prompt", "")[:120]
            if key in seen_prompts:
                failed += 1
                time.sleep(args.sleep * 0.3)
                continue
            seen_prompts.add(key)

            # 质量检查：SFT 要求 answer 至少 80 字；DPO 要求 chosen 比 rejected 长且合理
            if args.mode == "sft":
                if len(item.get("answer", "")) < 80:
                    failed += 1
                    continue
            elif args.mode == "dpo":
                if len(item.get("chosen", "")) < 80 or len(item.get("rejected", "")) < 50:
                    failed += 1
                    continue

            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            generated += 1

            if generated % 10 == 0 or generated == target:
                print(f"[进度] {generated}/{target}  失败={failed}  "
                      f"学科={item.get('subject')}")
            time.sleep(args.sleep)

    print(f"[完成] 成功生成 {generated} 条，失败 {failed} 条，输出到 {args.output}")


if __name__ == "__main__":
    main()
