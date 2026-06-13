# -*- coding: utf-8 -*-
"""
corpus_collector.py
===================
采集 & 清洗中文教育领域语料，合并成单一纯文本文件供预训练使用。

支持的输入：
  - 本地目录：txt / md / jsonl 文件（递归查找）
  - stdin：通过管道输入文本

清洗规则：
  - 去除 HTML 标签
  - 合并多余空白
  - 去除极短（< 20 chars）或空行
  - 按行去重
  - 保留中文、英文、数字、常用标点

使用：
    python data/corpus_collector.py --input-dir data/raw_src --output data/raw/edu_corpus.txt
    cat notes.txt | python data/corpus_collector.py --stdin --output data/raw/edu_corpus.txt
"""

import argparse
import html
import os
import re
import sys
import time
from typing import List, Set

# 去除 HTML 标签
HTML_RE = re.compile(r"<[^>]+>")
# 去除 URL
URL_RE = re.compile(r"https?://\S+")
# 多空白归一
WS_RE = re.compile(r"[ \t]+")
# 仅保留常用字符（中文、英文、数字、常用中英文标点）
ALLOWED_RE = re.compile(
    r"[^0-9A-Za-z\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef"
    r",.?!;:，。？！；：、\(\)\[\]\{\}\"''\n\-\+\*\/=<>%#@&$°℃'$^~`]"
)


def clean_text(text: str) -> str:
    """对一段文本做基础清洗。"""
    text = html.unescape(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\r", "\n").replace("\t", " ")
    text = WS_RE.sub(" ", text)
    # 把 3+ 个换行合并成 2 个
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_md(path: str) -> str:
    return read_txt(path)


def read_jsonl(path: str) -> str:
    """
    读取 JSONL，把每条的 text / content / 合并字段 拼起来。
    适合：
      - {"text": "..."}
      - {"content": "..."}
      - {"messages": [{"role":"user","content":"..."}, ...]}
    """
    parts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                import json
                obj = json.loads(line)
            except Exception:
                continue
            for key in ("text", "content", "answer"):
                if isinstance(obj, dict) and key in obj:
                    parts.append(str(obj[key]))
            if isinstance(obj, dict) and "messages" in obj:
                for m in obj["messages"]:
                    if isinstance(m, dict) and "content" in m:
                        parts.append(str(m["content"]))
            if isinstance(obj, dict) and "prompt" in obj and "answer" in obj:
                parts.append(str(obj["prompt"]) + "\n" + str(obj["answer"]))
    return "\n\n".join(parts)


READERS = {
    ".txt": read_txt,
    ".md": read_md,
    ".markdown": read_md,
    ".jsonl": read_jsonl,
}


def iter_files(input_dir: str) -> List[str]:
    out = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in READERS:
                out.append(os.path.join(root, fn))
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description="教育语料采集与清洗")
    p.add_argument("--input-dir", default=None,
                   help="输入目录（递归遍历 .txt/.md/.jsonl）")
    p.add_argument("--stdin", action="store_true",
                   help="从 stdin 读取文本")
    p.add_argument("--output", default="data/raw/edu_corpus.txt",
                   help="输出合并后的纯文本")
    p.add_argument("--min-line", type=int, default=20,
                   help="每行最小字符数，低于此阈值的行被过滤")
    p.add_argument("--dedup", action="store_true", default=True,
                   help="按行去重（默认开）")
    args = p.parse_args()

    if not args.input_dir and not args.stdin:
        print("需要 --input-dir 或 --stdin", file=sys.stderr)
        sys.exit(1)

    # 收集原始文本
    all_text: List[str] = []
    if args.input_dir:
        files = iter_files(args.input_dir)
        print(f"[INFO] 发现 {len(files)} 个文件")
        for i, fp in enumerate(files, 1):
            ext = os.path.splitext(fp)[1].lower()
            reader = READERS.get(ext, read_txt)
            try:
                text = reader(fp)
                all_text.append(text)
            except Exception as e:
                print(f"[WARN] 读取失败 {fp}: {e}", file=sys.stderr)
            if i % 100 == 0 or i == len(files):
                print(f"[进度] {i}/{len(files)} files")

    if args.stdin:
        text = sys.stdin.read()
        all_text.append(text)

    # 合并 & 清洗
    merged = "\n\n".join(all_text)
    merged = clean_text(merged)

    # 逐行过滤 / 去重
    lines = merged.splitlines()
    seen: Set[str] = set()
    kept: List[str] = []
    for line in lines:
        line = line.strip()
        if len(line) < args.min_line:
            continue
        key = line[:120]
        if args.dedup:
            if key in seen:
                continue
            seen.add(key)
        kept.append(line)

    final = "\n".join(kept) + "\n"

    # 写出
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"[完成] 输出 {args.output}  行数={len(kept)}  "
          f"大小={size_mb:.2f} MB  时间={time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
