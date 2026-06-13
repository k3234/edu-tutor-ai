"""
将语料文件用训练好的分词器转换为 token 序列并保存为 .pt 文件
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import PreTrainedTokenizerFast

def main():
    corpus_file = "data/raw/sample_corpus.txt"
    tokenizer_dir = "data/tokenizer"
    output_file = "data/tokens.pt"

    print(f"加载分词器: {tokenizer_dir}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

    print(f"读取语料: {corpus_file}")
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"语料大小: {len(text):,} 字符")

    # 分词
    print("正在分词...")
    tokens = tokenizer.encode(text)

    # 过滤掉特殊token（bos/eos），只保留纯文本token
    # 因为dataset会自行处理序列切分
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    tokens = [t for t in tokens if t not in (bos_id, eos_id)]

    print(f"Token 数量: {len(tokens):,}")
    print(f"词表大小: {tokenizer.vocab_size}")
    print(f"Token 范围: {min(tokens)} ~ {max(tokens)}")

    # 保存
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    torch.save(tokens_tensor, output_file)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nToken 文件已保存: {output_file} ({file_size_mb:.2f} MB)")
    print("数据管道验证通过 ✓")

if __name__ == "__main__":
    main()