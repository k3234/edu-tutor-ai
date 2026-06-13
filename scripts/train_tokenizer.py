"""
训练 BPE 分词器

使用 HuggingFace tokenizers 库训练 BPE 分词器，
为中文教育领域定制词表。

训练流程:
    1. 读取原始文本语料
    2. 训练 BPE 分词器（指定词表大小）
    3. 保存分词器文件
    4. 测试分词效果

使用:
    python scripts/train_tokenizer.py --data_dir data/raw --output_dir data/tokenizer --vocab_size 32000
"""

import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing


# 特殊 Token 定义
SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
}


def train_tokenizer(
    data_dir: str,
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
) -> Tokenizer:
    """
    训练 BPE 分词器

    Args:
        data_dir: 原始文本文件目录
        output_dir: 分词器保存目录
        vocab_size: 词表大小
        min_frequency: 最小词频（低于此频率的合并操作会被忽略）

    Returns:
        训练好的 Tokenizer 对象
    """
    print(f"开始训练 BPE 分词器...")
    print(f"  数据目录: {data_dir}")
    print(f"  词表大小: {vocab_size}")
    print(f"  输出目录: {output_dir}")

    # 1. 初始化 BPE 模型
    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS["unk_token"]))

    # 2. 设置预分词器（按空白字符切分）
    tokenizer.pre_tokenizer = Whitespace()

    # 3. 配置训练器
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS.values()),
        # 中文优化: 允许合并任意 Unicode 字符
        continuing_subword_prefix="",
        end_of_word_suffix="",
    )

    # 4. 收集训练文件
    data_path = Path(data_dir)
    files = list(data_path.glob("*.txt"))
    if not files:
        raise ValueError(f"在 {data_dir} 中未找到 .txt 文件")

    print(f"  找到 {len(files)} 个训练文件")

    # 5. 训练
    tokenizer.train([str(f) for f in files], trainer)

    # 6. 设置后处理模板（自动添加 BOS/EOS）
    tokenizer.post_processor = TemplateProcessing(
        single=f"{SPECIAL_TOKENS['bos_token']} $A {SPECIAL_TOKENS['eos_token']}",
        special_tokens=[
            (SPECIAL_TOKENS["bos_token"], tokenizer.token_to_id(SPECIAL_TOKENS["bos_token"])),
            (SPECIAL_TOKENS["eos_token"], tokenizer.token_to_id(SPECIAL_TOKENS["eos_token"])),
        ],
    )

    # 7. 保存
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # 8. 同时保存为 HuggingFace 格式（便于后续使用）
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=SPECIAL_TOKENS["pad_token"],
        unk_token=SPECIAL_TOKENS["unk_token"],
        bos_token=SPECIAL_TOKENS["bos_token"],
        eos_token=SPECIAL_TOKENS["eos_token"],
    )
    hf_tokenizer.save_pretrained(output_dir)

    print(f"\n分词器训练完成！")
    print(f"  保存路径: {output_dir}")
    print(f"  实际词表大小: {tokenizer.get_vocab_size()}")

    return tokenizer


def test_tokenizer(tokenizer_path: str):
    """测试分词器效果"""
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    test_texts = [
        "勾股定理是指直角三角形两直角边的平方和等于斜边的平方。",
        "牛顿第一定律：任何物体都要保持匀速直线运动或静止状态。",
        "光合作用的化学方程式：6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("\n分词器测试:")
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\n原文: {text}")
        print(f"Token数: {len(tokens)}")
        print(f"解码后: {decoded}")


def main():
    parser = argparse.ArgumentParser(description="训练 BPE 分词器")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="原始文本目录")
    parser.add_argument("--output_dir", type=str, default="data/tokenizer", help="输出目录")
    parser.add_argument("--vocab_size", type=int, default=32000, help="词表大小")
    parser.add_argument("--min_frequency", type=int, default=2, help="最小词频")
    parser.add_argument("--test", action="store_true", help="测试模式（不训练，只测试已有分词器）")

    args = parser.parse_args()

    if args.test:
        test_tokenizer(args.output_dir)
    else:
        tokenizer = train_tokenizer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        test_tokenizer(args.output_dir)


if __name__ == "__main__":
    main()