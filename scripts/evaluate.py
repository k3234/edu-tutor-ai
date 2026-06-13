"""
模型评估脚本

评估训练好的模型在多个维度上的表现:
    1. 验证集困惑度 (Perplexity)
    2. 教育领域知识问答
    3. 文本生成质量
    4. 推理速度测试

使用:
    python scripts/evaluate.py --checkpoint experiments/exp_xxx/checkpoints/best.pt
"""

import argparse
import os
import sys
import time
import math

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel
from data.dataset import prepare_dataset
from data.dataloader import create_dataloader


# 教育领域知识测试题
KNOWLEDGE_TESTS = [
    {"question": "勾股定理的内容是什么？", "keywords": ["直角", "平方", "和", "斜边"]},
    {"question": "牛顿第一定律又叫什么？", "keywords": ["惯性"]},
    {"question": "光合作用的产物是什么？", "keywords": ["氧气", "有机物", "葡萄糖"]},
    {"question": "元素周期表是谁发明的？", "keywords": ["门捷列夫"]},
    {"question": "DNA的中文名称是什么？", "keywords": ["脱氧核糖核酸"]},
    {"question": "《诗经》分为哪三部分？", "keywords": ["风", "雅", "颂"]},
    {"question": "唐朝的开国皇帝是谁？", "keywords": ["李渊", "李世民"]},
    {"question": "李白被尊称为什么？", "keywords": ["诗仙"]},
]


def evaluate_perplexity(model, val_loader, device, max_batches: int = None):
    """
    计算验证集困惑度

    困惑度 (Perplexity) 是语言模型的标准评估指标：
        PPL = exp(平均交叉熵损失)
        越低越好，随机猜测的 PPL ≈ 词表大小

    Args:
        model: 模型
        val_loader: 验证集 DataLoader
        device: 计算设备
        max_batches: 最大评估 batch 数（None=全部）

    Returns:
        平均困惑度
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    print("\n计算验证集困惑度...")

    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(val_loader):
            if max_batches and i >= max_batches:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)

            # 计算实际 token 数（排除 padding）
            valid_tokens = (target_ids != -100).sum().item()

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens
            n_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    print(f"  评估 batch 数: {n_batches}")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  困惑度 (PPL): {perplexity:.2f}")

    return perplexity


def evaluate_knowledge(model, tokenizer, device, max_new_tokens: int = 30):
    """
    评估教育领域知识掌握情况

    通过生成回答并检查关键词匹配来评估。

    Args:
        model: 模型
        tokenizer: 分词器
        device: 计算设备
        max_new_tokens: 最大生成 token 数

    Returns:
        准确率 (0-1)
    """
    print("\n评估教育领域知识...")

    model.eval()
    correct = 0
    results = []

    with torch.no_grad():
        for test in KNOWLEDGE_TESTS:
            question = test["question"]
            keywords = test["keywords"]

            # 编码问题
            if hasattr(tokenizer, 'encode'):
                input_ids = tokenizer.encode(question, return_tensors="pt").to(device)
            else:
                # 简单字符编码（fallback）
                input_ids = torch.tensor([[ord(c) for c in question[:50]]]).to(device)

            # 生成回答
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_k=50,
            )

            # 解码
            if hasattr(tokenizer, 'decode'):
                answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                answer = "".join([chr(c) for c in output_ids[0].tolist() if 0 < c < 128])

            # 检查关键词匹配
            matched = sum(1 for kw in keywords if kw in answer)
            is_correct = matched >= len(keywords) // 2  # 匹配一半以上关键词算对

            if is_correct:
                correct += 1

            results.append({
                "question": question,
                "answer": answer,
                "keywords": keywords,
                "matched": matched,
                "correct": is_correct,
            })

            print(f"  Q: {question}")
            print(f"  A: {answer[:100]}...")
            print(f"  关键词匹配: {matched}/{len(keywords)} {'✓' if is_correct else '✗'}")

    accuracy = correct / len(KNOWLEDGE_TESTS)
    print(f"\n知识问答准确率: {correct}/{len(KNOWLEDGE_TESTS)} = {accuracy:.1%}")

    return accuracy, results


def evaluate_generation_quality(model, tokenizer, device):
    """
    评估文本生成质量

    测试续写任务的流畅度和连贯性。

    Args:
        model: 模型
        tokenizer: 分词器
        device: 计算设备

    Returns:
        生成样本列表
    """
    print("\n评估文本生成质量...")

    prompts = [
        "在数学中，",
        "物理学告诉我们，",
        "化学反应的本质是",
        "中国古代的四大发明包括",
        "学习英语的重要性在于",
    ]

    model.eval()
    samples = []

    with torch.no_grad():
        for prompt in prompts:
            if hasattr(tokenizer, 'encode'):
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            else:
                input_ids = torch.tensor([[ord(c) for c in prompt]]).to(device)

            output_ids = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.8,
                top_k=50,
            )

            if hasattr(tokenizer, 'decode'):
                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                generated = prompt + " [解码失败]"

            samples.append({"prompt": prompt, "generated": generated})
            print(f"\n  Prompt: {prompt}")
            print(f"  Generated: {generated}")

    return samples


def evaluate_speed(model, device, seq_len: int = 1024, n_iters: int = 10):
    """
    评估推理速度

    测试模型生成 token 的速度（tokens/second）。

    Args:
        model: 模型
        device: 计算设备
        seq_len: 序列长度
        n_iters: 测试迭代次数

    Returns:
        平均生成速度 (tokens/s)
    """
    print("\n评估推理速度...")

    model.eval()

    # 准备输入
    dummy_input = torch.randint(0, 32000, (1, seq_len)).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # 正式测试
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()

    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(dummy_input)

    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start

    total_tokens = seq_len * n_iters
    tokens_per_sec = total_tokens / elapsed

    print(f"  序列长度: {seq_len}")
    print(f"  迭代次数: {n_iters}")
    print(f"  总耗时: {elapsed:.2f}s")
    print(f"  推理速度: {tokens_per_sec:.0f} tokens/s")

    return tokens_per_sec


def evaluate(checkpoint_path: str, data_config: dict = None):
    """
    主评估函数

    Args:
        checkpoint_path: checkpoint 路径
        data_config: 数据配置（可选）
    """
    print("=" * 70)
    print("LMM 模型评估")
    print("=" * 70)

    # 1. 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")

    config = LMMConfig()
    model = LMMModel(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"模型已加载: {checkpoint_path}")
    print(f"训练步数: {checkpoint.get('step', 'unknown')}")
    print(f"训练损失: {checkpoint.get('loss', 'unknown'):.4f}")

    # 2. 困惑度评估
    if data_config and os.path.exists(data_config.get("token_file", "")):
        _, val_dataset = prepare_dataset(
            token_file=data_config["token_file"],
            seq_len=config.max_seq_len,
            train_ratio=data_config.get("train_ratio", 0.95),
        )
        val_loader = create_dataloader(val_dataset, batch_size=8, shuffle=False)
        ppl = evaluate_perplexity(model, val_loader, device, max_batches=20)
    else:
        print("\n跳过困惑度评估（未提供数据配置或 token 文件不存在）")
        ppl = None

    # 3. 知识问答评估
    # 注意: 需要 tokenizer，如果没有则跳过
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("data/tokenizer")
        accuracy, knowledge_results = evaluate_knowledge(model, tokenizer, device)
    except Exception as e:
        print(f"\n跳过知识问答评估: {e}")
        accuracy = None
        knowledge_results = []

    # 4. 生成质量评估
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("data/tokenizer")
        generation_samples = evaluate_generation_quality(model, tokenizer, device)
    except Exception as e:
        print(f"\n跳过生成质量评估: {e}")
        generation_samples = []

    # 5. 速度评估
    speed = evaluate_speed(model, device)

    # 6. 总结
    print("\n" + "=" * 70)
    print("评估总结")
    print("=" * 70)
    if ppl:
        print(f"验证集困惑度: {ppl:.2f}")
    if accuracy is not None:
        print(f"知识问答准确率: {accuracy:.1%}")
    print(f"推理速度: {speed:.0f} tokens/s")

    # 保存结果
    import json
    results = {
        "checkpoint": checkpoint_path,
        "perplexity": ppl,
        "knowledge_accuracy": accuracy,
        "inference_speed": speed,
        "knowledge_results": knowledge_results,
        "generation_samples": generation_samples,
    }

    result_file = "evaluation_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="LMM 模型评估")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径")
    parser.add_argument("--token_file", type=str, default="data/tokens.pt", help="Token 文件")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"错误: Checkpoint 不存在: {args.checkpoint}")
        return

    data_config = {"token_file": args.token_file, "train_ratio": 0.95}
    evaluate(args.checkpoint, data_config)


if __name__ == "__main__":
    main()