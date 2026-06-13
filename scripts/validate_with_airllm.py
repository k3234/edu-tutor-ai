"""
使用 AirLLM 进行低显存推理验证

训练完成后，用 AirLLM 加载转换好的 HuggingFace 格式模型，
在低显存环境下验证模型生成质量。

AirLLM 核心优势:
    - 分层加载: 模型存硬盘，只加载当前计算层到显存
    - 4GB VRAM 即可运行 120M 模型（未来可扩展到更大模型）

使用:
    python scripts/validate_with_airllm.py --model_dir models/lmm_small_hf
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 教育领域评估 prompt
EDU_PROMPTS = [
    # 数学
    "勾股定理的内容是",
    "一元二次方程的求根公式是",
    "圆的面积公式是",

    # 物理
    "牛顿第一定律的内容是",
    "能量守恒定律指出",
    "欧姆定律的公式是",

    # 化学
    "光合作用的化学方程式是",
    "元素周期表中第一个元素是",
    "酸碱中和反应的产物是",

    # 生物
    "细胞的基本结构包括",
    "DNA的四种碱基是",
    "自然选择学说的核心是",

    # 语文/历史
    "《诗经》分为哪三部分",
    "唐朝的开国皇帝是",
    "李白被后人尊称为",

    # 英语
    "The capital of China is",
    "Water is composed of",
]


def validate_with_airllm(model_dir: str, max_new_tokens: int = 50):
    """
    使用 AirLLM 验证模型

    Args:
        model_dir: HuggingFace 格式模型目录
        max_new_tokens: 每次生成最大 token 数
    """
    print("=" * 70)
    print("AirLLM 低显存推理验证")
    print("=" * 70)

    try:
        from airllm import AutoModel
    except ImportError:
        print("错误: 未安装 AirLLM。请运行: pip install airllm")
        return

    # 检查模型目录
    if not os.path.exists(model_dir):
        print(f"错误: 模型目录不存在: {model_dir}")
        print("请先运行: python scripts/convert_to_hf.py 转换模型")
        return

    # 加载模型（AirLLM 会自动分层加载）
    print(f"\n加载模型: {model_dir}")
    print("（AirLLM 分层加载中，首次加载需要分解模型，请耐心等待...）")

    try:
        model = AutoModel.from_pretrained(model_dir)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("\n提示: 如果模型结构不被 AirLLM 直接支持，可以:")
        print("  1. 使用 transformers AutoModel 直接加载（需要更多显存）")
        print("  2. 将模型上传到 HuggingFace Hub 后再用 AirLLM 加载")
        return

    # 运行评估
    print("\n" + "=" * 70)
    print("开始评估生成质量")
    print("=" * 70)

    results = []
    for i, prompt in enumerate(EDU_PROMPTS, 1):
        print(f"\n[{i}/{len(EDU_PROMPTS)}] Prompt: {prompt}")

        try:
            # AirLLM generate
            # 注意: AirLLM 的接口可能与标准 transformers 略有不同
            # 这里使用通用接口
            inputs = model.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            input_ids = inputs['input_ids']

            if hasattr(model, 'generate'):
                generation_output = model.generate(
                    input_ids.cuda() if hasattr(input_ids, 'cuda') else input_ids,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                output_text = model.tokenizer.decode(generation_output.sequences[0])
            else:
                # 如果 AirLLM 不支持 generate，使用简单推理
                output_text = "[AirLLM generate 接口不可用，请检查版本]"

            print(f"Output: {output_text}")
            results.append({"prompt": prompt, "output": output_text})

        except Exception as e:
            print(f"生成失败: {e}")
            results.append({"prompt": prompt, "output": f"[错误: {e}]"})

    # 保存结果
    print("\n" + "=" * 70)
    print("评估完成")
    print("=" * 70)

    import json
    result_file = "validation_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {result_file}")

    # 统计
    success_count = sum(1 for r in results if not r["output"].startswith("["))
    print(f"成功生成: {success_count}/{len(EDU_PROMPTS)}")


def validate_with_transformers(model_dir: str, max_new_tokens: int = 50):
    """
    使用标准 Transformers 验证（备用方案，需要更多显存）

    Args:
        model_dir: HuggingFace 格式模型目录
        max_new_tokens: 每次生成最大 token 数
    """
    print("=" * 70)
    print("Transformers 标准推理验证（备用方案）")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("错误: 未安装 transformers。请运行: pip install transformers")
        return

    print(f"\n加载模型: {model_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"模型已加载到: {device}")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    print("\n开始评估...")
    for i, prompt in enumerate(EDU_PROMPTS[:5], 1):  # 只测前5个
        print(f"\n[{i}] Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_k=50,
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Output: {output_text}")


def main():
    parser = argparse.ArgumentParser(description="AirLLM 推理验证")
    parser.add_argument("--model_dir", type=str, required=True, help="HuggingFace 格式模型目录")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="最大生成 token 数")
    parser.add_argument("--use_transformers", action="store_true", help="使用标准 Transformers（不用 AirLLM）")

    args = parser.parse_args()

    if args.use_transformers:
        validate_with_transformers(args.model_dir, args.max_new_tokens)
    else:
        validate_with_airllm(args.model_dir, args.max_new_tokens)


if __name__ == "__main__":
    main()