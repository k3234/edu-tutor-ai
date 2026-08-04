"""
Checkpoint → HuggingFace 格式转换

将训练好的 PyTorch checkpoint 转换为 HuggingFace Transformers 格式，
便于使用 AirLLM 进行低显存推理。

转换内容:
    - model_state_dict → model.safetensors / pytorch_model.bin
    - config → config.json
    - tokenizer → tokenizer_config.json + vocab.json

使用:
    python scripts/convert_to_hf.py \
        --checkpoint experiments/exp_xxx/checkpoints/best.pt \
        --output_dir models/lmm_small_hf \
        --tokenizer_dir data/tokenizer
"""

import argparse
import os
import sys
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel


def convert_checkpoint_to_hf(checkpoint_path: str, output_dir: str, tokenizer_dir: str = None):
    """
    将训练 checkpoint 转换为 HuggingFace 格式

    Args:
        checkpoint_path: 训练 checkpoint 路径
        output_dir: HuggingFace 格式输出目录
        tokenizer_dir: 分词器目录（可选，如果提供则一并复制）
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 1. 恢复模型配置
    # 优先读 checkpoint 里保存的 config；否则从权重形状推断
    model_cfg = None
    sd = checkpoint.get("model_state_dict", checkpoint)
    if checkpoint.get("config"):
        try:
            model_cfg = LMMConfig.from_dict(checkpoint["config"])
        except Exception:
            model_cfg = None
    if model_cfg is None:
        wte_shape = sd["wte.weight"].shape
        wpe_shape = sd["wpe.weight"].shape
        n_layer = sum(1 for k in sd if k.endswith("attn.c_attn.weight"))
        n_embd = wte_shape[1]
        ffn_dim = sd["blocks.0.mlp.gate_proj.weight"].shape[0]
        model_cfg = LMMConfig(
            vocab_size=wte_shape[0],
            n_layer=n_layer,
            n_embd=n_embd,
            n_head=max(1, n_embd // 64),
            ffn_dim=ffn_dim,
            max_seq_len=wpe_shape[0],
        )
    print(f"恢复模型配置: {model_cfg}")

    # 2. 初始化模型并加载权重
    print("初始化模型...")
    model = LMMModel(model_cfg)
    model.load_state_dict(sd, strict=False)
    model.eval()

    # 3. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 4. 保存模型权重（PyTorch 格式）
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存: {model_path}")

    # 5. 保存配置（config.json）
    hf_config = {
        "architectures": ["LMMModel"],
        "model_type": "gpt2",  # 使用 gpt2 类型便于 HF 生态兼容
        "vocab_size": model_cfg.vocab_size,
        "n_layer": model_cfg.n_layer,
        "n_head": model_cfg.n_head,
        "n_embd": model_cfg.n_embd,
        "n_positions": model_cfg.max_seq_len,
        "n_ctx": model_cfg.max_seq_len,
        "ffn_dim": model_cfg.ffn_dim,
        "dropout": model_cfg.dropout,
        "bos_token_id": 2,  # <bos>
        "eos_token_id": 3,  # <eos>
        "pad_token_id": 0,  # <pad>
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)
    print(f"配置文件已保存: {config_path}")

    # 6. 复制分词器文件
    if tokenizer_dir and os.path.exists(tokenizer_dir):
        import shutil
        for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
            src = os.path.join(tokenizer_dir, f)
            if os.path.exists(src):
                dst = os.path.join(output_dir, f)
                shutil.copy2(src, dst)
                print(f"分词器文件已复制: {f}")

    # 创建模型卡（README）
    loss_val = checkpoint.get("loss")
    step_val = checkpoint.get("step", "unknown")
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# LMM-Small

基于 LMM 项目训练的中文教育领域语言模型。

## 模型信息

- 架构: GPT-style Decoder-only
- 参数量: ~120M
- 层数: {model_cfg.n_layer}
- 注意力头数: {model_cfg.n_head}
- 嵌入维度: {model_cfg.n_embd}
- 上下文长度: {model_cfg.max_seq_len}

## 训练信息

- 训练步数: {step_val}
- 最佳损失: {loss_val if loss_val is not None else 'unknown'}
- 训练框架: PyTorch (从零实现)

## 使用

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{output_dir}")
tokenizer = AutoTokenizer.from_pretrained("{output_dir}")
```
""")
    print(f"模型卡已保存: {readme_path}")

    print(f"\n转换完成! HuggingFace 格式模型保存于: {output_dir}")
    print(f"现在可以使用 AirLLM 加载该模型进行低显存推理。")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint 转 HuggingFace 格式")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, required=True, help="HuggingFace 格式输出目录")
    parser.add_argument("--tokenizer_dir", type=str, default="data/tokenizer", help="分词器目录")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"错误: Checkpoint 文件不存在: {args.checkpoint}")
        return

    convert_checkpoint_to_hf(args.checkpoint, args.output_dir, args.tokenizer_dir)


if __name__ == "__main__":
    main()