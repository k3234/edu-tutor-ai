# -*- coding: utf-8 -*-
"""
train_sft.py — 指令微调（Supervised Fine-Tuning）

核心原理：
    给定 (system, user, assistant) 三元组，把三段文本拼接成一个序列，
    但只在 assistant 部分的 token 上计算交叉熵损失 — 让模型学会"按指令给出回答"。

    典型的训练样例：
        system: "你是一位严谨的数学老师..."
        user:   "请解释勾股定理"
        assistant: "勾股定理是指..."

    构建序列：
        tokens = [system_tokens] + [user_tokens] + [assistant_tokens]
        labels = [-100, -100, ..., ..., ...]  # 前两部分 mask 为 -100，最后部分保留真实 token

使用：
    # 基于预训练模型做 SFT（全参数微调）
    python scripts/train_sft.py --data data/sft/train.jsonl \
        --checkpoint experiments/exp_small_v1/checkpoints/best.pt \
        --exp-dir experiments/exp_sft_01 --steps 2000 --batch-size 4

    # 用 LoRA（显存不足时优先选这个）
    python scripts/train_sft.py --data data/sft/train.jsonl \
        --use-lora --lora-rank 8 \
        --checkpoint experiments/exp_small_v1/checkpoints/best.pt \
        --exp-dir experiments/exp_sft_lora_01 --steps 3000
"""

import argparse
import json
import os
import random
import sys
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel
from models.lora import LoraModel


# ============ 特殊 token 占位符 ============
# 由于我们的 BPE tokenizer 没有专门的 instruction token，
# 这里用"文本分隔符"来模拟效果。之后训练完再接入真的 chat template。
SYSTEM_PREFIX = "系统: "
USER_PREFIX = "\n用户: "
ASSISTANT_PREFIX = "\n老师: "
ASSISTANT_SUFFIX = " "  # 表示对话结束


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============ 数据处理 ============

def load_sft_data(path: str) -> List[Dict]:
    """加载 JSONL 数据"""
    if not os.path.exists(path):
        print(f"[ERROR] 数据文件不存在: {path}", file=sys.stderr)
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
                continue
    print(f"[INFO] 加载 SFT 样本: {len(items)} 条")
    return items


def simple_char_tokenizer(text: str, max_seq_len: int, vocab_size: int = 32000) -> List[int]:
    """
    简单的字符级 tokenizer — 当没有训练 BPE tokenizer 时的兜底方案。
    把每个字节映射到 token id（0-255 为字节区，256-vocab_size-1 留给未来扩展）。

    生产环境请使用 scripts/train_tokenizer.py 训练的 BPE tokenizer。
    """
    encoded = text.encode("utf-8")
    tokens = [b for b in encoded]
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    return tokens


def build_training_sequence(item: Dict, max_seq_len: int, vocab_size: int) -> Tuple[List[int], List[int]]:
    """
    把一条 (system, user, assistant) 变成 (input_ids, labels)。
    labels 中 system / user 部分填 -100（被 CE 忽略），只在 assistant 部分计算损失。
    """
    system = item.get("system", "") or ""
    user = item.get("user", "") or ""
    assistant = item.get("assistant", "") or ""

    # 组装完整文本
    # 格式: [系统: ...][用户: ...][老师: ...]
    sys_tokens = simple_char_tokenizer(SYSTEM_PREFIX + system + "\n", max_seq_len, vocab_size)
    user_tokens = simple_char_tokenizer(USER_PREFIX + user + "\n", max_seq_len, vocab_size)
    # 加一个 ASSISTANT_PREFIX 让模型学会 "到这里就该开始回答"
    ans_tokens = simple_char_tokenizer(ASSISTANT_PREFIX + assistant + "\n", max_seq_len, vocab_size)

    # 如果总长度超了，优先截断 system，其次 user，最后才截 assistant
    total_keep = max_seq_len - 2
    while len(sys_tokens) + len(user_tokens) + len(ans_tokens) > total_keep:
        if len(sys_tokens) > 40:
            sys_tokens = sys_tokens[:len(sys_tokens) - 20]
        elif len(user_tokens) > 40:
            user_tokens = user_tokens[:len(user_tokens) - 20]
        elif len(ans_tokens) > 40:
            ans_tokens = ans_tokens[:len(ans_tokens) - 20]
        else:
            break

    # 组装 input / label
    input_ids = sys_tokens + user_tokens + ans_tokens
    labels = [-100] * len(sys_tokens) + [-100] * len(user_tokens) + list(ans_tokens)

    assert len(input_ids) == len(labels), f"len(input_ids)={len(input_ids)} != len(labels)={len(labels)}"
    return input_ids, labels


class SFTDataset(torch.utils.data.Dataset):
    """
    SFT 数据集。返回 (input_ids, labels)，labels 中非 assistant 位置是 -100。
    """
    def __init__(self, data: List[Dict], max_seq_len: int, vocab_size: int):
        self.data = data
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        input_ids, labels = build_training_sequence(
            self.data[idx], self.max_seq_len, self.vocab_size
        )
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_fn(batch):
    """
    把一批不同长度的序列 pad 到同一长度。
    用 0 填充 input_ids（对应 tokenizer 的 pad token），
    用 -100 填充 labels（被 loss 忽略）。
    """
    max_len = max(x[0].shape[0] for x in batch)
    bs = len(batch)
    padded_inputs = torch.zeros(bs, max_len, dtype=torch.long)
    padded_labels = torch.full((bs, max_len), -100, dtype=torch.long)
    for i, (inp, lbl) in enumerate(batch):
        L = inp.shape[0]
        padded_inputs[i, :L] = inp
        padded_labels[i, :L] = lbl
    # 去掉空的样本
    return padded_inputs, padded_labels


# ============ 训练主循环 ============

def train(args):
    # 设备
    device = get_device()
    print(f"[INFO] 使用设备: {device}")

    # 1. 确定模型配置
    # 先看从哪个 checkpoint 恢复 — 它一般带有完整权重
    checkpoint_path = args.checkpoint
    # 如果没给 checkpoint，就创建一个空的 LMM-Small 模型（用于测试 pipeline）
    config = LMMConfig()
    if args.n_layer:
        config.n_layer = args.n_layer
    if args.n_embd:
        config.n_embd = args.n_embd
        config.ffn_dim = 4 * args.n_embd

    model = LMMModel(config)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # 过滤掉不匹配的键（例如预训练和 SFT 之间 embedding 维度可能不同）
        filtered = {}
        for k, v in state_dict.items():
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
                filtered[k] = v
            else:
                print(f"[WARN] 跳过参数: {k} 形状不匹配")
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[INFO] 加载 checkpoint: {checkpoint_path}")
        print(f"[INFO] 缺失键 {len(missing)} 个，多余键 {len(unexpected)} 个")
    else:
        print(f"[WARN] 没有 checkpoint — 从头训练（仅测试 pipeline）")
    model = model.to(device)

    # 2. 如果用 LoRA，冻结大模型，只训练小矩阵
    if args.use_lora:
        lora_model = LoraModel(
            model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=["c_attn", "c_proj", "fc_1", "fc_2"],
        )
        trainable_params = list(lora_model.lora_parameters())
        if not trainable_params:
            print("[ERROR] LoRA 没有找到可训练参数 — 可能是 module 名不匹配。"
                  "请检查模型实际使用的线性层名称。", file=sys.stderr)
            sys.exit(1)
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
        print(f"[INFO] LoRA 模式 — 只训练 {sum(p.numel() for p in trainable_params):,} 个参数")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 3. 数据
    sft_data = load_sft_data(args.data)
    if len(sft_data) == 0:
        print("[ERROR] 没有可用的 SFT 数据。请先运行 teacher_inference.py",
              file=sys.stderr)
        sys.exit(1)
    random.seed(42)
    random.shuffle(sft_data)
    # 切 5% 做验证
    n_val = max(1, int(len(sft_data) * 0.05))
    val_data = sft_data[:n_val]
    train_data = sft_data[n_val:]

    dataset = SFTDataset(train_data, config.max_seq_len, config.vocab_size)
    val_dataset = SFTDataset(val_data, config.max_seq_len, config.vocab_size)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # 4. 实验目录
    exp_dir = args.exp_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 5. 训练
    model.train()
    step = 0
    best_loss = float("inf")
    running_loss = 0.0
    report_steps = 20
    val_interval = max(100, args.steps // 20)

    print(f"\n{'='*60}\n开始 SFT 训练\n总步数: {args.steps}  batch: {args.batch_size}"
          f"  lr: {args.lr}\n训练样本: {len(train_data)}  验证样本: {len(val_data)}\n{'='*60}\n")

    while step < args.steps:
        for input_ids, labels in loader:
            if step >= args.steps:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 前向
            logits, _ = model(input_ids)  # 我们的模型第二个返回值只有在传 targets 时才是 loss
            # 自己算 CE，因为我们要忽略 -100
            # logits: (B, T, V) -> (B*T, V)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
            # 过滤掉 -100
            mask = flat_labels != -100
            if mask.sum() == 0:
                continue  # 没有要优化的 token
            active_logits = flat_logits[mask]
            active_labels = flat_labels[mask]
            # 再做一次 vocab_size 裁剪（如果预训练用了不同大小）
            V = active_logits.size(-1)
            active_labels = torch.clamp(active_labels, 0, V - 1)
            loss = F.cross_entropy(active_logits, active_labels)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in (lora_model.lora_parameters() if args.use_lora else model.parameters())],
                max_norm=1.0,
            )
            optimizer.step()

            running_loss += loss.item()
            step += 1

            # 报告
            if step % report_steps == 0:
                avg = running_loss / report_steps
                print(f"  step {step}/{args.steps}  loss={avg:.4f}  lr={args.lr:.2e}"
                      f"  tokens/batch={input_ids.numel()}")
                running_loss = 0.0

            # 验证
            if step % val_interval == 0:
                val_loss = evaluate(model, val_loader, device)
                print(f"  [VAL] step {step}  val_loss={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    # 保存最佳模型
                    save_path = os.path.join(ckpt_dir, "best.pt")
                    save_dict = {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "loss": best_loss,
                        "config": config.to_dict(),
                    }
                    torch.save(save_dict, save_path)
                    print(f"  [SAVE] 保存最佳模型到 {save_path}  (loss={best_loss:.4f})")
                model.train()

    # 训练结束
    save_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": best_loss,
        "config": config.to_dict(),
    }, save_path)
    print(f"\n[DONE] SFT 训练完成！最终模型: {save_path}")

    # 如果是 LoRA，额外保存一份小的 LoRA 权重
    if args.use_lora:
        lora_save = os.path.join(ckpt_dir, "lora_only.pt")
        lora_state = {
            "lora_A": {}, "lora_B": {}, "rank": args.lora_rank, "alpha": args.lora_alpha,
        }
        for name, p in model.named_parameters():
            if "lora_A" in name:
                lora_state["lora_A"][name] = p.data.cpu()
            elif "lora_B" in name:
                lora_state["lora_B"][name] = p.data.cpu()
        torch.save(lora_state, lora_save)
        print(f"[DONE] LoRA 小权重已额外保存到 {lora_save}")


@torch.no_grad()
def evaluate(model, loader, device):
    """在验证集上评估平均 loss"""
    model.eval()
    total_loss = 0.0
    n = 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        logits, _ = model(input_ids)
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        mask = flat_labels != -100
        if mask.sum() == 0:
            continue
        active_logits = flat_logits[mask]
        active_labels = flat_labels[mask]
        V = active_logits.size(-1)
        active_labels = torch.clamp(active_labels, 0, V - 1)
        loss = F.cross_entropy(active_logits, active_labels)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="SFT 指令微调训练")
    p.add_argument("--data", required=True, help="SFT JSONL 数据路径")
    p.add_argument("--checkpoint", default=None, help="预训练模型 checkpoint 路径")
    p.add_argument("--exp-dir", default="experiments/exp_sft_default",
                   help="实验输出目录")
    p.add_argument("--steps", type=int, default=2000, help="训练步数")
    p.add_argument("--batch-size", type=int, default=4, help="batch size")
    p.add_argument("--lr", type=float, default=5e-4, help="学习率")

    # LoRA 相关参数
    p.add_argument("--use-lora", action="store_true", help="启用 LoRA 训练（省显存）")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA 秩 r，越小越省")
    p.add_argument("--lora-alpha", type=int, default=16, help="LoRA 缩放因子")

    # 模型结构
    p.add_argument("--n-layer", type=int, default=None, help="模型层数（覆盖配置）")
    p.add_argument("--n-embd", type=int, default=None, help="模型嵌入维度（覆盖配置）")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
