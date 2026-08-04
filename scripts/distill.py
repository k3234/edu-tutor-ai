# -*- coding: utf-8 -*-
"""
distill.py — 知识蒸馏（Knowledge Distillation）

把一个较大的模型（teacher，已经经过预训练 + SFT + DPO）
的知识"蒸馏"到一个较小的模型（student）中。

核心思想：
    让 student 的输出概率分布 P_student 逼近 teacher 的输出概率分布 P_teacher。
    不是直接学正确答案，而是学"教师模型对每个可能答案的置信度"。
    这样小模型能学到大模型的"不确定感"和细节。

损失函数（Hinton 2015）：
    L = alpha * KL( softmax(z_T / T) || softmax(z_S / T) )
      + (1 - alpha) * CE( z_S, y_true )
    其中 T 是温度（temperature），T 越大软化越明显。

本实现：
    - 加载一个 teacher 模型（来自 checkpoint），冻结权重
    - 创建一个 student 模型（层数/维度更小），随机初始化
    - 在同一批数据上，让 student 的 logits 逼近 teacher 的 logits
    - 同时用真实标签做监督（可选）

使用：
    # 把一个 12 层的模型蒸馏到 4 层
    python scripts/distill.py \
        --teacher-ckpt experiments/exp_dpo_01/checkpoints/best.pt \
        --teacher-n-layer 12 --teacher-n-embd 768 \
        --student-n-layer 4 --student-n-embd 256 \
        --data data/sft/train.jsonl \
        --steps 2000 --temperature 2.0 --alpha 0.7
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel


# ============ 工具函数（和其他脚本保持一致） ============

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


# ============ 数据集 ============

class DistillDataset(torch.utils.data.Dataset):
    """
    蒸馏数据集：每条样本包含 prompt + answer 文本，被拼接成 token 序列。
    训练时让 student 的 token 级 logits 逼近 teacher 的 logits。
    """

    def __init__(self, data: List[Dict], max_seq_len: int, vocab_size: int):
        self.data = data
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        text = (item.get("user") or item.get("prompt", "") or "")
        if item.get("assistant") or item.get("answer"):
            text += "\n" + (item.get("assistant") or item.get("answer", ""))
        tokens = simple_char_tokenizer(text, self.max_seq_len, self.vocab_size)
        if len(tokens) < 5:
            tokens = tokens + [0] * (5 - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)


def collate_pad(batch):
    """把不同长度的序列 pad 到相同长度"""
    max_len = max(x.shape[0] for x in batch)
    bs = len(batch)
    padded = torch.zeros(bs, max_len, dtype=torch.long)
    mask = torch.zeros(bs, max_len, dtype=torch.bool)
    for i, x in enumerate(batch):
        L = x.shape[0]
        padded[i, :L] = x
        mask[i, :L] = True
    return padded, mask


# ============ 蒸馏 loss ============

def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                      labels: torch.Tensor, mask: torch.Tensor,
                      temperature: float, alpha: float) -> torch.Tensor:
    """
    蒸馏损失 = alpha * KL(teacher || student) + (1-alpha) * CE(student, labels)

    student_logits: (B, T, V)
    teacher_logits: (B, T, V)
    labels: (B, T) - 真实 token id（可以设为 -100 表示忽略）
    mask: (B, T) - 哪些 token 是真实的，哪些是 padding

    返回标量 loss
    """
    B, T, V = student_logits.shape
    # 只计算非 padding 位置
    flat_mask = mask.view(-1)  # (B*T,)

    if flat_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

    # 扁平化
    s_logits = student_logits.view(-1, V)[flat_mask]   # (N, V)
    t_logits = teacher_logits.view(-1, V)[flat_mask]   # (N, V)
    flat_labels = labels.view(-1)[flat_mask]           # (N,)

    # 1) KL 散度（蒸馏损失）—— 让学生学老师的软概率
    # 注意 PyTorch KLDiv 需要 log-probs 作为第一个参数
    log_p_s = F.log_softmax(s_logits / temperature, dim=-1)
    p_t = F.softmax(t_logits / temperature, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)
    # 乘以 T²，因为温度缩放了 logits

    # 2) CE 损失（硬标签）—— 让学生同时学正确答案
    valid_labels = flat_labels.clone()
    # 限制 vocab 范围
    valid_labels = torch.clamp(valid_labels, 0, V - 1)
    ce = F.cross_entropy(s_logits, valid_labels)

    return alpha * kl + (1 - alpha) * ce


# ============ 训练主流程 ============

def train(args):
    device = get_device()
    print(f"[INFO] 使用设备: {device}")

    # 1. 构造 teacher 模型 + 加载 checkpoint
    teacher_cfg = LMMConfig(
        vocab_size=args.vocab_size,
        n_layer=args.teacher_n_layer,
        n_embd=args.teacher_n_embd,
        n_head=max(1, args.teacher_n_embd // 64),
        ffn_dim=args.teacher_n_embd * 4,
        max_seq_len=args.max_seq_len,
    )
    teacher = LMMModel(teacher_cfg).to(device)

    if os.path.exists(args.teacher_ckpt):
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        filtered = {k: v for k, v in sd.items()
                    if k in teacher.state_dict()
                    and teacher.state_dict()[k].shape == v.shape}
        missing, unexpected = teacher.load_state_dict(filtered, strict=False)
        print(f"[INFO] 加载 teacher 权重: {args.teacher_ckpt}")
        print(f"[INFO] 缺失 {len(missing)} 个键，多余 {len(unexpected)} 个键")
    else:
        print(f"[WARN] teacher checkpoint 不存在: {args.teacher_ckpt}")
        print("       （会用随机模型做演示 pipeline，正式训练请提供真实 ckpt）")

    # 冻结 teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 2. 构造 student 模型
    student_cfg = LMMConfig(
        vocab_size=args.vocab_size,
        n_layer=args.student_n_layer,
        n_embd=args.student_n_embd,
        n_head=max(1, args.student_n_embd // 64),
        ffn_dim=args.student_n_embd * 4,
        max_seq_len=args.max_seq_len,
    )
    student = LMMModel(student_cfg).to(device)

    # 如果有 student_ckpt，可继续训练
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        ckpt = torch.load(args.student_ckpt, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        filtered = {k: v for k, v in sd.items()
                    if k in student.state_dict()
                    and student.state_dict()[k].shape == v.shape}
        student.load_state_dict(filtered, strict=False)
        print(f"[INFO] 从 {args.student_ckpt} 继续训练 student")

    # 3. 统计参数
    t_params = sum(p.numel() for p in teacher.parameters())
    s_params = sum(p.numel() for p in student.parameters())
    print(f"[INFO] Teacher 参数量: {t_params:,} ({t_params / 1e6:.2f}M)")
    print(f"[INFO] Student 参数量: {s_params:,} ({s_params / 1e6:.2f}M)")
    print(f"[INFO] 压缩比: {t_params / s_params:.2f}×")

    # 4. 加载数据
    sft_data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sft_data.append(json.loads(line))
            except Exception:
                continue
    print(f"[INFO] 训练样本: {len(sft_data)}")

    dataset = DistillDataset(sft_data, args.max_seq_len, args.vocab_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pad, num_workers=0,
    )

    # 5. 优化器
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    # 6. 实验目录
    exp_dir = args.exp_dir
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 7. 训练循环
    print(f"\n{'='*60}\n开始知识蒸馏\n"
          f"总步数: {args.steps}  batch: {args.batch_size}  lr: {args.lr}"
          f"\n温度 T={args.temperature}  alpha={args.alpha}\n{'='*60}\n")

    step = 0
    running_loss = 0.0
    running_kl = 0.0
    running_ce = 0.0
    report_steps = 20
    best_loss = float("inf")
    t0 = time.time()

    while step < args.steps:
        for tokens, mask in loader:
            if step >= args.steps:
                break
            tokens = tokens.to(device)
            mask = mask.to(device)

            # 前向：把 token 序列 (..., T) 输入 teacher 和 student
            # 我们预测下一 token：输入 = tokens[:, :-1]，labels = tokens[:, 1:]
            input_tokens = tokens[:, :-1].contiguous()
            labels = tokens[:, 1:].contiguous()
            t_mask = mask[:, 1:].contiguous()

            # teacher (无梯度)
            with torch.no_grad():
                teacher_logits, _ = teacher(input_tokens)  # (B, T-1, V)

            # student (有梯度)
            student_logits, _ = student(input_tokens)     # (B, T-1, V)

            # 损失
            # 注意 student 和 teacher 的 vocab 必须相同（同一份 tokenizer）
            V_student = student_logits.size(-1)
            V_teacher = teacher_logits.size(-1)
            if V_student != V_teacher:
                # 不同的话，裁到最小
                V_min = min(V_student, V_teacher)
                student_logits = student_logits[:, :, :V_min]
                teacher_logits = teacher_logits[:, :, :V_min]

            # 分开计算 KL 和 CE 方便调试
            flat_mask = t_mask.view(-1)
            if flat_mask.sum() == 0:
                continue

            s_flat = student_logits.view(-1, student_logits.size(-1))[flat_mask]
            t_flat = teacher_logits.view(-1, teacher_logits.size(-1))[flat_mask]
            l_flat = labels.view(-1)[flat_mask]

            # KL
            log_p_s = F.log_softmax(s_flat / args.temperature, dim=-1)
            p_t = F.softmax(t_flat / args.temperature, dim=-1)
            kl = F.kl_div(log_p_s, p_t, reduction="batchmean") * (args.temperature ** 2)

            # CE
            V = s_flat.size(-1)
            valid_labels = torch.clamp(l_flat, 0, V - 1)
            ce = F.cross_entropy(s_flat, valid_labels)

            loss = args.alpha * kl + (1 - args.alpha) * ce

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_kl += kl.item()
            running_ce += ce.item()
            step += 1

            if step % report_steps == 0:
                avg_loss = running_loss / report_steps
                avg_kl = running_kl / report_steps
                avg_ce = running_ce / report_steps
                elapsed = time.time() - t0
                print(f"  step {step}/{args.steps}  loss={avg_loss:.4f}  "
                      f"KL={avg_kl:.4f}  CE={avg_ce:.4f}  "
                      f"用时={elapsed:.1f}s  lr={args.lr:.2e}")
                running_loss = 0.0
                running_kl = 0.0
                running_ce = 0.0

            if step % max(100, args.steps // 10) == 0:
                # 保存中间最佳模型
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    save_path = os.path.join(ckpt_dir, "best.pt")
                    torch.save({
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "loss": best_loss,
                        "config": student_cfg.to_dict(),
                        "teacher_info": {
                            "teacher_ckpt": args.teacher_ckpt,
                            "teacher_n_layer": args.teacher_n_layer,
                            "teacher_n_embd": args.teacher_n_embd,
                            "temperature": args.temperature,
                            "alpha": args.alpha,
                        },
                    }, save_path)
                    print(f"  [SAVE] 保存最佳 student 模型: {save_path} (loss={best_loss:.4f})")

    # 训练结束
    save_path = os.path.join(ckpt_dir, "final.pt")
    torch.save({
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": best_loss,
        "config": student_cfg.to_dict(),
        "teacher_info": {
            "teacher_ckpt": args.teacher_ckpt,
            "teacher_n_layer": args.teacher_n_layer,
            "teacher_n_embd": args.teacher_n_embd,
            "temperature": args.temperature,
            "alpha": args.alpha,
        },
    }, save_path)

    print(f"\n[DONE] 蒸馏完成！Student 模型: {save_path}")
    print(f"       Teacher: {t_params / 1e6:.2f}M params -> Student: {s_params / 1e6:.2f}M params")
    print(f"       压缩比: {t_params / s_params:.2f}×  推理速度约等于压缩比")


def main():
    p = argparse.ArgumentParser(description="知识蒸馏：大模型 -> 小模型")
    p.add_argument("--teacher-ckpt", required=True,
                   help="Teacher 模型的 checkpoint 路径")
    p.add_argument("--teacher-n-layer", type=int, default=12,
                   help="Teacher 模型层数")
    p.add_argument("--teacher-n-embd", type=int, default=768,
                   help="Teacher 模型嵌入维度")

    p.add_argument("--student-n-layer", type=int, default=6,
                   help="Student 模型层数")
    p.add_argument("--student-n-embd", type=int, default=384,
                   help="Student 模型嵌入维度")
    p.add_argument("--student-ckpt", default=None,
                   help="（可选）继续训练的 student checkpoint")

    p.add_argument("--data", default="data/sft/train.jsonl",
                   help="训练数据 JSONL 路径")
    p.add_argument("--exp-dir", default="experiments/exp_distill_01",
                   help="实验输出目录")
    p.add_argument("--steps", type=int, default=2000, help="训练步数")
    p.add_argument("--batch-size", type=int, default=4, help="batch size")
    p.add_argument("--lr", type=float, default=5e-4, help="学习率")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="蒸馏温度 T，T 越大软分布越平滑（1.0~5.0 常用）")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="KL 损失的权重 alpha；(1-alpha) 给 CE 损失")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="最大序列长度")
    p.add_argument("--vocab-size", type=int, default=32000,
                   help="词表大小")

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
