# -*- coding: utf-8 -*-
"""
train_dpo.py — Direct Preference Optimization（直接偏好优化）

核心原理（来自 DPO 论文 https://arxiv.org/abs/2305.18290）：

    对同一个 prompt x，教师给出了两个回答：
        y_chosen  — 教师认为更好的那个
        y_rejected — 教师认为更差的那个

    训练目标：让我们的模型 π_θ 对 y_chosen 分配更高概率，对 y_rejected 分配更低概率。
    损失函数：
        L_DPO(θ) = -log σ( beta * ( log π_θ(y_chosen|x) - log π_ref(y_chosen|x)
                                      - log π_θ(y_rejected|x) + log π_ref(y_rejected|x) ) )

    其中 π_ref 是训练开始时的模型（"reference model"，相当于 SFT 后的初始状态）。

直观：
    如果模型现在更"倾向"于 chosen 而不是 rejected（相对于最初状态），损失就小。
    这样模型就会"学习教师的偏好"，输出更像老师认可的回答。

使用：
    python scripts/train_dpo.py --data data/dpo/raw.jsonl \
        --checkpoint experiments/exp_sft_01/checkpoints/best.pt \
        --exp-dir experiments/exp_dpo_01 --beta 0.1 --steps 1000
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel


# ============ 工具函数（与 train_sft.py 保持一致） ============

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def simple_char_tokenizer(text: str, max_seq_len: int, vocab_size: int = 32000) -> List[int]:
    """简单的字符级 tokenizer，同 train_sft.py"""
    encoded = text.encode("utf-8")
    tokens = [b for b in encoded]
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    return tokens


SYSTEM_PREFIX = "系统: "
USER_PREFIX = "\n用户: "
ASSISTANT_PREFIX = "\n老师: "


def build_dpo_sequence(item: Dict, max_seq_len: int, vocab_size: int) -> Tuple[List[int], List[int], int]:
    """
    DPO 不需要 mask 出 assistant 部分来决定 loss 权重，
    而是计算整个序列在 chosen/rejected 两个回答下的 log prob，
    只对 assistant 部分累加（system/user 两者相同，会抵消）。

    返回: (input_tokens, label_tokens, assistant_start)
    assistant_start: assistant 文本开始的 token 索引。只有 >= assistant_start 的位置参与 log prob 计算。
    """
    prompt = item.get("prompt", "") or item.get("user", "") or ""
    subject = item.get("subject", "") or ""

    system_text = item.get("system", f"你是一位{subject}老师，会详细解释问题。")
    # chosen / rejected 回答
    chosen = item.get("chosen", "") or ""
    rejected = item.get("rejected", "") or ""

    # 构造 prompt 部分（system + user）
    prompt_tokens = simple_char_tokenizer(
        SYSTEM_PREFIX + system_text + "\n" + USER_PREFIX + prompt + "\n" + ASSISTANT_PREFIX,
        max_seq_len, vocab_size,
    )
    assistant_start = len(prompt_tokens)

    # chosen 回答 tokens
    chosen_ans = simple_char_tokenizer(chosen + "\n", max_seq_len, vocab_size)
    # rejected 回答 tokens
    rejected_ans = simple_char_tokenizer(rejected + "\n", max_seq_len, vocab_size)

    # 拼接：prompt + chosen / rejected
    # 如果总长超过 max_seq_len，就截断后面的回答
    max_ans = max_seq_len - assistant_start - 1
    if len(chosen_ans) > max_ans:
        chosen_ans = chosen_ans[:max_ans]
    if len(rejected_ans) > max_ans:
        rejected_ans = rejected_ans[:max_ans]

    chosen_tokens = prompt_tokens + chosen_ans
    rejected_tokens = prompt_tokens + rejected_ans

    return chosen_tokens, rejected_tokens, assistant_start


# ============ log prob 计算工具 ============

def compute_log_probs(model, tokens: List[int], assistant_start: int, device: str) -> float:
    """
    计算模型在给定 tokens 序列上，从 assistant_start 开始到结尾的对数概率和。
    ∑ log P(t_i | t_{<i})，i 从 assistant_start 到 len(tokens)-1

    因为模型 forward 会返回 logits 形状 (B, T, V)，
    我们取最后一个维度上对应真实 token 的 log_softmax，然后求和。
    """
    if len(tokens) <= assistant_start + 1:
        return 0.0
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _ = model(input_ids)  # (1, T, V)
    # 去掉最后一个 token 的 logits（它是预测 "结尾之后" 的，我们不需要）
    logits = logits[:, :-1, :]  # (1, T-1, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (1, T-1, V)
    # tokens[1:] 是目标序列
    targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    # 取出每个位置目标 token 的 log prob
    # gather: (1, T-1, 1) -> (1, T-1)
    gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    # 只累加 assistant 部分
    start = max(0, assistant_start - 1)  # 因为我们去掉了第一个位置，索引要 -1
    end = gathered.size(1)
    if end <= start:
        return 0.0
    # 限制在有效范围内
    total = gathered[:, start:end].sum(dim=1).item()
    return total


# ============ 数据集 ============

class DPODataset(torch.utils.data.Dataset):
    """DPO 数据集：返回 (chosen_tokens, rejected_tokens, assistant_start)"""

    def __init__(self, data: List[Dict], max_seq_len: int, vocab_size: int):
        self.data = data
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        chosen_tokens, rejected_tokens, assistant_start = build_dpo_sequence(
            self.data[idx], self.max_seq_len, self.vocab_size
        )
        return (
            torch.tensor(chosen_tokens, dtype=torch.long),
            torch.tensor(rejected_tokens, dtype=torch.long),
            assistant_start,
        )


def dpo_collate(batch):
    """把不同长度的 chosen / rejected 序列 pad 到同一长度"""
    chosen_list, rejected_list, starts = zip(*batch)
    max_len = max(max(t.shape[0] for t in chosen_list),
                  max(t.shape[0] for t in rejected_list))
    bs = len(chosen_list)

    chosen_pad = torch.zeros(bs, max_len, dtype=torch.long)
    rejected_pad = torch.zeros(bs, max_len, dtype=torch.long)
    # mask 用来标记哪些位置是真数据（1 = 真实 token，0 = padding）
    chosen_mask = torch.zeros(bs, max_len, dtype=torch.bool)
    rejected_mask = torch.zeros(bs, max_len, dtype=torch.bool)
    for i in range(bs):
        Lc = chosen_list[i].shape[0]
        Lr = rejected_list[i].shape[0]
        chosen_pad[i, :Lc] = chosen_list[i]
        rejected_pad[i, :Lr] = rejected_list[i]
        chosen_mask[i, :Lc] = True
        rejected_mask[i, :Lr] = True

    return chosen_pad, rejected_pad, torch.tensor(starts, dtype=torch.long), \
           chosen_mask, rejected_mask


# ============ 主训练函数 ============

def compute_log_probs_batch(model, tokens: torch.Tensor, assistant_starts: torch.Tensor,
                            mask: torch.Tensor, device: str) -> torch.Tensor:
    """
    批量计算每个样本在 assistant 部分的 log prob 之和。
    tokens: (B, T)
    assistant_starts: (B,)
    mask: (B, T) — True 表示是真实 token
    返回: (B,) 每个样本的 ∑ log P
    """
    B, T = tokens.shape
    tokens = tokens.to(device)
    logits, _ = model(tokens)  # (B, T, V)
    logits = logits[:, :-1, :]  # (B, T-1, V)  — 对应预测 tokens[1:]
    log_probs = F.log_softmax(logits, dim=-1)

    # 目标: tokens[:, 1:]
    targets = tokens[:, 1:]  # (B, T-1)
    V = log_probs.size(-1)
    targets = torch.clamp(targets, 0, V - 1)

    # gather log probs (B, T-1)
    gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # 计算每个样本的累加，只累加 [assistant_start, T-1] 区间
    # 对每个样本 b:
    #   起始位置 = assistant_starts[b] - 1  (因为我们去掉了 tokens[0])
    #   结束位置 = 最后一个真实 token 位置
    batch_log_probs = []
    for b in range(B):
        start_idx = max(0, assistant_starts[b].item() - 1)
        # 找最后一个 True 的位置
        row_mask = mask[b, 1:]  # (T-1,)
        if row_mask.sum() == 0:
            batch_log_probs.append(torch.tensor(0.0, device=device))
            continue
        # 取 [start_idx, end] 的累加
        valid_log_probs = gathered[b, start_idx:]  # 从 assistant_start 到结尾
        batch_log_probs.append(valid_log_probs.sum())

    return torch.stack(batch_log_probs)


def train(args):
    device = get_device()
    print(f"[INFO] 使用设备: {device}")

    # 1. 初始化模型
    config = LMMConfig()
    if args.n_layer:
        config.n_layer = args.n_layer
    if args.n_embd:
        config.n_embd = args.n_embd
        config.ffn_dim = 4 * args.n_embd

    model = LMMModel(config).to(device)

    # 加载 checkpoint
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        filtered = {k: v for k, v in sd.items()
                    if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"[INFO] 加载 checkpoint: {args.checkpoint}")
        print(f"[INFO] 缺失 {len(missing)} 个键，多余 {len(unexpected)} 个键")
    else:
        print(f"[WARN] 没有提供 checkpoint，随机初始化（仅用于测试 pipeline）")

    # 2. 参考模型（reference model）— 复制一份，冻结权重，用来算 π_ref
    ref_model = LMMModel(config)
    ref_model.load_state_dict(model.state_dict())
    ref_model = ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"[INFO] 参考模型已创建（冻结）")

    # 3. 数据
    dpo_data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                dpo_data.append(json.loads(line))
            except Exception:
                continue
    if len(dpo_data) == 0:
        print("[ERROR] 没有 DPO 数据。请先运行 teacher_inference.py --mode dpo",
              file=sys.stderr)
        sys.exit(1)

    random.seed(42)
    random.shuffle(dpo_data)
    n_val = max(1, int(len(dpo_data) * 0.05))
    val_data = dpo_data[:n_val]
    train_data = dpo_data[n_val:]
    print(f"[INFO] DPO 样本: 训练 {len(train_data)} 条，验证 {len(val_data)} 条")

    dataset = DPODataset(train_data, config.max_seq_len, config.vocab_size)
    val_dataset = DPODataset(val_data, config.max_seq_len, config.vocab_size)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=dpo_collate, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dpo_collate, num_workers=0,
    )

    # 4. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 5. 实验目录
    os.makedirs(os.path.join(args.exp_dir, "checkpoints"), exist_ok=True)

    # 6. 训练
    model.train()
    step = 0
    best_loss = float("inf")
    running_loss = 0.0
    report_steps = 10
    val_interval = max(50, args.steps // 20)
    beta = args.beta

    print(f"\n{'='*60}\n开始 DPO 训练\n总步数: {args.steps}  batch: {args.batch_size}"
          f"  lr: {args.lr}  beta={beta}\n训练样本: {len(train_data)}\n{'='*60}\n")

    while step < args.steps:
        for chosen_ids, rejected_ids, starts, c_mask, r_mask in loader:
            if step >= args.steps:
                break
            chosen_ids = chosen_ids.to(device)
            rejected_ids = rejected_ids.to(device)
            c_mask = c_mask.to(device)
            r_mask = r_mask.to(device)

            # === 前向传播 ===
            # current model 对 chosen / rejected 的 log probs
            log_probs_chosen = compute_log_probs_batch(
                model, chosen_ids, starts, c_mask, device
            )
            log_probs_rejected = compute_log_probs_batch(
                model, rejected_ids, starts, r_mask, device
            )

            # reference model 对 chosen / rejected 的 log probs（无梯度）
            with torch.no_grad():
                ref_log_probs_chosen = compute_log_probs_batch(
                    ref_model, chosen_ids, starts, c_mask, device
                )
                ref_log_probs_rejected = compute_log_probs_batch(
                    ref_model, rejected_ids, starts, r_mask, device
                )

            # === DPO 损失 ===
            # log π_θ(y_c|x) - log π_ref(y_c|x) - (log π_θ(y_r|x) - log π_ref(y_r|x))
            log_ratios = (
                (log_probs_chosen - ref_log_probs_chosen)
                - (log_probs_rejected - ref_log_probs_rejected)
            )
            # -log σ(beta * log_ratios)
            losses = -F.logsigmoid(beta * log_ratios)
            loss = losses.mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            # 报告
            if step % report_steps == 0:
                avg = running_loss / report_steps
                # 算一下正确率（chosen 比 rejected 概率高的比例）
                with torch.no_grad():
                    diff = log_probs_chosen - log_probs_rejected
                    acc = (diff > 0).float().mean().item()
                print(f"  step {step}/{args.steps}  loss={avg:.4f}  "
                      f"acc={acc:.2f}  lr={args.lr:.2e}")
                running_loss = 0.0

            # 验证
            if step % val_interval == 0:
                val_loss, val_acc = evaluate_dpo(
                    model, ref_model, val_loader, device, beta
                )
                print(f"  [VAL] step {step}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_path = os.path.join(args.exp_dir, "checkpoints", "best.pt")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step, "loss": best_loss, "config": config.to_dict(),
                        "beta": beta,
                    }, save_path)
                    print(f"  [SAVE] 保存最佳模型到 {save_path}")
                model.train()

    # 训练结束
    save_path = os.path.join(args.exp_dir, "checkpoints", "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "loss": best_loss, "config": config.to_dict(),
        "beta": beta,
    }, save_path)
    print(f"\n[DONE] DPO 训练完成！最终模型: {save_path}")


@torch.no_grad()
def evaluate_dpo(model, ref_model, loader, device, beta):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for chosen_ids, rejected_ids, starts, c_mask, r_mask in loader:
        chosen_ids = chosen_ids.to(device)
        rejected_ids = rejected_ids.to(device)
        c_mask = c_mask.to(device)
        r_mask = r_mask.to(device)

        log_probs_chosen = compute_log_probs_batch(model, chosen_ids, starts, c_mask, device)
        log_probs_rejected = compute_log_probs_batch(model, rejected_ids, starts, r_mask, device)
        ref_log_probs_chosen = compute_log_probs_batch(ref_model, chosen_ids, starts, c_mask, device)
        ref_log_probs_rejected = compute_log_probs_batch(ref_model, rejected_ids, starts, r_mask, device)

        log_ratios = (
            (log_probs_chosen - ref_log_probs_chosen)
            - (log_probs_rejected - ref_log_probs_rejected)
        )
        losses = -F.logsigmoid(beta * log_ratios)
        total_loss += losses.mean().item()
        total_acc += (log_ratios > 0).float().mean().item()
        n += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="DPO 偏好对齐训练")
    p.add_argument("--data", required=True, help="DPO JSONL 数据路径")
    p.add_argument("--checkpoint", default=None, help="SFT 后模型 checkpoint")
    p.add_argument("--exp-dir", default="experiments/exp_dpo_default",
                   help="实验输出目录")
    p.add_argument("--steps", type=int, default=1000, help="训练步数")
    p.add_argument("--batch-size", type=int, default=4, help="batch size")
    p.add_argument("--lr", type=float, default=1e-5, help="学习率（DPO 通常用较小的 lr）")
    p.add_argument("--beta", type=float, default=0.1,
                   help="温度系数 beta，越大则对偏离参考模型越敏感（推荐 0.01~1.0）")

    p.add_argument("--n-layer", type=int, default=None, help="模型层数")
    p.add_argument("--n-embd", type=int, default=None, help="模型嵌入维度")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
