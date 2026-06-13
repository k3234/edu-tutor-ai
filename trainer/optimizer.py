"""
优化器和学习率调度器

配置 AdamW 优化器 + Cosine Warmup 学习率调度。
这是训练大模型的标准组合，兼顾收敛速度和稳定性。
"""

import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler


def create_optimizer(
    model,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple = (0.9, 0.95),
) -> AdamW:
    """
    创建 AdamW 优化器

    对偏置和 LayerNorm 参数使用不同的 weight_decay 策略：
        - 权重矩阵: weight_decay = 配置值（正则化）
        - 偏置和 LayerNorm: weight_decay = 0（不正则化）

    Args:
        model: 模型对象
        learning_rate: 初始学习率
        weight_decay: 权重衰减系数
        betas: Adam 的 beta1, beta2

    Returns:
        AdamW 优化器
    """
    # 分离参数：哪些需要 weight_decay，哪些不需要
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 偏置和 LayerNorm 参数不需要 weight_decay
        if "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # 参数组
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=learning_rate, betas=betas)

    print(f"优化器配置:")
    print(f"  学习率: {learning_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  衰减参数: {sum(p.numel() for p in decay_params):,}")
    print(f"  不衰减参数: {sum(p.numel() for p in no_decay_params):,}")

    return optimizer


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine 退火 + 线性预热 学习率调度器

    训练初期（warmup 阶段）：学习率从 0 线性增加到目标值
    训练后期：学习率按余弦曲线衰减到最小值

    这有助于：
        - 预热期: 稳定训练初期，避免大学习率导致梯度爆炸
        - 退火期: 精细调整，帮助模型收敛到更好的局部最优

    Args:
        optimizer: PyTorch 优化器
        warmup_steps: 预热步数
        max_steps: 总训练步数
        min_lr_ratio: 最小学习率比例（相对于初始学习率）
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        min_lr_ratio: float = 0.1,
    ):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        """计算当前学习率"""
        # 当前步数（从 optimizer 的 state 中获取）
        step = self._step_count

        if step < self.warmup_steps:
            # 预热阶段: 线性增加
            # lr = base_lr * (step / warmup_steps)
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # 退火阶段: 余弦衰减
            # progress = (step - warmup) / (max_steps - warmup)
            # lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            return [
                base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay)
                for base_lr in self.base_lrs
            ]

    def get_last_lr(self):
        """获取当前学习率（用于日志记录）"""
        return self._last_lr