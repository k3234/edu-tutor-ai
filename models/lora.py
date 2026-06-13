# -*- coding: utf-8 -*-
"""
LoRA (Low-Rank Adaptation) — 低秩适配模块

核心思路：
    不直接更新大模型权重 W，而是在旁边挂两个小矩阵 A 和 B：
        W_new = W + B * A
    其中 A ∈ R(r, d_in)，B ∈ R(d_out, r)，r << min(d_in, d_out)
    只训练 A 和 B，冻结 W — 极大减少显存占用和训练时间。

本实现：
    1. LoraLinear — 单一层的 LoRA 替换
    2. LoraModel  — 自动把 LMMModel 里的线性层替换为 LoRA 版

使用：
    from models.lora import LoraModel
    model = LMMModel(config)
    # 先加载预训练权重
    model.load_state_dict(...)
    # 注入 LoRA
    lora_model = LoraModel(model, rank=8, alpha=16, target_modules=["attn", "mlp"])
    # 只训练 LoRA 参数（自动冻结其他）
    optimizer = AdamW(lora_model.lora_parameters(), lr=3e-4)
"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional


class LoraLinear(nn.Module):
    """
    带 LoRA 的线性层：
        y = x @ W.T + (1/alpha) * (x @ A.T) @ B.T
    其中 W 冻结，A/B 可训练。

    Args:
        in_features:  输入维度 d_in
        out_features: 输出维度 d_out
        rank:         LoRA 秩 r（越小越少参数）
        alpha:        缩放因子，一般设为 2*rank 或 rank
        bias:         是否启用偏置（偏置冻结，不上 LoRA）
        dropout:      对 LoRA 路径的 dropout（可选）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank  # 缩放系数

        # 主干权重 W（冻结）—— 在外面会把预训练权重拷进来
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA 小矩阵 A / B
        # A 用 Kaiming 初始化，B 初始化为 0（保证开始时等同于原模型）
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 可选的 dropout
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # 默认冻结 W 和 bias — 外面调用 freeze_base() 时也会确认
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """
        y = x @ W.T + scaling * dropout(x @ A.T) @ B.T
        """
        # 主干（冻结路径）
        base_out = x @ self.weight.T
        if self.bias is not None:
            base_out = base_out + self.bias

        # LoRA 路径（可训练路径）
        lora_out = self.lora_dropout(x) @ self.lora_A.T  # (..., rank)
        lora_out = lora_out @ self.lora_B.T               # (..., out_features)

        return base_out + self.scaling * lora_out


class LoraModel(nn.Module):
    """
    把 LMMModel 中指定的线性层替换为 LoraLinear。

    典型用法：
        model = LMMModel(config)
        model.load_state_dict(torch.load("pretrained.pt")["model_state_dict"])
        lora_model = LoraModel(model, rank=8, alpha=16, target_modules=["c_attn", "c_proj", "fc_1", "fc_2"])

        optimizer = torch.optim.AdamW(lora_model.lora_parameters(), lr=3e-4)

    也可以在 SFT 训练脚本中调用 lora_model.merge_lora() 把 LoRA 合并回 W，
    之后就可以像正常模型一样推理/保存了。
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        alpha: int = 16,
        target_modules: Optional[List[str]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha

        # 默认要注入 LoRA 的模块名关键字
        # 适配 LMM 里的 c_attn / c_proj / fc_1 / fc_2 / lm_head / wte 等
        if target_modules is None:
            target_modules = ["c_attn", "c_proj", "fc_1", "fc_2"]
        self.target_modules = target_modules

        # 递归替换
        self._replace_modules(self.model, parent_name="")

        # 统计参数
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.lora_parameters())
        print(f"[LoRA] 总参数量: {total:,}")
        print(f"[LoRA] 可训练参数量: {trainable:,} ({100 * trainable / total:.2f}%)")

    def _replace_modules(self, module: nn.Module, parent_name: str):
        """递归遍历并替换满足条件的 nn.Linear 为 LoraLinear"""
        for name, child in list(module.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name

            # 先递归进子模块
            self._replace_modules(child, full_name)

            # 如果是 nn.Linear 且名字匹配关键字，就替换
            if isinstance(child, nn.Linear) and any(k in name for k in self.target_modules):
                in_features = child.in_features
                out_features = child.out_features
                has_bias = child.bias is not None

                # 创建 LoraLinear 并把 W / bias 拷过去
                lora = LoraLinear(
                    in_features=in_features,
                    out_features=out_features,
                    rank=self.rank,
                    alpha=self.alpha,
                    bias=has_bias,
                    dropout=0.0,
                )
                with torch.no_grad():
                    lora.weight.copy_(child.weight)
                    if has_bias:
                        lora.bias.copy_(child.bias)

                setattr(module, name, lora)

    def lora_parameters(self) -> List[nn.Parameter]:
        """返回所有可训练（LoRA）参数"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def freeze_base(self):
        """显式冻结非 LoRA 参数（一般调用 _replace_modules 后已经冻结，但双保险）"""
        for p in self.model.parameters():
            p.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, LoraLinear):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None):
        """直接代理到原模型的 forward"""
        return self.model(idx, targets)

    @torch.no_grad()
    def merge_lora(self):
        """
        把 LoRA 路径合并回 W，返回普通权重的模型（用于推理/保存）。
        合并后 W_new = W + scaling * B @ A
        之后模型不再有 LoraLinear 子模块。
        """
        for module in self.model.modules():
            if isinstance(module, LoraLinear):
                delta = module.scaling * (module.lora_B @ module.lora_A)
                module.weight.add_(delta)

    def state_dict_for_save(self):
        """只保存 LoRA 参数（文件非常小，几 MB 级别）"""
        return {
            "lora_state_dict": {
                k: v for k, v in self.model.state_dict().items()
                if "lora_A" in k or "lora_B" in k
            },
            "rank": self.rank,
            "alpha": self.alpha,
        }
