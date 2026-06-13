"""
LMMBlock — Transformer 解码器块

标准的 Pre-LayerNorm Transformer Block，包含:
    1. LayerNorm -> CausalSelfAttention -> 残差连接
    2. LayerNorm -> SwiGLUMLP -> 残差连接

Pre-LayerNorm 相比 Post-LayerNorm 训练更稳定，
是现代大模型（GPT-3, LLaMA, DeepSeek 等）的标准选择。
"""

import torch.nn as nn
from torch import Tensor

from .attention import CausalSelfAttention
from .mlp import SwiGLUMLP


class LMMBlock(nn.Module):
    """
    单个 Transformer 解码器块

    前向流程:
        x ──→ LayerNorm ──→ Attention ──→ + ──→ LayerNorm ──→ MLP ──→ + ──→ 输出
        │                                    ↑                  ↑
        └────────────────────────────────────┘                  └─────┘
                          (残差连接)                          (残差连接)

    Attributes:
        ln_1: 注意力前的 LayerNorm
        attn: 因果自注意力层
        ln_2: MLP 前的 LayerNorm
        mlp: SwiGLU 前馈网络
    """

    def __init__(self, n_embd: int, n_head: int, ffn_dim: int, max_seq_len: int = 1024,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()

        # 第一层: LayerNorm + Attention + 残差
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, max_seq_len, dropout, bias)

        # 第二层: LayerNorm + MLP + 残差
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = SwiGLUMLP(n_embd, ffn_dim, dropout, bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (B, T, C)

        Returns:
            输出张量，形状 (B, T, C)
        """
        # 第一层: Attention + 残差
        # Pre-LayerNorm: 先归一化，再计算注意力
        x = x + self.attn(self.ln_1(x))

        # 第二层: MLP + 残差
        x = x + self.mlp(self.ln_2(x))

        return x