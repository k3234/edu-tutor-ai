"""
SwiGLUMLP — SwiGLU 激活的前馈网络

SwiGLU 是 Swish (SiLU) 激活 + Gated Linear Unit 的组合，
相比传统 ReLU/GELU + 两层 MLP，训练更稳定，效果更优。

现代大模型（如 LLaMA、Mistral）普遍采用 SwiGLU。
"""

import torch
import torch.nn as nn
from torch import Tensor


class SwiGLUMLP(nn.Module):
    """
    SwiGLU 前馈网络

    结构:
        输入 x
        ├── gate_proj: Linear(n_embd, ffn_dim)  → gate
        ├── up_proj:   Linear(n_embd, ffn_dim)  → up
        ├── gate = SiLU(gate) * up               (SwiGLU 核心)
        └── down_proj: Linear(ffn_dim, n_embd)  → 输出

    参数量: 3 * n_embd * ffn_dim (比传统 MLP 多一个门控投影)

    Attributes:
        gate_proj: 门控投影层 (n_embd -> ffn_dim)
        up_proj:   升维投影层 (n_embd -> ffn_dim)
        down_proj: 降维投影层 (ffn_dim -> n_embd)
        dropout:   Dropout 层
    """

    def __init__(self, n_embd: int, ffn_dim: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()

        # 门控投影: 决定哪些信息通过
        self.gate_proj = nn.Linear(n_embd, ffn_dim, bias=bias)

        # 升维投影: 将信息映射到更高维度
        self.up_proj = nn.Linear(n_embd, ffn_dim, bias=bias)

        # 降维投影: 将信息映射回原始维度
        self.down_proj = nn.Linear(ffn_dim, n_embd, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for proj in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (B, T, C)

        Returns:
            输出张量，形状 (B, T, C)
        """
        # 1. 计算门控和升维分支
        gate = self.gate_proj(x)  # (B, T, ffn_dim)
        up = self.up_proj(x)      # (B, T, ffn_dim)

        # 2. SwiGLU: SiLU(gate) * up
        # SiLU(x) = x * sigmoid(x)，平滑的 ReLU 变体
        hidden = torch.nn.functional.silu(gate) * up  # (B, T, ffn_dim)

        # 3. 降维投影 + Dropout
        output = self.dropout(self.down_proj(hidden))  # (B, T, C)

        return output