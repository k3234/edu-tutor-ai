"""
CausalSelfAttention — 因果自注意力机制

实现带因果掩码的多头自注意力（Masked Multi-Head Self-Attention）。
因果掩码确保模型只能看到当前位置及之前的 token，不能"偷看"未来信息。

这是 GPT 等自回归语言模型的核心组件。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    """
    因果自注意力层

    计算过程:
        1. 输入 x 通过线性层投影为 Q, K, V
        2. 将 Q, K, V 分割为多个头 (multi-head)
        3. 计算注意力分数: attention = softmax(Q·K^T / sqrt(d_k))
        4. 应用因果掩码（上三角矩阵，防止看到未来 token）
        5. 注意力加权求和: output = attention · V
        6. 多头结果拼接后通过输出投影层

    Attributes:
        n_head: 注意力头数
        n_embd: 嵌入维度
        head_dim: 每个头的维度 (n_embd // n_head)
        attn_dropout: 注意力 dropout
        resid_dropout: 残差连接 dropout
    """

    def __init__(self, n_embd: int, n_head: int, max_seq_len: int = 1024,
                 dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd 必须能被 n_head 整除"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.max_seq_len = max_seq_len

        # Q, K, V 的联合投影层: [n_embd -> 3 * n_embd]
        # 一次性计算 Q/K/V，效率更高
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # 输出投影层: [n_embd -> n_embd]
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 因果掩码缓存（注册为 buffer，不计算梯度）
        # 动态适配 max_seq_len，不再硬编码 1024
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(1, 1, max_seq_len, max_seq_len)),  # 下三角矩阵
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重 — 使用 GPT 标准初始化"""
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=0.02)
        if self.c_attn.bias is not None:
            nn.init.zeros_(self.c_attn.bias)
        if self.c_proj.bias is not None:
            nn.init.zeros_(self.c_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (B, T, C)
                B = batch_size, T = 序列长度, C = n_embd

        Returns:
            输出张量，形状 (B, T, C)
        """
        B, T, C = x.size()

        # 1. 计算 Q, K, V 联合投影
        # qkv 形状: (B, T, 3 * C)
        qkv = self.c_attn(x)

        # 2. 分割为 Q, K, V，并 reshape 为多头的形状
        # 目标形状: (B, n_head, T, head_dim)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, n_head, T, head_dim)

        # 3. 计算注意力分数
        # attention_scores = Q · K^T / sqrt(head_dim)
        # 形状: (B, n_head, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. 应用因果掩码（关键步骤！）
        # 将未来位置的注意力分数设为 -inf，softmax 后变为 0
        causal_mask = self.bias_mask[:, :, :T, :T]
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        # 5. Softmax 归一化 + Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 6. 注意力加权求和
        # 形状: (B, n_head, T, head_dim)
        y = attn_weights @ v

        # 7. 多头结果拼接回原始维度
        # 先 transpose 回 (B, T, n_head, head_dim)，再 reshape 为 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. 输出投影 + Dropout
        y = self.resid_dropout(self.c_proj(y))

        return y