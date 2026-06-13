"""
LMMConfig — 模型配置类

定义 LMM-Small 模型的所有超参数。
使用 Python dataclass 实现，支持字典转换和序列化。
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LMMConfig:
    """
    LMM-Small 模型配置

    模型规格: ~120M 参数，GPT-style Decoder-only
    可在 8GB VRAM 消费级 GPU 上训练

    Attributes:
        vocab_size: 词表大小（默认 32000，覆盖中文教育领域）
        n_layer: Transformer 层数（默认 12）
        n_head: 注意力头数（默认 12）
        n_embd: 嵌入维度（默认 768）
        ffn_dim: 前馈网络中间维度（默认 3072 = 4 * n_embd）
        max_seq_len: 最大序列长度（默认 1024）
        dropout: Dropout 概率（默认 0.1，仅训练时使用）
        bias: 线性层是否使用偏置（默认 True）
        tie_weights: 是否共享嵌入和输出层权重（默认 True，减少参数量）
    """

    # 词表与嵌入
    vocab_size: int = 32000
    n_embd: int = 768

    # Transformer 结构
    n_layer: int = 12
    n_head: int = 12
    ffn_dim: int = 3072  # 4 * n_embd，SwiGLU 标准配置

    # 序列与正则化
    max_seq_len: int = 1024
    dropout: float = 0.1

    # 线性层配置
    bias: bool = True
    tie_weights: bool = True

    # 初始化
    def __post_init__(self):
        """验证配置一致性"""
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) 必须能被 n_head ({self.n_head}) 整除"
        )
        self.head_dim = self.n_embd // self.n_head  # 每个头的维度

    def to_dict(self) -> dict:
        """转换为字典，便于保存为 JSON/YAML"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LMMConfig":
        """从字典恢复配置"""
        return cls(**config_dict)

    def __repr__(self) -> str:
        """打印模型规模估算"""
        # 粗略参数量估算
        emb_params = self.vocab_size * self.n_embd  # 词嵌入
        pos_params = self.max_seq_len * self.n_embd  # 位置编码
        attn_params = self.n_layer * (
            3 * self.n_embd * self.n_embd + self.n_embd * self.n_embd
        )  # QKV + 输出投影
        ffn_params = self.n_layer * (
            self.n_embd * self.ffn_dim * 2 + self.ffn_dim * self.n_embd
        )  # SwiGLU 有3个矩阵
        ln_params = self.n_layer * 2 * self.n_embd + self.n_embd  # LayerNorm
        total = emb_params + pos_params + attn_params + ffn_params + ln_params
        total_m = total / 1e6
        return (
            f"LMMConfig(vocab={self.vocab_size}, layers={self.n_layer}, "
            f"heads={self.n_head}, embd={self.n_embd}, seq_len={self.max_seq_len}, "
            f"~{total_m:.1f}M params)"
        )


# 预定义配置（方便快速使用）
def get_lmm_small_config() -> LMMConfig:
    """返回 LMM-Small 标准配置（~120M 参数）"""
    return LMMConfig()


def get_lmm_tiny_config() -> LMMConfig:
    """返回 LMM-Tiny 轻量配置（~30M 参数，用于快速实验）"""
    return LMMConfig(
        vocab_size=32000,
        n_layer=6,
        n_head=6,
        n_embd=384,
        ffn_dim=1536,
        max_seq_len=512,
    )