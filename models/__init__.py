"""
LMM 模型模块

包含从零实现的 GPT-style Transformer 语言模型。
所有组件均为 PyTorch nn.Module 子类，可直接用于训练和推理。

模块结构:
    config.py    -> LMMConfig 配置类
    attention.py -> CausalSelfAttention 因果自注意力
    mlp.py       -> SwiGLUMLP 前馈网络
    block.py     -> LMMBlock Transformer块
    model.py     -> LMMModel 完整模型

使用示例:
    from models import LMMConfig, LMMModel
    config = LMMConfig()
    model = LMMModel(config)
"""

from .config import LMMConfig
from .model import LMMModel

__all__ = ["LMMConfig", "LMMModel"]