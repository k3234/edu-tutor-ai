"""
LMMDataset — 语言模型数据集

将分词后的 token 序列切分为固定长度的训练样本。
使用滑动窗口方式生成输入-目标对，用于自回归语言建模。

训练样本格式:
    输入:  [t0, t1, t2, ..., t(n-1)]
    目标:  [t1, t2, t3, ..., tn]
    即目标序列是输入序列右移一位，模型学习预测下一个 token。
"""

import torch
from torch.utils.data import Dataset


class LMMDataset(Dataset):
    """
    语言模型数据集

    将长 token 序列切分为 (seq_len) 长度的训练样本。
    每个样本包含输入序列和右移一位的目标序列。

    Args:
        tokens: 完整的 token ID 列表或张量
        seq_len: 每个样本的序列长度（默认 1024）

    Example:
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        dataset = LMMDataset(tokens, seq_len=4)
        # dataset[0] -> (input=[1,2,3,4], target=[2,3,4,5])
        # dataset[1] -> (input=[5,6,7,8], target=[6,7,8,9])  # 9=padding 或下一个token
    """

    def __init__(self, tokens, seq_len: int = 1024):
        super().__init__()

        # 转换为张量
        if isinstance(tokens, list):
            self.tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            self.tokens = tokens

        self.seq_len = seq_len

        # 计算样本数量
        # 总 token 数 // seq_len，向下取整
        self.n_samples = max(0, len(self.tokens) // seq_len)

        if self.n_samples == 0:
            raise ValueError(
                f"token 数量 ({len(self.tokens)}) 不足以生成一个 seq_len={seq_len} 的样本"
            )

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.n_samples

    def __getitem__(self, idx: int):
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (input_ids, target_ids): 两个形状为 (seq_len,) 的张量
        """
        # 计算起始位置
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # 多取一个 token 作为目标

        # 截取 chunk
        chunk = self.tokens[start:end]

        # 如果长度不足，用 padding token (0) 填充
        if len(chunk) < self.seq_len + 1:
            padding = torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])

        # 输入: 前 seq_len 个 token
        input_ids = chunk[:self.seq_len]

        # 目标: 后 seq_len 个 token（右移一位）
        target_ids = chunk[1:self.seq_len + 1]

        # 将 padding 位置的 target 设为 -100
        # cross_entropy 的 ignore_index=-100 会跳过这些位置
        target_ids = target_ids.clone()
        target_ids[target_ids == 0] = -100

        return input_ids, target_ids

    def get_vocab_coverage(self) -> dict:
        """
        统计词表覆盖情况

        Returns:
            dict: 包含 token 种类数、总 token 数等信息
        """
        unique_tokens = torch.unique(self.tokens)
        return {
            "total_tokens": len(self.tokens),
            "unique_tokens": len(unique_tokens),
            "vocab_coverage": len(unique_tokens) / (self.tokens.max().item() + 1),
        }


def prepare_dataset(
    token_file: str,
    seq_len: int = 1024,
    train_ratio: float = 0.95,
) -> tuple:
    """
    从 token 文件准备训练集和验证集

    Args:
        token_file: token 序列保存的文件路径 (.pt 或 .npy)
        seq_len: 序列长度
        train_ratio: 训练集比例

    Returns:
        (train_dataset, val_dataset)
    """
    # 加载 token
    if token_file.endswith('.pt'):
        tokens = torch.load(token_file)
    elif token_file.endswith('.npy'):
        import numpy as np
        tokens = torch.from_numpy(np.load(token_file))
    else:
        raise ValueError(f"不支持的文件格式: {token_file}")

    # 分割训练/验证
    n_train = int(len(tokens) * train_ratio)
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]

    train_dataset = LMMDataset(train_tokens, seq_len)
    val_dataset = LMMDataset(val_tokens, seq_len)

    print(f"数据集准备完成:")
    print(f"  训练集: {len(train_dataset)} 样本, {len(train_tokens):,} tokens")
    print(f"  验证集: {len(val_dataset)} 样本, {len(val_tokens):,} tokens")

    return train_dataset, val_dataset