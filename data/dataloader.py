"""
DataLoader 工厂函数

创建训练/验证用的 PyTorch DataLoader，支持:
    - 多进程数据加载
    - 自动 batch 拼接
    - 固定内存（pin_memory）加速 GPU 传输
"""

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset import LMMDataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    创建 DataLoader

    Args:
        dataset: 数据集对象
        batch_size: 每批样本数
        shuffle: 是否打乱数据（训练集 True，验证集 False）
        num_workers: 数据加载进程数（0=主进程加载）
        pin_memory: 是否将数据固定在锁页内存（加速 GPU 传输）

    Returns:
        PyTorch DataLoader
    """
    # 自动检测是否有 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = pin_memory and (device == "cuda")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 丢弃不完整的最后一批，避免梯度累积问题
    )

    return dataloader


def create_train_val_loaders(
    train_dataset: LMMDataset,
    val_dataset: LMMDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple:
    """
    同时创建训练集和验证集的 DataLoader

    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        batch_size: 批大小
        num_workers: 加载进程数

    Returns:
        (train_loader, val_loader)
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集需要打乱
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不需要打乱
        num_workers=num_workers,
    )

    return train_loader, val_loader