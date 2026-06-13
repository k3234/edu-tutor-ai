"""
Checkpoint 管理

保存和加载训练状态，支持:
    - 模型权重
    - 优化器状态
    - 学习率调度器状态
    - 训练步数和其他元信息

支持断点续训：加载 checkpoint 后可从上次中断处继续训练。
"""

import os
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    loss: float,
    save_dir: str,
    is_best: bool = False,
):
    """
    保存训练 checkpoint

    Args:
        model: 模型对象
        optimizer: 优化器
        scheduler: 学习率调度器
        step: 当前训练步数
        loss: 当前损失值
        save_dir: 保存目录
        is_best: 是否为最佳模型（额外保存一份 best.pt）
    """
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    }

    # 保存为 step_{step}.pt
    checkpoint_path = os.path.join(save_dir, f"step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)

    # 同时保存最新 checkpoint（覆盖）
    latest_path = os.path.join(save_dir, "latest.pt")
    torch.save(checkpoint, latest_path)

    # 如果是最佳模型，额外保存
    if is_best:
        best_path = os.path.join(save_dir, "best.pt")
        torch.save(checkpoint, best_path)

    print(f"Checkpoint 已保存: {checkpoint_path} (step={step}, loss={loss:.4f})")

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    device: str = "cuda",
):
    """
    加载训练 checkpoint

    Args:
        checkpoint_path: checkpoint 文件路径
        model: 模型对象（会被原地修改）
        optimizer: 优化器（可选，为 None 则不加载）
        scheduler: 学习率调度器（可选，为 None 则不加载）
        device: 加载到的设备

    Returns:
        dict: 包含 step, loss 等元信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {checkpoint_path}")

    # 加载到指定设备
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器状态
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 加载调度器状态
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", float("inf"))

    print(f"Checkpoint 已加载: {checkpoint_path}")
    print(f"  恢复步数: {step}")
    print(f"  恢复损失: {loss:.4f}")

    return {"step": step, "loss": loss}


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    列出目录中的所有 checkpoint 文件

    Args:
        checkpoint_dir: checkpoint 目录

    Returns:
        按步数排序的 checkpoint 文件列表
    """
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("step_") and f.endswith(".pt"):
            # 提取步数
            step = int(f.replace("step_", "").replace(".pt", ""))
            checkpoints.append((step, os.path.join(checkpoint_dir, f)))

    # 按步数排序
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """
    获取最新的 checkpoint 路径

    Args:
        checkpoint_dir: checkpoint 目录

    Returns:
        最新 checkpoint 文件路径，如果没有则返回 None
    """
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path

    # 如果没有 latest.pt，找步数最大的
    checkpoints = list_checkpoints(checkpoint_dir)
    if checkpoints:
        return checkpoints[-1][1]

    return None