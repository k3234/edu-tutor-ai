"""
训练主脚本

LMM 模型的预训练入口。完整流程:
    1. 加载配置
    2. 初始化模型
    3. 准备数据
    4. 配置优化器和调度器
    5. 训练循环（含验证和 checkpoint）

使用:
    python scripts/train.py --config configs/lmm_small.yaml
    python scripts/train.py --resume experiments/exp_xxx/checkpoints/latest.pt
"""

import argparse
import os
import sys
import time
import math

import torch
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import LMMConfig, LMMModel
from data.dataset import prepare_dataset
from data.dataloader import create_train_val_loaders
from trainer import create_optimizer, CosineWarmupScheduler
from trainer import save_checkpoint, load_checkpoint, get_latest_checkpoint
from trainer import TrainingLogger


def get_device(device: str = None) -> str:
    """获取可用的计算设备"""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def validate(model, val_loader, device, max_batches: int = 10):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证集 DataLoader
        device: 计算设备
        max_batches: 最多评估的 batch 数（加快验证速度）

    Returns:
        平均验证损失
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, (input_ids, target_ids) in enumerate(val_loader):
            if i >= max_batches:
                break

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            _, loss = model(input_ids, target_ids)
            total_loss += loss.item()
            n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(config_path: str, resume_path: str = None,
          steps: int = None, device_name: str = None):
    """
    主训练函数

    Args:
        config_path: 配置文件路径
        resume_path: 恢复训练的 checkpoint 路径（可选）
        steps: 覆盖配置中的最大训练步数（可选）
        device_name: 覆盖自动检测的计算设备（可选）
    """
    # ========== 1. 加载配置 ==========
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = LMMConfig(**cfg["model"])
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # --steps 覆盖配置里的 max_steps
    if steps:
        train_cfg["max_steps"] = steps

    # 创建实验目录
    exp_dir = train_cfg.get("exp_dir", "experiments/exp_default")
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ========== 2. 初始化设备 ==========
    device = get_device(device_name)
    print(f"使用设备: {device}")

    # 设置随机种子（保证可复现）
    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # ========== 3. 初始化模型 ==========
    print("\n" + "=" * 60)
    print("初始化模型")
    print("=" * 60)
    model = LMMModel(model_cfg).to(device)

    # 混合精度训练
    use_amp = train_cfg.get("use_amp", True) and (device == "cuda")
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("启用混合精度训练 (AMP)")

    # ========== 4. 准备数据 ==========
    print("\n" + "=" * 60)
    print("准备数据")
    print("=" * 60)

    token_file = data_cfg.get("token_file", "data/tokens.pt")
    if not os.path.exists(token_file):
        print(f"错误: Token 文件不存在: {token_file}")
        print("请先运行: python scripts/train_tokenizer.py")
        print("然后对文本进行分词并保存为 .pt 文件")
        return

    train_dataset, val_dataset = prepare_dataset(
        token_file=token_file,
        seq_len=model_cfg.max_seq_len,
        train_ratio=data_cfg.get("train_ratio", 0.95),
    )

    train_loader, val_loader = create_train_val_loaders(
        train_dataset,
        val_dataset,
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ========== 5. 优化器和调度器 ==========
    print("\n" + "=" * 60)
    print("配置优化器")
    print("=" * 60)
    optimizer = create_optimizer(
        model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.1),
    )

    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.1),
    )

    # ========== 6. 恢复训练（如果有 checkpoint） ==========
    start_step = 0
    best_val_loss = float("inf")

    if resume_path:
        meta = load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_step = meta["step"]
        best_val_loss = meta["loss"]
    elif os.path.exists(checkpoint_dir):
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            print(f"\n发现已有 checkpoint，是否恢复? (y/n)")
            # 自动恢复（生产环境可改为交互式）
            meta = load_checkpoint(latest, model, optimizer, scheduler, device)
            start_step = meta["step"]
            best_val_loss = meta["loss"]

    # ========== 7. 初始化日志 ==========
    logger = TrainingLogger(log_dir)
    logger.info(f"实验目录: {exp_dir}")
    logger.info(f"配置: {cfg}")

    # ========== 8. 训练循环 ==========
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    model.train()
    train_iter = iter(train_loader)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    val_interval = train_cfg.get("val_interval", 500)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 2000)
    max_steps = train_cfg["max_steps"]

    for step in range(start_step, max_steps):
        step_start = time.time()

        # 获取下一个 batch（支持循环迭代）
        try:
            input_ids, target_ids = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, target_ids = next(train_iter)

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # 前向传播（混合精度）
        if use_amp:
            with autocast():
                _, loss = model(input_ids, target_ids)
                loss = loss / grad_accum_steps  # 梯度累积缩放
        else:
            _, loss = model(input_ids, target_ids)
            loss = loss / grad_accum_steps

        # 反向传播
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 梯度累积: 每 grad_accum_steps 步更新一次
        if (step + 1) % grad_accum_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # 记录日志
        if step % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            logger.log_step(step, loss.item() * grad_accum_steps, lr)

        # 验证
        if step > 0 and step % val_interval == 0:
            val_loss = validate(model, val_loader, device)
            logger.log_validation(step, val_loss)
            logger.log_gpu_memory()

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, step, val_loss,
                    checkpoint_dir, is_best=True,
                )

            model.train()

        # 保存 Checkpoint
        if step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, step, loss.item() * grad_accum_steps,
                checkpoint_dir,
            )

        # 记录步耗时
        step_time = time.time() - step_start
        logger.step_times.append(step_time)

    # ========== 9. 训练结束 ==========
    logger.log_training_summary(max_steps, best_val_loss)

    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, scheduler, max_steps, best_val_loss,
                    checkpoint_dir, filename="final.pt")
    print(f"\n训练完成! 最终模型保存于: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="LMM 模型训练")
    parser.add_argument("--config", type=str, default="configs/lmm_small.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    parser.add_argument("--device", type=str, default=None, help="计算设备 (cuda/cpu/mps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="最大训练步数（覆盖配置文件中的 max_steps）")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return

    train(args.config, args.resume, steps=args.steps, device_name=args.device)


if __name__ == "__main__":
    main()