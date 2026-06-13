"""
训练日志工具

记录训练过程中的关键指标，支持:
    - 控制台输出（带进度条）
    - 日志文件保存
    - 训练指标统计（loss, perplexity, learning rate, 显存使用等）
"""

import os
import time
import logging
from datetime import datetime


class TrainingLogger:
    """
    训练日志记录器

    统一管理训练过程中的日志输出和指标记录。

    Attributes:
        log_dir: 日志保存目录
        log_file: 日志文件路径
        logger: Python logging 实例
        start_time: 训练开始时间
    """

    def __init__(self, log_dir: str = "experiments/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 日志文件命名: train_YYYYMMDD_HHMMSS.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"train_{timestamp}.log")

        # 配置 logging
        self.logger = logging.getLogger("LMMTraining")
        self.logger.setLevel(logging.INFO)

        # 避免重复添加 handler
        if not self.logger.handlers:
            # 文件 handler
            file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
            file_handler.setLevel(logging.INFO)

            # 控制台 handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式化
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        self.start_time = time.time()
        self.step_times = []  # 记录每步耗时

        self.info(f"训练日志已初始化: {self.log_file}")

    def info(self, message: str):
        """记录 INFO 级别日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录 WARNING 级别日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录 ERROR 级别日志"""
        self.logger.error(message)

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float = None,
        tokens_per_sec: float = None,
    ):
        """
        记录单步训练指标

        Args:
            step: 当前步数
            loss: 训练损失
            lr: 当前学习率
            grad_norm: 梯度范数（可选）
            tokens_per_sec: 每秒处理 token 数（可选）
        """
        # 计算 perplexity
        perplexity = self._compute_perplexity(loss)

        msg = f"Step {step:>6} | Loss: {loss:.4f} | PPL: {perplexity:.2f} | LR: {lr:.2e}"

        if grad_norm is not None:
            msg += f" | Grad: {grad_norm:.4f}"
        if tokens_per_sec is not None:
            msg += f" | Tok/s: {tokens_per_sec:.0f}"

        self.info(msg)

    def log_validation(self, step: int, val_loss: float):
        """
        记录验证指标

        Args:
            step: 当前步数
            val_loss: 验证集损失
        """
        perplexity = self._compute_perplexity(val_loss)
        self.info(f"Validation | Step {step:>6} | Val Loss: {val_loss:.4f} | Val PPL: {perplexity:.2f}")

    def log_gpu_memory(self):
        """记录 GPU 显存使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                self.info(f"GPU Memory | Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB")
        except Exception:
            pass

    def log_checkpoint(self, step: int, checkpoint_path: str):
        """记录 checkpoint 保存信息"""
        self.info(f"Checkpoint saved at step {step}: {checkpoint_path}")

    def log_training_summary(self, total_steps: int, best_loss: float):
        """
        记录训练总结

        Args:
            total_steps: 总训练步数
            best_loss: 最佳损失
        """
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)

        self.info("=" * 60)
        self.info("训练完成!")
        self.info(f"  总步数: {total_steps}")
        self.info(f"  最佳损失: {best_loss:.4f}")
        self.info(f"  总耗时: {hours}h {minutes}m")
        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            self.info(f"  平均每步耗时: {avg_time:.3f}s")
        self.info("=" * 60)

    def _compute_perplexity(self, loss: float) -> float:
        """计算困惑度 (Perplexity = exp(loss))"""
        import math
        return math.exp(loss) if loss < 20 else float("inf")  # 防止 overflow

    def get_elapsed_time(self) -> float:
        """获取已训练时间（秒）"""
        return time.time() - self.start_time

    def estimate_remaining_time(self, current_step: int, total_steps: int) -> str:
        """
        估算剩余训练时间

        Args:
            current_step: 当前步数
            total_steps: 总步数

        Returns:
            格式化的时间字符串
        """
        if current_step == 0 or not self.step_times:
            return "未知"

        avg_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = total_steps - current_step
        remaining_seconds = avg_time * remaining_steps

        hours = int(remaining_seconds // 3600)
        minutes = int((remaining_seconds % 3600) // 60)

        return f"{hours}h {minutes}m"