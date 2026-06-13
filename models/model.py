"""
LMMModel — 完整的 GPT-style 语言模型

从零实现的 Transformer Decoder-only 语言模型。
包含词嵌入、位置编码、多层 Transformer Block、最终归一化和语言建模头。

支持:
    - 训练模式: forward(idx, targets) 返回 loss
    - 推理模式: forward(idx) 返回 logits
    - 文本生成: generate(idx, max_new_tokens) 自回归生成

使用示例:
    config = LMMConfig()
    model = LMMModel(config)
    logits, loss = model(input_ids, target_ids)  # 训练
    generated = model.generate(prompt_ids, max_new_tokens=100)  # 推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple

from .config import LMMConfig
from .block import LMMBlock


class LMMModel(nn.Module):
    """
    LMM 语言模型

    结构:
        Token Embedding -> Positional Embedding
            -> [LMMBlock × n_layer]
            -> LayerNorm -> LMHead -> Logits

    Attributes:
        config: LMMConfig 配置对象
        wte: Token 嵌入层 (vocab_size -> n_embd)
        wpe: 位置编码层 (max_seq_len -> n_embd)
        drop: 嵌入 Dropout
        blocks: Transformer Block 列表
        ln_f: 最终 LayerNorm
        lm_head: 语言建模头 (n_embd -> vocab_size)
    """

    def __init__(self, config: LMMConfig):
        super().__init__()
        self.config = config

        # 词嵌入层: 将 token ID 映射为向量
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 位置编码: 可学习的位置嵌入
        self.wpe = nn.Embedding(config.max_seq_len, config.n_embd)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks (堆叠 n_layer 层)
        self.blocks = nn.ModuleList([
            LMMBlock(
                n_embd=config.n_embd,
                n_head=config.n_head,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                bias=config.bias,
            )
            for _ in range(config.n_layer)
        ])

        # 最终 LayerNorm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 语言建模头: 将隐藏状态映射回词表空间
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重共享: 词嵌入和输出层共享权重（减少参数量，提升效果）
        if config.tie_weights:
            self.wte.weight = self.lm_head.weight

        # 初始化所有权重
        self.apply(self._init_weights)

        # 特殊初始化: 残差投影层使用更小的 std
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

        # 打印模型信息
        print(f"模型初始化完成: {config}")
        print(f"总参数量: {self.get_num_params():,} ({self.get_num_params()/1e6:.2f}M)")

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        计算模型参数量

        Args:
            non_embedding: 是否排除位置编码的参数量

        Returns:
            总参数量
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # 减去位置编码的参数量（位置编码不是"学习到的知识"）
            n_params -= self.wpe.weight.numel()
        return n_params

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        前向传播

        Args:
            idx: 输入 token IDs，形状 (B, T)
            targets: 目标 token IDs（训练时使用），形状 (B, T)

        Returns:
            logits: 预测分数，形状 (B, T, vocab_size)
            loss:   交叉熵损失（targets 为 None 时返回 None）
        """
        device = idx.device
        B, T = idx.size()

        # 检查序列长度
        assert T <= self.config.max_seq_len, (
            f"序列长度 {T} 超过最大长度 {self.config.max_seq_len}"
        )

        # 1. Token 嵌入
        tok_emb = self.wte(idx)  # (B, T, n_embd)

        # 2. 位置编码
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.wpe(pos)  # (1, T, n_embd)

        # 3. 嵌入相加 + Dropout
        x = self.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        # 4. 通过所有 Transformer Blocks
        for block in self.blocks:
            x = block(x)  # (B, T, n_embd)

        # 5. 最终归一化
        x = self.ln_f(x)  # (B, T, n_embd)

        # 6. 语言建模头 -> 词表空间
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 7. 计算损失（训练模式）
        loss = None
        if targets is not None:
            # 展平为 (B*T, vocab_size) 和 (B*T)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,  # 忽略 padding
            )

        return logits, loss

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        """
        自回归文本生成

        每次生成一个新 token，将其追加到序列末尾，继续生成下一个。
        这是 GPT 等自回归模型的标准生成方式。

        Args:
            idx: 初始 prompt token IDs，形状 (B, T)
            max_new_tokens: 最多生成多少个新 token
            temperature: 采样温度（越高越随机，越低越确定）
            top_k: 只从概率最高的 k 个 token 中采样（None=不限制）

        Returns:
            生成的完整序列，形状 (B, T + max_new_tokens)
        """
        self.eval()  # 切换到评估模式

        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]

            # 前向传播获取 logits
            with torch.no_grad():
                logits, _ = self(idx_cond)

            # 取最后一个位置的 logits（预测下一个 token）
            logits = logits[:, -1, :]  # (B, vocab_size)

            # 温度缩放
            logits = logits / temperature

            # Top-k 采样（可选）
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # 采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 追加到序列
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        估算模型浮点运算利用率 (Model FLOPs Utilization)

        用于评估 GPU 利用率，参考值:
            - A100: ~50-60% 为优秀
            - RTX 4090: ~30-40% 为优秀

        Args:
            fwdbwd_per_iter: 每次迭代的浮点运算次数
            dt: 每次迭代耗时（秒）

        Returns:
            MFU 百分比
        """
        # A100 bfloat16 峰值: 312 TFLOPS
        # RTX 4090: 82.6 TFLOPS
        N = self.get_num_params()
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.max_seq_len,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # 假设 A100
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 bfloat16 peak
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate_text(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        device: str = "cuda",
    ) -> str:
        """
        便捷的文本生成接口（自动处理 tokenizer）

        Args:
            tokenizer: 分词器对象（需有 encode/decode 方法）
            prompt: 输入提示文本
            max_new_tokens: 最多生成 token 数
            temperature: 采样温度
            top_k: Top-k 采样
            device: 计算设备

        Returns:
            生成的文本字符串
        """
        # 编码 prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 生成
        output_ids = self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        # 解码
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text