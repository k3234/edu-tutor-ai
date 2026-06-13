#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — 端到端多教师小模型训练流水线

功能：
    1) 语料收集清洗
    2) 教师模型生成 SFT/DPO 数据
    3) SFT 指令微调
    4) DPO 偏好对齐
    5) 蒸馏到更小模型（可选）
    6) 评估基准

用法：
    # 完整跑一遍（默认配置）
    python scripts/pipeline.py --run-all

    # 只跑数据生成
    python scripts/pipeline.py --data-only --sft-num 1000 --dpo-num 300

    # 只跑训练（假设数据已准备好）
    python scripts/pipeline.py --train-only

    # 自定义规模
    python scripts/pipeline.py --n-layer 6 --n-embd 384 \
        --sft-steps 3000 --dpo-steps 1000 \
        --exp-dir experiments/exp_edu_v1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# 项目根
PROJ_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJ_DIR)


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def run_cmd(cmd: str) -> int:
    log(f"执行: {cmd}")
    t0 = time.time()
    ret = os.system(cmd)
    if ret == 0:
        log(f"完成，耗时 {time.time() - t0:.1f}s")
    else:
        log(f"失败 (exit={ret})，耗时 {time.time() - t0:.1f}s", "ERROR")
    return ret


# ============ 各阶段 ============

def step_corpus(args):
    raw_src = os.path.join(PROJ_DIR, "data/raw_src")
    out_file = os.path.join(PROJ_DIR, "data/raw/edu_corpus.txt")
    if not os.path.exists(raw_src):
        log(f"语料目录不存在: {raw_src} — 跳过（如果需要请先把课本/题库/教案放进去）",
            "WARN")
        return True

    if run_cmd(f'cd "{PROJ_DIR}" && python data/corpus_collector.py '
               f'--input-dir data/raw_src --output data/raw/edu_corpus.txt') != 0:
        return False
    if os.path.exists(out_file):
        size_mb = os.path.getsize(out_file) / 1024 / 1024
        log(f"语料大小: {size_mb:.2f} MB")
    return True


def step_sft_data(args):
    out_path = os.path.join(PROJ_DIR, "data/sft/raw.jsonl")
    os.makedirs(os.path.join(PROJ_DIR, "data/sft"), exist_ok=True)
    cmd = (f'cd "{PROJ_DIR}" && python scripts/teacher_inference.py '
           f'--mode sft --output data/sft/raw.jsonl --num {args.sft_num} '
           f'--models {" ".join(args.models)}')
    if run_cmd(cmd) != 0:
        return False

    # 格式整理 + 切分验证集
    cmd2 = (f'cd "{PROJ_DIR}" && python scripts/prepare_sft.py '
            f'--input data/sft/raw.jsonl --output data/sft/train.jsonl')
    return run_cmd(cmd2) == 0


def step_dpo_data(args):
    out_path = os.path.join(PROJ_DIR, "data/dpo/raw.jsonl")
    os.makedirs(os.path.join(PROJ_DIR, "data/dpo"), exist_ok=True)
    cmd = (f'cd "{PROJ_DIR}" && python scripts/teacher_inference.py '
           f'--mode dpo --output data/dpo/raw.jsonl --num {args.dpo_num} '
           f'--models {" ".join(args.models)}')
    return run_cmd(cmd) == 0


def step_pretrain(args):
    """预训练 — 如果用户已有 checkpoint 可跳过"""
    ckpt_path = os.path.join(PROJ_DIR, args.exp_dir, "checkpoints/best.pt")
    if os.path.exists(ckpt_path):
        log(f"发现已有 checkpoint: {ckpt_path} — 跳过预训练", "WARN")
        return True

    config_path = os.path.join(PROJ_DIR, "configs/lmm_small.yaml")
    if not os.path.exists(config_path):
        log(f"找不到配置文件: {config_path}，跳过预训练", "WARN")
        return True

    cmd = (f'cd "{PROJ_DIR}" && python scripts/train.py '
           f'--config configs/lmm_small.yaml --steps {args.pretrain_steps}')
    return run_cmd(cmd) == 0


def _find_best_checkpoint(exp_dir):
    """在 exp_dir 的 checkpoints/ 下找 best.pt；找不到就看 experiments/*/checkpoints/best.pt"""
    candidates = [
        os.path.join(PROJ_DIR, exp_dir, "checkpoints/best.pt"),
        os.path.join(PROJ_DIR, "experiments/exp_small_v1/checkpoints/best.pt"),
        os.path.join(PROJ_DIR, "experiments/exp_default/checkpoints/best.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def step_sft(args):
    data_path = os.path.join(PROJ_DIR, "data/sft/train.jsonl")
    if not os.path.exists(data_path):
        log(f"找不到 SFT 数据: {data_path}，请先跑 --data-only", "ERROR")
        return False

    ckpt = _find_best_checkpoint(args.exp_dir)
    ckpt_flag = f'--checkpoint {os.path.relpath(ckpt, PROJ_DIR)}' if ckpt else ''

    out_dir = os.path.join(args.exp_dir, "sft")
    cmd = (f'cd "{PROJ_DIR}" && python scripts/train_sft.py '
           f'--data data/sft/train.jsonl '
           f'{ckpt_flag} '
           f'--exp-dir {out_dir} '
           f'--steps {args.sft_steps} '
           f'--batch-size {args.batch_size} '
           f'--lr {args.lr} '
           f'--n-layer {args.n_layer} --n-embd {args.n_embd} '
           f'{"--use-lora --lora-rank 8 --lora-alpha 16" if args.use_lora else ""}')
    return run_cmd(cmd) == 0


def step_dpo(args):
    data_path = os.path.join(PROJ_DIR, "data/dpo/raw.jsonl")
    if not os.path.exists(data_path):
        log(f"找不到 DPO 数据: {data_path}，请先跑 --data-only --dpo-num XXX",
            "WARN")
        return False

    # SFT 后的 checkpoint 作为 DPO 起点
    sft_ckpt = os.path.join(PROJ_DIR, args.exp_dir, "sft/checkpoints/best.pt")
    if not os.path.exists(sft_ckpt):
        # 退回其他候选
        sft_ckpt = _find_best_checkpoint(args.exp_dir) or _find_best_checkpoint(
            os.path.join(args.exp_dir, "sft"))

    ckpt_flag = f'--checkpoint {os.path.relpath(sft_ckpt, PROJ_DIR)}' if sft_ckpt else ''
    out_dir = os.path.join(args.exp_dir, "dpo")

    cmd = (f'cd "{PROJ_DIR}" && python scripts/train_dpo.py '
           f'--data data/dpo/raw.jsonl '
           f'{ckpt_flag} '
           f'--exp-dir {out_dir} '
           f'--steps {args.dpo_steps} '
           f'--batch-size {args.batch_size} '
           f'--lr {args.dpo_lr} --beta {args.beta} '
           f'--n-layer {args.n_layer} --n-embd {args.n_embd}')
    return run_cmd(cmd) == 0


def step_eval(args):
    ckpt = _find_best_checkpoint(os.path.join(args.exp_dir, "dpo")) \
        or _find_best_checkpoint(os.path.join(args.exp_dir, "sft")) \
        or _find_best_checkpoint(args.exp_dir)
    if not ckpt:
        log("没有找到任何 checkpoint，跳过评估", "WARN")
        return False

    val_data = os.path.join(PROJ_DIR, "data/sft/val.jsonl")
    if not os.path.exists(val_data):
        val_data = os.path.join(PROJ_DIR, "data/sft/train.jsonl")

    out_report = os.path.join(PROJ_DIR, args.exp_dir, "eval_report.json")
    cmd = (f'cd "{PROJ_DIR}" && python scripts/eval_edu_benchmark.py '
           f'--checkpoint {os.path.relpath(ckpt, PROJ_DIR)} '
           f'--val-data {os.path.relpath(val_data, PROJ_DIR)} '
           f'--compare-teacher {args.compare_teacher} '
           f'--output {os.path.relpath(out_report, PROJ_DIR)}')
    return run_cmd(cmd) == 0


def step_distill(args):
    teacher_ckpt = _find_best_checkpoint(
        os.path.join(args.exp_dir, "dpo")) or _find_best_checkpoint(
        os.path.join(args.exp_dir, "sft"))
    if not teacher_ckpt:
        log("没有找到 teacher checkpoint，跳过蒸馏", "WARN")
        return False

    out_dir = os.path.join(args.exp_dir, "distill")
    cmd = (f'cd "{PROJ_DIR}" && python scripts/distill.py '
           f'--teacher-ckpt {os.path.relpath(teacher_ckpt, PROJ_DIR)} '
           f'--teacher-n-layer {args.n_layer} --teacher-n-embd {args.n_embd} '
           f'--student-n-layer {args.student_n_layer} '
           f'--student-n-embd {args.student_n_embd} '
           f'--data data/sft/train.jsonl --exp-dir {out_dir} '
           f'--steps {args.distill_steps} --temperature 2.0 --alpha 0.7')
    return run_cmd(cmd) == 0


# ============ 主入口 ============

def main():
    p = argparse.ArgumentParser(description="LMM 端到端训练流水线")
    p.add_argument("--run-all", action="store_true",
                   help="跑完整流程：数据 → 预训练 → SFT → DPO → 评估 → 蒸馏")
    p.add_argument("--data-only", action="store_true",
                   help="只跑数据生成（语料 + SFT + DPO）")
    p.add_argument("--train-only", action="store_true",
                   help="只跑训练阶段（假设数据已准备好）")

    p.add_argument("--exp-dir", default="experiments/exp_edu_v1",
                   help="实验输出目录")
    p.add_argument("--models", nargs="+",
                   default=["qwen2.5:7b-instruct", "deepseek-r1:1.5b"],
                   help="教师模型列表（Ollama 模型名）")

    # 数据规模
    p.add_argument("--sft-num", type=int, default=1000,
                   help="SFT 问答对数量")
    p.add_argument("--dpo-num", type=int, default=300,
                   help="DPO 偏好对数量")

    # 模型规模
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--n-embd", type=int, default=768)
    p.add_argument("--student-n-layer", type=int, default=6,
                   help="蒸馏后小模型层数")
    p.add_argument("--student-n-embd", type=int, default=384,
                   help="蒸馏后小模型嵌入维度")

    # 训练参数
    p.add_argument("--pretrain-steps", type=int, default=3000)
    p.add_argument("--sft-steps", type=int, default=2000)
    p.add_argument("--dpo-steps", type=int, default=1000)
    p.add_argument("--distill-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--dpo-lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    p.add_argument("--use-lora", action="store_true",
                   help="SFT 阶段用 LoRA（省显存）")
    p.add_argument("--compare-teacher", default="qwen2.5:7b-instruct",
                   help="评估时对比的教师模型")

    args = p.parse_args()

    # 绝对路径的实验目录
    args.exp_dir = os.path.relpath(args.exp_dir, PROJ_DIR) \
        if os.path.isabs(args.exp_dir) else args.exp_dir
    os.makedirs(os.path.join(PROJ_DIR, args.exp_dir), exist_ok=True)

    # 保存本次运行参数（方便回放）
    with open(os.path.join(PROJ_DIR, args.exp_dir, "run_config.json"),
              "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  LMM 端到端多教师训练流水线")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  实验目录: {args.exp_dir}")
    print(f"  模型规模: {args.n_layer} 层 × {args.n_embd} 维嵌入")
    print(f"  教师模型: {', '.join(args.models)}")
    print(f"  SFT 问答对: {args.sft_num}  DPO 偏好对: {args.dpo_num}")
    print(f"  SFT 步数: {args.sft_steps}  DPO 步数: {args.dpo_steps}")
    print(f"  LoRA: {'启用' if args.use_lora else '未启用'}")
    print("=" * 70 + "\n")

    results = {}

    try:
        if args.run_all or args.data_only:
            results["corpus"] = step_corpus(args)
            results["sft_data"] = step_sft_data(args)
            results["dpo_data"] = step_dpo_data(args)

        if args.run_all or args.train_only:
            results["pretrain"] = step_pretrain(args)
            results["sft"] = step_sft(args)
            results["dpo"] = step_dpo(args)
            results["eval"] = step_eval(args)
            if args.distill_steps > 0:
                results["distill"] = step_distill(args)

    except KeyboardInterrupt:
        log("用户中断，保存进度中...", "WARN")

    # 结果汇总
    print("\n" + "=" * 70)
    print("  流水线结果汇总")
    print("=" * 70)
    for stage, ok in results.items():
        status = "✅ 成功" if ok else "❌ 失败/跳过"
        print(f"  {stage:12s} {status}")
    print()

    # 最终模型路径提示
    final_candidates = [
        os.path.join(PROJ_DIR, args.exp_dir, "distill/checkpoints/best.pt"),
        os.path.join(PROJ_DIR, args.exp_dir, "dpo/checkpoints/best.pt"),
        os.path.join(PROJ_DIR, args.exp_dir, "sft/checkpoints/best.pt"),
    ]
    for path in final_candidates:
        if os.path.exists(path):
            print(f"  🎯 最终模型: {path}")
            break

    report_path = os.path.join(PROJ_DIR, args.exp_dir, "eval_report.json")
    if os.path.exists(report_path):
        print(f"  📊 评估报告: {report_path}")

    print(f"\n  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
