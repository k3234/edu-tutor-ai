# -*- coding: utf-8 -*-
"""冒烟测试：检查脚本语法、数据格式是否正确"""
import ast
import json
import pathlib

root = pathlib.Path(__file__).parent
print(f"项目根目录: {root}\n")

# 1. 检查所有 Python 脚本的语法
scripts = [
    "models/model.py", "models/lora.py",
    "scripts/teacher_inference.py", "scripts/prepare_sft.py",
    "scripts/train_sft.py", "scripts/train_dpo.py",
    "scripts/distill.py", "scripts/eval_edu_benchmark.py",
    "scripts/pipeline.py",
    "data/corpus_collector.py",
]
print("=" * 60)
print("  Python 脚本语法检查")
print("=" * 60)
errors = 0
for s in scripts:
    path = root / s
    if not path.exists():
        print(f"  ❌ 不存在: {s}")
        errors += 1
        continue
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        print(f"  ✅ {s}")
    except SyntaxError as e:
        print(f"  ❌ {s}: {e}")
        errors += 1
print(f"  结果: {len(scripts)-errors}/{len(scripts)} 通过\n")

# 2. 检查示例 SFT 数据
print("=" * 60)
print("  SFT 示例数据检查")
print("=" * 60)
data_path = root / "data" / "sample_sft.jsonl"
if data_path.exists():
    lines = [l.strip() for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = 0
    total_answer_chars = 0
    subjects = {}
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
            assert "prompt" in obj and "answer" in obj, "缺少必要字段"
            total_answer_chars += len(obj["answer"])
            subj = obj.get("subject", "未分类")
            subjects[subj] = subjects.get(subj, 0) + 1
            valid += 1
        except Exception as e:
            print(f"  ❌ 第{i+1}行: {e}")
    print(f"  ✅ {valid}/{len(lines)} 条数据合法")
    print(f"  - 答案总字数: {total_answer_chars} 字")
    print(f"  - 平均每条: {total_answer_chars//valid} 字")
    print(f"  - 学科分布:")
    for subj, cnt in sorted(subjects.items(), key=lambda x: -x[1]):
        print(f"      · {subj}: {cnt} 条")
else:
    print(f"  ❌ 找不到 sample_sft.jsonl")
print()

# 3. 检查 configs 是否齐全
print("=" * 60)
print("  配置文件检查")
print("=" * 60)
configs = ["configs/lmm_small.yaml", "configs/lmm_3b.yaml", "configs/lmm_7b.yaml"]
for c in configs:
    path = root / c
    if path.exists():
        print(f"  ✅ {c}")
    else:
        print(f"  ⚠️  缺少: {c} (训练预训练阶段会需要)")
print()

# 4. 检查 AGENTS.md
agents_path = root / "AGENTS.md"
if agents_path.exists():
    content = agents_path.read_text(encoding="utf-8")
    # 简单检查是否包含核心关键词
    keywords = ["Qwen", "DeepSeek", "Trae", "SFT", "DPO", "蒸馏", "教师"]
    hits = sum(1 for k in keywords if k in content)
    print(f"  ✅ AGENTS.md 存在 ({hits}/{len(keywords)} 个核心关键词命中)")
else:
    print("  ❌ 缺少 AGENTS.md")

print("\n" + "=" * 60)
print("  🎉 冒烟测试完成，框架结构已就绪")
print("  下一步: pip install torch, 然后运行 scripts/pipeline.py --run-all")
print("=" * 60)
