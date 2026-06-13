# -*- coding: utf-8 -*-
"""
发布脚本 2 —— 已知 Token 有权限限制（不能通过 API 创建仓库），
所以改为：检查仓库是否存在 → 本地 commit → 直接 push。
如果仓库不存在，会提示你到网页手动创建。
"""
import os
import sys
import json
import time
import urllib.request
import subprocess
import pathlib
import shutil

REPO_NAME = "LMM-Edu-Training"
GITHUB_USERNAME = "k3234"  # 从 API 验证得到的用户名
PROJECT_ROOT = pathlib.Path(r"e:\学习LLM\开发者学习\09-LLM从零学习\LMM")


def banner(text):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and token.strip():
        return token.strip()
    return ""


def gh_api_get(path, token, timeout=30):
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "LMM-Uploader",
    }
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            return resp.status, json.loads(text) if text.strip() else {}
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="ignore")
        return e.code, (json.loads(text) if text.strip() else {})
    except Exception as e:
        return 0, {"error": str(e)}


def main():
    token = get_token()
    if not token:
        print("❌ 未找到 GITHUB_TOKEN")
        print("请先在 PowerShell 执行:")
        print('  $env:GITHUB_TOKEN="你的Token"')
        sys.exit(1)

    print()
    print("=" * 70)
    print("  LMM 教育训练材料 —— 发布到 GitHub 私有仓库")
    print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  项目目录: {PROJECT_ROOT}")
    print(f"  Token 摘要: {token[:4]}...{token[-4:]} ({len(token)} 字符)")
    print("=" * 70)

    # Step 1: 验证 Token + 检查仓库
    banner("第 1/4 步: 检查仓库是否已存在")
    code, data = gh_api_get(f"/repos/{GITHUB_USERNAME}/{REPO_NAME}", token)
    repo_exists = (code == 200)
    if repo_exists:
        print(f"  ✅ 仓库已存在: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
        print(f"     默认分支: {data.get('default_branch', 'main')}")
    else:
        print(f"  ⚠️  仓库不存在 (HTTP {code})")
        print()
        print("  请按以下步骤在 GitHub 网页手动创建仓库:")
        print(f"    1) 打开: https://github.com/new")
        print(f"    2) Repository name 填: {REPO_NAME}")
        print(f"    3) 勾选 Private (私有)")
        print(f"    4) 其他保持默认，点绿色按钮 Create repository")
        print(f"    5) 创建完成后，回到本对话说一声 '已创建'")
        print()
        print("  或者你也可以直接在命令行通过 gh CLI:")
        print(f'    gh repo create {REPO_NAME} --private --description "LMM 多教师模型训练框架"')
        print()
        sys.exit(0)

    # Step 2: 初始化 git 并准备文件
    banner("第 2/4 步: 初始化 git 并准备文件清单")
    os.chdir(str(PROJECT_ROOT))

    git_dir = PROJECT_ROOT / ".git"
    if git_dir.exists():
        print("  已存在 .git，清理后重新初始化保证干净")
        shutil.rmtree(git_dir, ignore_errors=True)

    subprocess.run(["git", "init", "-b", "main"], cwd=str(PROJECT_ROOT), capture_output=True)
    time.sleep(0.5)
    subprocess.run(["git", "config", "user.name", "LMM Trainer"], cwd=str(PROJECT_ROOT))
    subprocess.run(["git", "config", "user.email", "lmm@trainer.local"], cwd=str(PROJECT_ROOT))
    print("  ✅ git 初始化完成")

    # 写 .gitignore
    (PROJECT_ROOT / ".gitignore").write_text(
        "# 训练输出\n"
        "experiments/\n"
        "checkpoints/\n"
        "*.pt\n"
        "*.pth\n"
        "# 敏感\n"
        ".env\n"
        ".github_token\n"
        "# 临时\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".uploads/\n"
        ".venv/\n"
        "venv/\n"
        "# 压缩\n"
        "*.tar.gz\n"
        "*.zip\n",
        encoding="utf-8"
    )
    print("  ✅ .gitignore 已创建")

    # 列出文件
    include = [
        "models", "configs", "scripts", "data", "trainer", "docs",
        "workspace", "rules",
        "AGENTS.md", "PROJECT_SUMMARY.md", "README.md", "requirements.txt",
        "DESIGN.md", "PRODUCT.md",
        "shared_chat.json", "shared_chat (2).json",
        "smoke_test.py", "Dockerfile", "docker-compose.yml",
        ".gitignore", "publish_to_github.py",
    ]
    files_found = []
    for item in include:
        p = PROJECT_ROOT / item
        if p.exists():
            files_found.append(item)
            kind = "目录" if p.is_dir() else "文件"
            print(f"  + {item} ({kind})")
    print(f"  共 {len(files_found)} 项")

    # Step 3: git add + commit
    banner("第 3/4 步: git add + commit")
    add_ok = True
    for item in files_found:
        r = subprocess.run(["git", "add", item], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if r.returncode != 0 and r.stderr.strip():
            print(f"  ⚠️  add {item}: {r.stderr.strip()[:200]}")
            add_ok = False

    r = subprocess.run(
        ["git", "commit", "-m", "初始化 LMM 教育训练框架 - 多教师蒸馏训练材料", "--allow-empty"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    if r.returncode == 0:
        out_lines = [l for l in r.stdout.splitlines() if l.strip()]
        print(f"  ✅ commit 成功 —— {out_lines[-1] if out_lines else ''}")
    else:
        print(f"  commit stdout: {r.stdout[:500]}")
        print(f"  commit stderr: {r.stderr[:500]}")

    # Step 4: push
    banner("第 4/4 步: push 到 GitHub 私有仓库")
    push_url = f"https://x-access-token:{token}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
    r = subprocess.run(["git", "push", "-u", push_url, "main"],
                       cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=600)

    out = r.stdout.replace(token, "***TOKEN***")
    err = r.stderr.replace(token, "***TOKEN***")

    if out.strip():
        print(f"  stdout: {out[:500]}")
    if err.strip():
        print(f"  stderr: {err[:500]}")

    print()
    print("=" * 70)
    if r.returncode == 0:
        print("🎯 全部完成！训练材料已上传到 GitHub 私有仓库")
        print(f"  🔗 https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
        print(f"  📦 已上传 {len(files_found)} 项")
        print(f"  🛠  接下来可以:")
        print(f'     python scripts/fetch_github_data.py  ← 从 GitHub 拉取数据训练')
        print(f'     python scripts/pipeline.py --run-all    ← 跑完整训练流程')
    else:
        print(f"❌ push 失败 (返回码 {r.returncode})")
        if "Repository not found" in err:
            print("  → 原因: 仓库不存在。请先到 https://github.com/new 手动创建私有仓库")
        elif "Permission denied" in err or "403" in err or "Forbidden" in err:
            print("  → 原因: Token 没有该仓库的 push 权限。请检查 Token 是否勾选 repo 权限")
        elif "Large files" in err or "100 MB" in err:
            print("  → 原因: 有文件超过 GitHub 的 100 MB 限制")
        else:
            print("  → 请把上方 stderr 信息发给我继续诊断")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
