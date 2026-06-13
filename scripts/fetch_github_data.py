#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_github_data.py —— 从 GitHub 私有仓库自动拉取最新训练材料

使用方式:
    python scripts/fetch_github_data.py                     # 拉取全部
    python scripts/fetch_github_data.py --only sft         # 只拉取 data/scripts（SFT 训练用）
    python scripts/fetch_github_data.py --only data      # 只拉取 data
    python scripts/fetch_github_data.py --branch lmm-training-materials  # 指定分支

训练流程:
    1) python scripts/fetch_github_data.py  ← 拉取最新训练数据
    2) python scripts/pipeline.py --run-all  ← 跑完整训练流程

安全性: Token 仅从 $GITHUB_TOKEN 或 ~/.github_token 读取，绝不写入仓库文件
"""
import os
import sys
import json
import urllib.request
import zipfile
import tempfile
import argparse
import pathlib
from datetime import datetime

REPO_OWNER = "k3234"
REPO_NAME = "edu-tutor-ai"
BRANCH = "lmm-training-materials"
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".github_token")


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and token.strip():
        return token.strip()
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            t = f.read().strip()
        if t:
            return t
    return ""


def gh_api(method, path, token, timeout=30):
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "LMM-Edu-Training",
    }
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return resp.status, json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return e.code, json.loads(body) if body.strip() else {}
    except Exception as e:
        return 0, {"error": str(e)}


def download_repo_zip(token, branch):
    print(f"  下载 {REPO_OWNER}/{REPO_NAME}@{branch} ...")
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/zipball/{branch}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "LMM-Edu-Training",
    }
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp.write(resp.read())
            tmp.close()
            return tmp.name
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return None


def extract_and_merge(zip_path, mode):
    print(f"  解压并合并到: {PROJECT_DIR}")
    stats = {
        "models": 0, "configs": 0, "scripts": 0,
        "data": 0, "trainer": 0, "docs": 0,
        "workspace": 0, "rules": 0, "other": 0, "total": 0,
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # GitHub zipball 的第一层是 <owner>-<repo>-<hash>/
        prefix = ""
        for n in names:
            if "/" in n and not n.startswith("."):
                prefix = n.split("/")[0]
                break

        extracted = 0
        skipped = 0
        for item in names:
            # 去掉前缀
            rel = item[len(prefix) + 1:] if prefix and item.startswith(prefix + "/") else item
            if not rel or rel.endswith("/"):
                continue

            # 模式过滤
            if mode == "sft" and not (rel.startswith("data/") or rel.startswith("scripts/")):
                skipped += 1
                continue
            if mode == "data" and not rel.startswith("data/"):
                skipped += 1
                continue
            # 跳过敏感/临时文件
            skip_list = [".github_token", ".env", "__pycache__", "experiments/"]
            if any(s in rel for s in skip_list):
                skipped += 1
                continue

            target = PROJECT_DIR / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(item) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                stats["total"] += 1
                found = False
                for key in list(stats.keys())[:-1]:
                    if rel.startswith(key + "/"):
                        stats[key] += 1
                        found = True
                        break
                if not found:
                    stats["other"] += 1
            except Exception as e:
                print(f"  ⚠️  {rel}: {e}")

        print(f"  ✅ 共提取 {extracted} 个文件，跳过 {skipped} 个")
    return stats


def main():
    parser = argparse.ArgumentParser(description="从 GitHub 私有仓库拉取训练材料")
    parser.add_argument("--only", choices=["all", "sft", "data"], default="all",
                        help="all(默认)/sft(data+scripts)/data")
    parser.add_argument("--branch", default=BRANCH, help="指定分支")
    args = parser.parse_args()

    token = get_token()
    if not token:
        print("❌ 未找到 GitHub Token")
        print("请在 PowerShell 执行:")
        print('  $env:GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"')
        print('  或: echo "ghp_xxxxxxxxxxxxxxxxxxxx" > "$env:USERPROFILE\\.github_token"')
        sys.exit(1)

    print("=" * 70)
    print(f"  从 GitHub 私有仓库拉取训练数据")
    print(f"  仓库: {REPO_OWNER}/{REPO_NAME}")
    print(f"  分支: {args.branch}   模式: {args.only}")
    print("=" * 70)

    # 验证仓库可访问
    code, info = gh_api("GET", f"/repos/{REPO_OWNER}/{REPO_NAME}", token)
    if code != 200:
        print(f"❌ 无法访问仓库 (HTTP {code})")
        if code == 404:
            print(f"  → 仓库 {REPO_OWNER}/{REPO_NAME} 不存在或 Token 无权限")
        sys.exit(1)
    print(f"✅ 仓库可访问 - {info.get('description', '无描述')}")

    # 下载 zip
    zip_path = download_repo_zip(token, args.branch)
    if not zip_path:
        sys.exit(1)
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"✅ 下载完成: {size_mb:.2f} MB")

    # 解压合并
    stats = extract_and_merge(zip_path, args.only)
    try:
        os.unlink(zip_path)
    except Exception:
        pass

    print()
    print("=" * 70)
    print("✅ 拉取完成 - 各目录文件数:")
    for k, v in stats.items():
        print(f"  {k:15s}: {v}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    print("  接下来可以开始训练:")
    print(f"    python scripts/pipeline.py --run-all")
    print()


if __name__ == "__main__":
    main()
