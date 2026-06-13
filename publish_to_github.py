#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
publish_to_github.py —— 将 LMM 训练材料发布到 GitHub 私有仓库
安全性: Token 仅从 $GITHUB_TOKEN 或 ~/.github_token 读取，绝不打印到日志
流程:
    1. 验证 GitHub Token
    2. 创建私有仓库 LMM-Edu-Training
    3. 本地初始化 git，整理训练材料文件
    4. 第一次 commit 并 push
    5. 生成 fetch_github_data.py（训练时自动拉取数据）
"""
import os
import sys
import json
import time
import urllib.request
import subprocess
import pathlib
from datetime import datetime

# ===== 配置 =====
REPO_NAME = "LMM-Edu-Training"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
TOKEN_FILE = os.path.join(os.path.expanduser("~"), ".github_token")

# 需要上传的目录/文件
INCLUDE_LIST = [
    "models",
    "configs",
    "scripts",
    "data",
    "trainer",
    "docs",
    "workspace",
    "rules",
    "AGENTS.md",
    "PROJECT_SUMMARY.md",
    "README.md",
    "requirements.txt",
    "DESIGN.md",
    "PRODUCT.md",
    "shared_chat.json",
    "shared_chat (2).json",
    "smoke_test.py",
    "Dockerfile",
    "docker-compose.yml",
]


# ===== 工具函数 =====
def banner(text):
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)
    flush()


def flush():
    try:
        sys.stdout.flush()
    except Exception:
        pass


def get_token():
    """从环境变量或主目录文件读取 Token"""
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and token.strip():
        return token.strip()
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r", encoding="utf-8") as f:
            token = f.read().strip()
        if token:
            return token
    return ""


def mask_token(text):
    """把 Token 字符串替换为 ***MASKED***"""
    token = get_token()
    if not token or not text:
        return text or ""
    try:
        return text.replace(token, "***MASKED***")
    except Exception:
        return text


def gh_api(method, path, token, data=None, timeout=30):
    """调用 GitHub REST API，不打印 Token"""
    url = f"https://api.github.com{path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "LMM-Training-Uploader",
    }
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            return resp.status, (json.loads(text) if text.strip() else {})
    except urllib.error.HTTPError as e:
        try:
            text = e.read().decode("utf-8", errors="ignore")
            return e.code, (json.loads(text) if text.strip() else {})
        except Exception:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


def run_cmd(cmd_parts, cwd=None, timeout=600):
    """执行命令，不打印 Token"""
    cwd = cwd or str(PROJECT_ROOT)
    # 打印命令但隐藏可能包含 Token 的部分
    display = " ".join(str(c) for c in cmd_parts[:3]) + (" ..." if len(cmd_parts) > 3 else "")
    print(f"  $ {display}")
    flush()
    try:
        result = subprocess.run(
            cmd_parts,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        if result.stdout:
            print(mask_token(result.stdout.strip()[:300]))
        if result.stderr:
            print(mask_token(result.stderr.strip()[:300]))
        flush()
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ⏱️  超时")
        return False
    except Exception as e:
        print(f"  ❌ {e}")
        return False


# ===== 各阶段 =====
def step1_verify_token():
    banner("第 1/5 步: 验证 GitHub Token")
    token = get_token()
    if not token:
        print("❌ 未找到 Token！请先在 PowerShell 执行:")
        print('  $env:GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"')
        print('  或: echo "ghp_xxxxxxxxxxxxxxxxxxxx" > "$env:USERPROFILE\\.github_token"')
        sys.exit(1)

    # 显示摘要 (不打印完整 Token)
    print(f"  Token: {token[:4]}...{token[-4:]} ({len(token)} 字符)")
    code, data = gh_api("GET", "/user", token)
    if code == 200:
        username = data.get("login", "unknown")
        print(f"  ✅ 验证成功 —— GitHub 用户名: {username}")
        return token, username
    print(f"  ❌ Token 无效 (HTTP {code}): {json.dumps(data, ensure_ascii=False)[:400]}")
    print("  请检查: 1) Token 内容是否正确  2) 是否勾选 repo 权限")
    sys.exit(1)


def step2_create_repo(token, username):
    banner("第 2/5 步: 创建 GitHub 私有仓库")
    print(f"  仓库: {username}/{REPO_NAME}")

    # 先检查是否已存在
    code, data = gh_api("GET", f"/repos/{username}/{REPO_NAME}", token)
    if code == 200:
        print(f"  ℹ️  仓库已存在: {data.get('html_url')}")
        return data.get("clone_url"), data.get("html_url")

    # 创建新的私有仓库
    body = {
        "name": REPO_NAME,
        "description": "LMM 多教师模型教育领域小模型训练框架",
        "private": True,
        "has_issues": True,
        "has_wiki": True,
        "has_projects": True,
        "auto_init": False,
    }
    code, data = gh_api("POST", "/user/repos", token, data=body)
    if code in (200, 201):
        print(f"  ✅ 仓库创建成功！")
        print(f"  地址: {data.get('html_url')}")
        return data.get("clone_url"), data.get("html_url")
    print(f"  ❌ 创建失败 (HTTP {code}): {json.dumps(data, ensure_ascii=False)[:400]}")
    sys.exit(1)


def step3_prepare_files():
    banner("第 3/5 步: 初始化 git 并整理文件")

    # 初始化 git
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.exists():
        print("  ℹ️  已存在 .git 目录，跳过初始化")
    else:
        run_cmd(["git", "init", "-b", "main"])
        time.sleep(1)
        run_cmd(["git", "config", "user.name", "LMM Trainer"])
        run_cmd(["git", "config", "user.email", "lmm@trainer.local"])

    # 创建 .gitignore
    gitignore = PROJECT_ROOT / ".gitignore"
    ignore_content = (
        "# 训练输出\n"
        "experiments/\n"
        "checkpoints/\n"
        "*.pt\n"
        "*.pth\n"
        "# 敏感信息\n"
        ".env\n"
        ".github_token\n"
        "# 临时文件\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".uploads/\n"
        ".venv/\n"
        "venv/\n"
        "# 压缩包\n"
        "*.tar.gz\n"
        "*.zip\n"
        "# 数据临时备份\n"
        "data/raw/*.bak\n"
    )
    with open(gitignore, "w", encoding="utf-8") as f:
        f.write(ignore_content)
    print("  ✅ .gitignore 已创建")

    # 列出并添加文件
    files_found = []
    for item in INCLUDE_LIST:
        p = PROJECT_ROOT / item
        if p.exists():
            kind = "目录" if p.is_dir() else "文件"
            print(f"  + {item} ({kind})")
            files_found.append(item)
        else:
            print(f"  ⚠️  {item} 不存在，跳过")

    print(f"\n  共 {len(files_found)} 项准备上传")
    return files_found


def step4_push(token, files, username, clone_url):
    banner("第 4/5 步: 提交并 push 到 GitHub")

    # 设置 remote
    run_cmd(["git", "remote", "remove", "origin"])
    clean_url = clone_url or f"https://github.com/{username}/{REPO_NAME}.git"
    run_cmd(["git", "remote", "add", "origin", clean_url])

    # 添加文件
    run_cmd(["git", "add", "--all"] + files)

    # 提交
    commit_msg = "初始化 LMM 教育训练框架 - 多教师蒸馏训练材料"
    run_cmd(["git", "commit", "-m", commit_msg, "--allow-empty"])

    # push —— 用 HTTPS Header 认证 (避免 Token 写入 git config)
    print("\n  正在 push 到 GitHub (首次可能较慢)...")
    push_url = f"https://x-access-token:{token}@github.com/{username}/{REPO_NAME}.git"
    ok = run_cmd(["git", "push", "-u", push_url, "main"])

    if not ok:
        # 换一种方式: 用 git -c http.extraHeader
        print("  ⚠️  方式 1 失败，尝试方式 2...")
        header = f"Authorization: Basic {__import__('base64').b64encode(f'{token}:x-oauth-basic'.encode()).decode()}"
        ok = run_cmd(["git", "-c", f"http.extraHeader={header}", "push", "-u", "origin", "main"])

    if ok:
        print("  ✅ push 成功！")
    else:
        print("  ❌ push 失败，请检查网络和 Token 权限")
    return ok


def step5_generate_fetch_script(username):
    banner("第 5/5 步: 生成自动读取脚本")

    # 脚本模板 (用三重引号避免 f-string 嵌套)
    template = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_github_data.py —— 从 GitHub 私有仓库自动拉取最新训练材料

使用方式:
    python scripts/fetch_github_data.py                     # 拉取全部
    python scripts/fetch_github_data.py --only sft         # 只拉取 data/scripts
    python scripts/fetch_github_data.py --only data        # 只拉取 data
    python scripts/fetch_github_data.py --branch dev       # 指定分支

训练流程中使用:
    1) 跑本脚本拉取最新训练数据
    2) python scripts/pipeline.py --run-all 训练
    3) 把训练结果 push 回 GitHub

安全性: Token 仅从 $GITHUB_TOKEN 或 ~/.github_token 读取，绝不写入仓库
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

REPO_OWNER = "''' + username + '''"
REPO_NAME = "''' + REPO_NAME + '''"
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


def download_repo_zip(token, branch="main"):
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


def extract_and_merge(zip_path, mode="all"):
    print(f"  解压并合并到: {PROJECT_DIR}")
    stats = {
        "models": 0, "configs": 0, "scripts": 0,
        "data": 0, "trainer": 0, "docs": 0,
        "workspace": 0, "rules": 0, "other": 0, "total": 0,
    }

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # 找出 GitHub 添加的 <owner>-<repo>-<hash>/ 前缀
        prefix = ""
        for n in names:
            if "/" in n and not n.startswith("."):
                prefix = n.split("/")[0]
                break
        if not prefix:
            print("  ⚠️  zip 格式异常，尝试直接解压")

        extracted = 0
        skipped = 0
        for item in names:
            # 去掉前缀
            rel = item[len(prefix) + 1:] if prefix and item.startswith(prefix + "/") else item
            if not rel or rel.endswith("/"):
                continue

            # 根据模式决定是否保留
            if mode == "sft" and not (rel.startswith("data/") or rel.startswith("scripts/")):
                skipped += 1
                continue
            if mode == "data" and not rel.startswith("data/"):
                skipped += 1
                continue
            # 跳过敏感文件
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
                for key in list(stats.keys())[:-1]:
                    if rel.startswith(key + "/"):
                        stats[key] += 1
                        break
                else:
                    stats["other"] += 1
            except Exception as e:
                print(f"  ⚠️  {rel}: {e}")

        print(f"  ✅ 共提取 {extracted} 个文件，跳过 {skipped} 个")
    return stats


def main():
    parser = argparse.ArgumentParser(description="从 GitHub 私有仓库拉取训练材料")
    parser.add_argument("--only", choices=["all", "sft", "data"], default="all",
                        help="只拉取特定内容: all(默认)/sft/data")
    parser.add_argument("--branch", default="main", help="指定分支")
    args = parser.parse_args()

    token = get_token()
    if not token:
        print("❌ 未找到 GitHub Token")
        print("请在 PowerShell 执行:")
        print('  $env:GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"')
        print('  或: echo "ghp_xxxxxxxxxxxxxxxxxxxx" > "$env:USERPROFILE\\\\.github_token"')
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
        sys.exit(1)
    print(f"✅ 仓库可访问 - {info.get('description', '')}")

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

    print("\n" + "=" * 70)
    print("✅ 拉取完成 - 各目录文件数:")
    for k, v in stats.items():
        print(f"  {k:15s}: {v}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'''

    script_path = PROJECT_DIR / "scripts" / "fetch_github_data.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(template)
    print(f"  ✅ scripts/fetch_github_data.py 已生成")
    print(f"  绑定用户: {username}")
    return script_path


# ===== 主流程 =====
def main():
    print()
    print("=" * 70)
    print("  LMM 教育训练材料 - 发布到 GitHub 私有仓库")
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  项目目录: {PROJECT_ROOT}")
    print("=" * 70)
    flush()

    token, username = step1_verify_token()
    clone_url, html_url = step2_create_repo(token, username)
    files = step3_prepare_files()
    ok = step4_push(token, files, username, clone_url)
    step5_generate_fetch_script(username)

    print()
    print("=" * 70)
    if ok:
        print("🎯 全部完成！")
        print(f"  🔗 GitHub 仓库: {html_url}")
        print(f"  📦 已上传 {len(files)} 个目录/文件")
        print(f"  📂 本地目录: {PROJECT_ROOT}")
        print(f"  🛠  训练时拉取数据: python scripts/fetch_github_data.py")
    else:
        print("⚠️  push 阶段失败，其他文件已在本地准备好")
        print("  请检查网络后手动执行:")
        print(f'  cd "{PROJECT_ROOT}"')
        print('  git push -u origin main')
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
