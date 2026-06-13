# LMM Training Dockerfile (CPU Version)
# 基于 Python 3.10 slim 镜像，适配天虹主机无 GPU 环境

FROM python:3.10-slim

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖（python 已由基础镜像提供）
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    tmux \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（CPU 版本 PyTorch）
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# 复制整个项目
COPY . .

# 创建必要目录
RUN mkdir -p /workspace/data /workspace/models /workspace/logs /workspace/experiments

# 设置 Python 路径
ENV PYTHONPATH="/workspace"

# 设置默认命令
CMD ["/bin/bash"]