#!/bin/bash
# 快速运行 jax/basic/main.py 的脚本

# 切换到 jax 目录
cd "$(dirname "$0")"

# 激活虚拟环境并运行
if [ -d ".venv" ]; then
    source .venv/bin/activate
    python basic/main.py
else
    echo "虚拟环境不存在，请先运行: uv sync"
    exit 1
fi
