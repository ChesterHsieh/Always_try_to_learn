.PHONY: help run-basic run-verify run-transformer setup test clean

help:
	@echo "可用的命令："
	@echo "  make setup          - 安装依赖 (使用 uv sync)"
	@echo "  make run-basic      - 运行 basic/main.py"
	@echo "  make run-verify     - 运行 verify_jax.py"
	@echo "  make run-transformer - 运行 transformer/train.py"
	@echo "  make test           - 运行测试"
	@echo "  make clean          - 清理缓存文件"

setup:
	uv sync

run-basic:
	uv run python basic/main.py

run-verify:
	uv run python verify_jax.py

run-transformer:
	uv run python transformer/train.py

test:
	uv run pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
