# JAX 项目使用指南

## 快速运行方式

### 方式 1: 使用 Makefile (推荐)

在 `jax` 目录下运行：

```bash
cd jax

# 查看所有可用命令
make help

# 运行 basic/main.py
make run-basic

# 运行验证脚本
make run-verify

# 运行 transformer 训练
make run-transformer

# 运行测试
make test
```

### 方式 2: 使用 uv run

```bash
cd jax

# 运行任何 Python 文件
uv run python basic/main.py
uv run python verify_jax.py
uv run python transformer/train.py
```

### 方式 3: 使用便捷脚本

```bash
cd jax

# 运行 basic/main.py
./run_basic.sh
```

### 方式 4: 在 Cursor/VS Code 中运行

1. 打开 `jax/basic/main.py` 文件
2. 按 `F5` 或点击右上角的运行按钮
3. 选择 "Python: JAX Basic Main" 或 "Python: Current File (JAX)"

你也可以：
- 在 Run and Debug 面板中选择预配置的运行选项
- 使用断点进行调试

## 环境设置

首次使用需要安装依赖：

```bash
cd jax
uv sync
```

## 项目结构

```
jax/
├── basic/           # 基础示例
│   └── main.py
├── transformer/     # Transformer 实现
│   ├── model.py
│   ├── train.py
│   └── ...
├── pyproject.toml   # 项目配置和依赖
├── Makefile         # 便捷命令
└── run_basic.sh     # 运行脚本
```

## VS Code/Cursor 配置

配置文件位于 `.vscode/` 目录：
- `settings.json` - Python 环境配置
- `launch.json` - 调试和运行配置

这些配置会自动：
- 使用正确的 Python 解释器 (`.venv/bin/python`)
- 设置正确的工作目录
- 激活虚拟环境
