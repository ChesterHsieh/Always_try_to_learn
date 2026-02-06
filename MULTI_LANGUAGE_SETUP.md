# 多语言项目配置说明

这个 workspace 包含了 **Python (JAX)** 和 **Rust (Transformer)** 两个项目，它们可以完美共存，互不干扰。

## 🎯 项目结构

```
Always_try_to_learn/
├── .vscode/
│   ├── settings.json    # Python + Rust 通用设置
│   ├── launch.json      # Python + Rust 调试配置
│   └── tasks.json       # Rust 构建任务
├── jax/                 # Python JAX 项目
│   ├── basic/
│   ├── transformer/
│   ├── pyproject.toml
│   ├── Makefile
│   └── USAGE.md
└── from-zero-to-smallest-transformer/  # Rust 项目
    ├── src/
    ├── Cargo.toml
    ├── Makefile
    └── USAGE.md
```

## ✅ 为什么不会冲突？

### 1. **不同的调试器类型**
- **Python**: 使用 `debugpy` 调试器
- **Rust**: 使用 `lldb` 调试器
- 两者完全独立，不会互相干扰

### 2. **独立的构建系统**
- **Python**: 使用 `uv` 和 `pip`
- **Rust**: 使用 `cargo`
- 各自管理自己的依赖和构建

### 3. **独立的虚拟环境**
- **Python**: `.venv` 在 `jax/` 目录下
- **Rust**: 编译产物在 `target/` 目录下
- 互不影响

### 4. **明确的工作目录**
每个配置都指定了自己的 `cwd` (工作目录)：
- Python 配置: `${workspaceFolder}/jax`
- Rust 配置: `${workspaceFolder}/from-zero-to-smallest-transformer`

## 🚀 如何使用

### 在 Cursor/VS Code 中运行

按 `F5` 或点击运行按钮，会出现以下选项：

#### Python 选项:
- **Python: JAX Basic Main** - 运行 `jax/basic/main.py`
- **Python: Current File (JAX)** - 运行当前打开的 Python 文件
- **Python: JAX Verify** - 验证 JAX 安装
- **Python: JAX Transformer Train** - 训练 Transformer

#### Rust 选项:
- **Rust: Run Transformer** - 运行 Rust Transformer (debug)
- **Rust: Run Transformer (Release)** - 运行优化版本
- **Rust: Test Transformer** - 运行 Rust 测试

### 在 Terminal 中运行

#### Python 项目:
```bash
cd jax
make run-basic      # 运行 basic/main.py
make run-verify     # 验证安装
make test          # 运行测试
```

#### Rust 项目:
```bash
cd from-zero-to-smallest-transformer
make run           # 运行项目
make test          # 运行测试
make release       # 编译并运行优化版本
```

## 📝 配置文件说明

### `.vscode/settings.json`
包含了 Python 和 Rust 的编辑器配置：
- Python 解释器路径
- Rust analyzer 设置
- 格式化选项
- 保存时自动格式化

### `.vscode/launch.json`
包含了所有的运行和调试配置，按语言分组：
- Python 配置使用 `type: "debugpy"`
- Rust 配置使用 `type: "lldb"`

### `.vscode/tasks.json`
包含了 Rust 的 Cargo 任务：
- build
- test
- check
- clippy
- run

## 🎨 编辑器体验

### Python 文件 (`.py`)
- 自动使用 `jax/.venv` 中的 Python 解释器
- 保存时自动格式化
- 自动 import 排序
- pytest 测试支持

### Rust 文件 (`.rs`)
- rust-analyzer 提供智能提示
- 保存时自动格式化 (rustfmt)
- clippy 代码检查
- 内联类型提示

## 💡 推荐的 VS Code 扩展

### Python 开发:
- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)

### Rust 开发:
- rust-analyzer (rust-lang.rust-analyzer)
- CodeLLDB (vadimcn.vscode-lldb) - 用于调试

## 🔧 故障排除

### Python 找不到模块
```bash
cd jax
uv sync  # 重新安装依赖
```

### Rust 编译错误
```bash
cd from-zero-to-smallest-transformer
cargo clean
cargo build
```

### 调试器无法启动
- **Python**: 确保安装了 Python 扩展
- **Rust**: 确保安装了 CodeLLDB 扩展

## 📚 更多信息

- Python 项目详情: `jax/USAGE.md`
- Rust 项目详情: `from-zero-to-smallest-transformer/USAGE.md`

## ✨ 总结

这个配置允许你：
- ✅ 同时开发 Python 和 Rust 项目
- ✅ 使用 F5 快速运行任何项目
- ✅ 独立的环境和依赖管理
- ✅ 统一的编辑器体验
- ✅ 零冲突，完美共存
