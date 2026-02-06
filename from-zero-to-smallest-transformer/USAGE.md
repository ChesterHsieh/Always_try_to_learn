# Rust Transformer 项目使用指南

## 快速运行方式

### 方式 1: 使用 Makefile (推荐)

在 `from-zero-to-smallest-transformer` 目录下运行：

```bash
cd from-zero-to-smallest-transformer

# 查看所有可用命令
make help

# 运行项目
make run

# 编译项目
make build

# 运行测试
make test

# 代码检查
make check
make clippy

# 格式化代码
make fmt

# 编译并运行优化版本 (更快)
make release
```

### 方式 2: 使用 Cargo 直接运行

```bash
cd from-zero-to-smallest-transformer

# 运行
cargo run

# 运行优化版本
cargo run --release

# 运行测试
cargo test

# 检查代码
cargo check
cargo clippy

# 格式化
cargo fmt
```

### 方式 3: 在 Cursor/VS Code 中运行

1. 打开 `from-zero-to-smallest-transformer/src/main.rs` 文件
2. 按 `F5` 或点击右上角的运行按钮
3. 选择以下配置之一：
   - **Rust: Run Transformer** - 运行 debug 版本
   - **Rust: Run Transformer (Release)** - 运行优化版本
   - **Rust: Test Transformer** - 运行测试

你也可以：
- 在 Run and Debug 面板中选择预配置的运行选项
- 使用断点进行调试

### 方式 4: 直接运行编译后的二进制文件

```bash
# 先编译
cargo build

# 运行 debug 版本
./target/debug/transformer

# 或编译并运行 release 版本
cargo build --release
./target/release/transformer
```

## 项目结构

```
from-zero-to-smallest-transformer/
├── src/
│   ├── main.rs              # 主程序入口
│   ├── math/                # 数学运算模块
│   │   ├── vector.rs        # 向量运算
│   │   ├── matrix.rs        # 矩阵运算
│   │   └── activation.rs    # 激活函数
│   ├── nn/                  # 神经网络层
│   │   ├── linear.rs        # 线性层
│   │   ├── attention.rs     # 注意力机制
│   │   ├── feed_forward.rs  # 前馈网络
│   │   └── layer_norm.rs    # 层归一化
│   ├── transformer/         # Transformer 模块
│   │   ├── encoder.rs       # 编码器
│   │   ├── decoder.rs       # 解码器
│   │   └── transformer.rs   # 完整模型
│   └── training/            # 训练相关
│       ├── loss.rs          # 损失函数
│       ├── optimizer.rs     # 优化器
│       └── trainer.rs       # 训练循环
├── Cargo.toml               # 项目配置
└── Makefile                 # 便捷命令

```

## VS Code/Cursor 配置

配置文件位于 `.vscode/` 目录：
- `settings.json` - Rust 和 Python 环境配置
- `launch.json` - 调试和运行配置（包含 Rust 和 Python）
- `tasks.json` - Cargo 构建任务

## 性能提示

- **开发时使用**: `cargo run` 或 `make run` (编译快，运行慢)
- **测试性能时使用**: `cargo run --release` 或 `make release` (编译慢，运行快)

## 与 Python (JAX) 项目共存

这个 Rust 项目和 JAX Python 项目可以完美共存，不会有任何冲突：
- Python 使用 `debugpy` 调试器
- Rust 使用 `lldb` 调试器
- 两者有独立的配置和构建系统
- 可以在同一个 workspace 中同时开发
