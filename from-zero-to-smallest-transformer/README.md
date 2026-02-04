# From Zero to Smallest Transformer

Building a transformer model from scratch in Rust without any third-party dependencies.

## 项目目标 (Project Goals)

从零开始构建一个完整的 Transformer 模型，包括：
- 基础的数学运算（矩阵、向量、激活函数）
- 神经网络层（线性层、注意力机制、层归一化）
- Transformer 架构（编码器、解码器）
- 训练循环（损失函数、优化器）

所有实现仅使用 Rust 标准库，不依赖任何第三方库。

## 项目结构 (Project Structure)

```
src/
├── main.rs                 # 主入口
├── math/                   # 数学运算模块
│   ├── mod.rs
│   ├── matrix.rs          # 矩阵运算
│   ├── vector.rs          # 向量运算
│   └── activation.rs      # 激活函数（ReLU, GELU, Softmax等）
├── nn/                     # 神经网络层
│   ├── mod.rs
│   ├── linear.rs          # 线性层（全连接层）
│   ├── attention.rs       # 多头自注意力机制
│   ├── layer_norm.rs      # 层归一化
│   └── feed_forward.rs    # 前馈网络
├── transformer/            # Transformer 架构
│   ├── mod.rs
│   ├── encoder.rs         # 编码器块
│   ├── decoder.rs         # 解码器块
│   └── transformer.rs     # 完整 Transformer 模型
└── training/               # 训练相关
    ├── mod.rs
    ├── loss.rs            # 损失函数（交叉熵、MSE）
    ├── optimizer.rs       # 优化器（SGD、Adam）
    └── trainer.rs         # 训练器
```

## 核心组件 (Core Components)

### 1. 数学运算 (Math Operations)
- **Matrix**: 矩阵运算（乘法、加法、转置等）
- **Vector**: 向量运算（点积、加法等）
- **Activation Functions**: ReLU, GELU, Softmax, Sigmoid

### 2. 神经网络层 (Neural Network Layers)
- **Linear**: 全连接层
- **MultiHeadAttention**: 多头自注意力机制
- **LayerNorm**: 层归一化
- **FeedForward**: 前馈网络（两层线性层 + GELU）

### 3. Transformer 架构
- **EncoderBlock**: Transformer 编码器块（自注意力 + 前馈网络 + 残差连接）
- **DecoderBlock**: Transformer 解码器块（自注意力 + 交叉注意力 + 前馈网络）
- **Transformer**: 完整的 Transformer 模型

### 4. 训练组件
- **CrossEntropyLoss**: 交叉熵损失函数
- **MSELoss**: 均方误差损失函数
- **SGD**: 随机梯度下降优化器
- **Adam**: Adam 优化器（简化版）
- **Trainer**: 训练器封装

## 使用方法 (Usage)

### 构建项目
```bash
cargo build
```

### 运行
```bash
cargo run
```

### 运行测试
```bash
cargo test
```

## 实现状态 (Implementation Status)

### ✅ 已完成
- [x] 基础矩阵和向量运算
- [x] 激活函数实现
- [x] 线性层
- [x] 多头自注意力机制
- [x] 层归一化
- [x] 前馈网络
- [x] Transformer 编码器和解码器块
- [x] 损失函数（交叉熵、MSE）
- [x] 优化器（SGD、简化版 Adam）

### 🚧 待完成
- [ ] Token 嵌入层（Embedding）
- [ ] 位置编码（Positional Encoding）
- [ ] 完整的反向传播实现
- [ ] 梯度累积和更新
- [ ] 数据加载器
- [ ] 训练循环完善
- [ ] 模型保存和加载
- [ ] 评估指标（准确率等）

## 设计原则 (Design Principles)

1. **零依赖**: 仅使用 Rust 标准库
2. **教育性**: 代码清晰，注释详细，便于理解 Transformer 原理
3. **模块化**: 每个组件独立，易于测试和扩展
4. **简洁性**: 优先实现核心功能，避免过度工程化

## 学习资源 (Learning Resources)

这个项目旨在帮助理解 Transformer 架构的核心原理：
- Attention Is All You Need (Vaswani et al., 2017)
- The Illustrated Transformer (Jay Alammar)
- 各种从零实现 Transformer 的教程

## 许可证 (License)

MIT License

## 贡献 (Contributing)

欢迎提交 Issue 和 Pull Request！
