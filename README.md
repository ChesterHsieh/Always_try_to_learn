# Always Try to Learn

多语言学习仓库，整合了多个项目的代码和学习资料。

## 仓库结构

本仓库包含以下子项目：

### 1. streamming_lab
Spark 流处理实验室 - 最小可行实验设置，用于流式事件模拟
- **技术栈**: Java, Python, Spark
- **内容**: Spark batch processing, streaming syntax exercises
- **原仓库**: https://github.com/ChesterHsieh/streamming_lab

### 2. from-zero-to-smallest-transformer
从零开始构建 Transformer 模型（Rust 实现）
- **技术栈**: Rust
- **内容**: 纯 Rust 实现的 Transformer 架构，无第三方依赖
- **特点**: 包含矩阵运算、注意力机制、层归一化、前馈网络等完整实现
- **原仓库**: https://github.com/ChesterHsieh/from-zero-to-smallest-transformer

### 3. DDIA-in-real
设计数据密集型应用的实战项目
- **技术栈**: Python
- **内容**: 数据生成器、数据摄取模式、数据质量模式
- **包含模式**: 
  - Data Ingestion (Append-only, CDC, Idempotent, Transactional, Upsert)
  - Data Quality
  - Observability
  - Schema Evolution
  - Security and Governance
- **原仓库**: https://github.com/ChesterHsieh/DDIA-in-real

### 4. Data-QA-engineer
数据 QA 工程师工具和流程
- **技术栈**: Python
- **内容**: 数据管道处理、规则验证、数据质量检查
- **特点**: 包含订单和产品库存数据处理示例
- **原仓库**: https://github.com/ChesterHsieh/Data-QA-engineer

### 5. unit-test-pardigm
单元测试范式和最佳实践
- **技术栈**: Python
- **内容**: Clean Architecture 实践、测试模式、反模式讨论
- **主题**: 
  - Mock objects 和依赖注入
  - Setup/Teardown
  - Repository vs DAO
  - ORM 讨论
- **原仓库**: https://github.com/ChesterHsieh/unit-test-pardigm

### 6. jax
JAX 学习和实践项目
- **技术栈**: Python, JAX
- **内容**: JAX 基础示例、矩阵运算、Transformer 实现
- **特点**: 
  - 基础 JAX 操作和自动微分
  - 矩阵运算示例
  - Transformer 模型实现

## 技术栈概览

- **语言**: Python, Rust, Java
- **框架/工具**: Spark, FastAPI, pytest, JAX
- **领域**: 
  - 大数据处理
  - 机器学习 (Transformer, JAX)
  - 数据工程
  - 软件测试
  - Clean Architecture

## 使用说明

每个子目录都是一个独立的项目，具有自己的 README 和依赖配置。请查看各子目录的 README 文件了解具体使用方法。

## 许可证

各子项目保留其原有的许可证，详见各子目录的 LICENSE 文件。

## 作者

Chester Hsieh

## 更新日期

2026-02-06
