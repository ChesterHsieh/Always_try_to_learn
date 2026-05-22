# Always Try to Learn

多語言學習倉庫，整合多個獨立的學習與研究子專案：資料工程、機器學習、強化學習、可觀測性與軟體工程實踐。

每個子目錄都是獨立專案，擁有自己的 README、環境與相依設定。本檔提供全倉庫導覽；細節請見各子專案的 README。

## 倉庫結構

### 1. ai-monitor-system

以監控為優先的參考框架，協助 DataOps 團隊驗證可觀測性堆疊能否偵測、分類、關聯並告警真實的 PySpark Pipeline 故障。

- **技術棧**：Python、PySpark、Kubernetes、Helm
- **可觀測性堆疊**：Prometheus、Grafana、OpenTelemetry、Grafana Tempo、Marquez（OpenLineage）
- **核心**：刻意簡化的 PySpark Batch Pipeline + 10 個可重現故障情境 + Probe 驅動的驗證框架
- **詳情**：[ai-monitor-system/README.md](ai-monitor-system/README.md)

### 2. heuristic-learning

複現「啟發式學習（Heuristic Learning）」典範——在不訓練神經網路、不靠梯度的前提下，用手寫規則控制 agent，並以環境反饋反覆迭代；在 LunarLander-v3 上與自訓 RL 做硬核對照。

- **技術棧**：Python、JAX、Flax（禁用 TensorFlow / PyTorch）
- **結論**：零訓練的手寫規則 `baseline_v1`（+264.4）與梯度訓練 200 萬步的 `rl_ppo`（+245.8）同級
- **詳情**：[heuristic-learning/README.md](heuristic-learning/README.md)

### 3. lora-image-gen

研究 LoRA（Low-Rank Adaptation）在文生圖／圖生圖模型上的微調與應用，從原理、訓練到推論。

- **技術棧**：Python、diffusers / kohya_ss、ComfyUI
- **執行環境**：ComfyUI client-server 分離架構，RunPod 遠端 GPU + 本機 API
- **目前案例**：Stacklands 風格 LoRA（SDXL backbone）
- **詳情**：[lora-image-gen/README.md](lora-image-gen/README.md)

### 4. learn-jax

JAX 學習與實踐專案，涵蓋基礎運算到強化學習。

- **技術棧**：Python、JAX
- **內容**：JAX 基礎操作與自動微分、矩陣運算、Transformer 實作、`purejaxrl`（純 JAX 端對端 RL）
- **特點**：支援 Apple Silicon Metal 加速
- **使用指南**：[learn-jax/USAGE.md](learn-jax/USAGE.md)、[learn-jax/QUICKSTART.md](learn-jax/QUICKSTART.md)

### 5. streamming_lab

Spark 流處理實驗室——最小可行實驗設置，用於流式事件模擬。

- **技術棧**：Java、Python、Spark
- **內容**：Spark batch processing、streaming syntax 練習
- **原倉庫**：[ChesterHsieh/streamming_lab](https://github.com/ChesterHsieh/streamming_lab)

### 6. DDIA-in-real

《Designing Data-Intensive Applications》的實戰專案。

- **技術棧**：Python
- **內容**：資料產生器、資料攝入模式、資料品質模式
- **包含模式**：Data Ingestion（Append-only、CDC、Idempotent、Transactional、Upsert）、Data Quality、Observability、Schema Evolution、Security and Governance
- **原倉庫**：[ChesterHsieh/DDIA-in-real](https://github.com/ChesterHsieh/DDIA-in-real)

### 7. Data-QA-engineer

資料 QA 工程師工具與流程。

- **技術棧**：Python
- **內容**：資料管道處理、規則驗證、資料品質檢查
- **特點**：包含訂單與產品庫存資料處理範例
- **原倉庫**：[ChesterHsieh/Data-QA-engineer](https://github.com/ChesterHsieh/Data-QA-engineer)

### 8. unit-test-pardigm

單元測試範式與最佳實踐。

- **技術棧**：Python
- **內容**：Clean Architecture 實踐、測試模式與反模式討論
- **主題**：Mock objects 與依賴注入、Setup/Teardown、Repository vs DAO、ORM 討論
- **原倉庫**：[ChesterHsieh/unit-test-pardigm](https://github.com/ChesterHsieh/unit-test-pardigm)

### 9. leetcode-DSA

演算法與資料結構練習。

- **技術棧**：Python
- **內容**：LeetCode 題解與資料結構實作（如 MonoStack、OrderedDict）

## 技術棧概覽

- **語言**：Python、Java
- **框架／工具**：JAX、Flax、Spark、PySpark、Kubernetes、Helm、Prometheus、Grafana、OpenTelemetry、diffusers / kohya_ss、ComfyUI、pytest
- **領域**：
  - 大數據處理與資料工程
  - 機器學習（Transformer、JAX）
  - 強化學習（PPO、Heuristic Learning）
  - 生成式 AI（LoRA 文生圖）
  - 可觀測性與 DataOps
  - 軟體測試與 Clean Architecture

## 開發流程

本倉庫採用 Kiro 風格的 Spec-Driven Development（規格驅動開發）於 agentic SDLC 上：

- 規格與 steering 設定位於 [.kiro/](.kiro/)
- OpenSpec 變更與規格位於 [openspec/](openspec/)

詳見 [CLAUDE.md](CLAUDE.md)。

## 使用說明

每個子目錄都是獨立專案，擁有自己的 README 與相依設定。請查看各子目錄的 README 了解具體環境設定與使用方法（多數 Python 專案以 `uv` 管理虛擬環境）。

## 授權

各子專案保留其原有授權，詳見各子目錄的 LICENSE 檔。

## 作者

Chester Hsieh

## 更新日期

2026-05-22
