# Data Engineering Design Patterns

A Python project implementing **Data Engineering Design Patterns** from the book "Data Engineering Design Patterns" (2025) in real-world scenarios.

## 專案說明

這個專案專注於實作《Data Engineering Design Patterns》書中的各種設計模式。模式依照書中的章節主題來組織，而非技術類別（如 batch/streaming/etl）。

### 專案重點

**Data Engineering Design Patterns**

依照書中的主題章節組織：

- **Data Ingestion** (`patterns/data_ingestion/`) ✅ **已完成**
  - Transactional Writer（事務寫入器，確保寫入「全有或全無」）
  - Idempotent Writer（冪等寫入器，確保重複執行不會產生重複數據）
  - Upsert Writer（插入或更新寫入器，原子性插入或更新）
  - Append-Only Writer（僅追加寫入器，不可變日誌）
  - Change Data Capture (CDC)（變更資料捕獲，即時複製變更）
- **Data Quality** (`patterns/data_quality/`)（預留）
- **Updates / Merges** (`patterns/updates/`)（預留）
- **Time & Lateness** (`patterns/time_and_lateness/`)（預留）
- **Schema Evolution** (`patterns/schema_evolution/`)（預留）
- **Security & Governance** (`patterns/security_and_governance/`)（預留）
- **Observability** (`patterns/observability/`)（預留）

**工具模組**

- **Utils** (`patterns/utils/`)
  - `simple_db.py`: 記憶體資料庫模擬器，支援 Transaction，用於模式示範

> 📚 參考資料：`docs/Data_Engineering_Design_Patterns_2025.pdf`

## 環境需求

- Python 3.11+
- uv (Python 套件管理工具)

## 安裝與執行

### 1. 安裝依賴
```bash
uv sync
```

### 2. 執行主要示範
```bash
uv run main.py
```

### 3. 使用統一的 Demo 執行器
```bash
# 執行特定模式示範
uv run scripts/run_demos.py ingestion-tx-writer      # Transactional Writer
uv run scripts/run_demos.py ingestion-idempotent     # Idempotent Writer
uv run scripts/run_demos.py ingestion-upsert         # Upsert Writer
uv run scripts/run_demos.py ingestion-append-only   # Append-Only Writer
uv run scripts/run_demos.py ingestion-cdc            # Change Data Capture

# 執行所有示範
uv run scripts/run_demos.py all
```

## 專案結構

```
Data-Engineering-Design-Patterns/
├── patterns/                      # ⭐ Data Engineering Design Patterns 實作
│   ├── __init__.py
│   ├── data_ingestion/            # Data ingestion patterns（如 Transactional Writer）
│   ├── data_quality/              # Data quality & validation patterns（預留）
│   ├── updates/                   # Updates / merges patterns（預留）
│   ├── time_and_lateness/         # Time, lateness, windowing patterns（預留）
│   ├── schema_evolution/          # Schema evolution & compatibility patterns（預留）
│   ├── security_and_governance/   # Security, PII, partitioning patterns（預留）
│   ├── observability/             # Monitoring, SLAs, WAP patterns（預留）
│   └── utils/                     # 共享工具（如 SimpleDB 用於示範）
│       ├── __init__.py
│       └── simple_db.py           # 記憶體資料庫模擬器（支援 Transaction）
│
├── tests/                         # 測試資料夾
│   ├── __init__.py
│   └── test_patterns/             # Patterns 測試
│       └── data_ingestion/
│
├── scripts/                       # 執行腳本
│   └── run_demos.py               # 統一的 demo 執行器
│
├── docs/                          # 參考文件
│   └── Data_Engineering_Design_Patterns_2025.pdf  # ⭐ 主要參考
│
├── main.py                        # 主入口
├── pyproject.toml                 # 專案配置
├── README.md                      # 專案說明
├── .cursorrules                   # Cursor AI 專案規則
└── LICENSE                        # 授權檔案
```

## 開發指南

### 專案組織

專案採用模組化結構，按書中的主題章節分類：
- **patterns/**: ⭐ **主要實作** - Data Engineering Design Patterns 實作
  - 每個主題一個子目錄（如 `data_ingestion/`, `data_quality/`）
  - `utils/` 包含共享工具（如 SimpleDB 用於示範）
- **tests/**: 測試程式碼
- **scripts/**: 執行腳本和工具

### 常用指令

1. 使用 `uv add <package>` 新增依賴
2. 使用 `uv run <script>` 執行腳本
3. 使用 `uv run scripts/run_demos.py <demo>` 執行特定示範
4. 修改程式碼後重新執行 `uv run main.py`

### 新增設計模式

要新增 Data Engineering Design Pattern（這是專案的主要工作）：

1. **研究模式**：先閱讀 `docs/Data_Engineering_Design_Patterns_2025.pdf` 了解模式
2. **建立實作**：在對應的 `patterns/` 子資料夾中建立新檔案
3. **組織程式碼**：參考現有結構，確保程式碼清晰且可讀
4. **撰寫測試**：在 `tests/test_patterns/` 中新增對應的測試
5. **更新文件**：更新 README.md 和相關文件
6. **執行器**：更新 `scripts/run_demos.py` 加入新的執行選項

每個模式實作應包含：
- 模式名稱和描述
- 解決的問題
- 使用場景
- 實作細節
- 範例程式碼
- 權衡考量（Trade-offs）

> 💡 提示：參考 `.cursorrules` 檔案了解詳細的實作指南

## 學習重點

1. **Data Engineering Design Patterns**: 學習書中各種模式的實作與應用
2. **模式組織**: 理解如何按主題而非技術類別來組織模式
3. **實作範例**: 透過實際程式碼理解每個模式的運作方式
4. **測試與驗證**: 確保模式實作的正確性與可靠性

## 授權

請查看 LICENSE 檔案了解授權詳情。