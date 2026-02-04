# Data Ingestion Patterns

本目錄包含來自《Data Engineering Design Patterns》(2025) 第二章「Data Ingestion」的所有模式實現。

## 已實現的模式

### 1. Transactional Writer（事務寫入器）

**問題**：寫入目標資料集時，如果作業中途失敗，可能會產生部分或不一致的寫入。下游讀取者可能觀察到半寫入的分區或表。

**解決方案**：使用事務寫入器，將所有變更暫存並原子性提交。要麼所有變更都可見，要麼都不可見。

**使用場景**：
- 將每日批次資料寫入倉庫表
- 從外部 API 擷取資料到湖倉
- 在多步驟管道中更新衍生表

**檔案**：`transactional_writer.py`

**Demo**：
```bash
uv run scripts/run_demos.py ingestion-tx-writer
```

**Data Generator 配置**：`data_generator/examples/transactional_writer_config.yaml`

---

### 2. Idempotent Writer（冪等寫入器）

**問題**：如果資料擷取作業失敗並重試，可能會將重複記錄寫入目標資料集，導致資料品質問題和下游聚合錯誤。

**解決方案**：使用冪等寫入器，在插入前檢查記錄是否已寫入。通常使用唯一識別碼（如記錄 ID、雜湊或複合鍵）來檢測重複。

**使用場景**：
- 重試失敗的批次擷取作業
- 多次處理相同的來源資料
- 在分散式系統中確保 exactly-once 語義
- 處理網路重試和部分失敗

**檔案**：`idempotent_writer.py`

**Demo**：
```bash
uv run scripts/run_demos.py ingestion-idempotent
```

**Data Generator 配置**：`data_generator/examples/idempotent_writer_config.yaml`

---

### 3. Upsert Writer（插入或更新寫入器）

**問題**：擷取資料時，可能需要根據唯一鍵插入新記錄或更新現有記錄。簡單的插入操作在遇到重複時會失敗，而分離的插入/更新邏輯容易出錯。

**解決方案**：使用 upsert 寫入器，原子性地執行「插入或更新」操作。如果給定鍵的記錄存在，則更新它；否則，插入它。

**使用場景**：
- 從外部系統同步資料，其中記錄可能隨時間變化
- 維護資料倉庫中的維度表
- 更新用戶檔案或帳戶資訊
- 處理緩慢變化維度（SCD Type 1）

**檔案**：`upsert_writer.py`

**Demo**：
```bash
uv run scripts/run_demos.py ingestion-upsert
```

**Data Generator 配置**：`data_generator/examples/upsert_writer_config.yaml`

---

### 4. Append-Only Writer（僅追加寫入器）

**問題**：某些資料系統需要不可變的、僅追加的日誌，其中記錄永遠不會更新或刪除。這提供了審計追蹤並支援時間旅行查詢，但需要仔細處理以避免重複並確保順序。

**解決方案**：使用僅追加寫入器，只將新記錄添加到資料集末尾。記錄通常會加上時間戳，並可能包含序列號以維持順序。可以使用記錄 ID 結合時間戳來進行重複檢測。

**使用場景**：
- 事件日誌和審計追蹤
- 時間序列資料擷取
- 不可變資料湖
- 變更資料捕獲（CDC）事件流
- 金融交易日誌

**檔案**：`append_only_writer.py`

**Demo**：
```bash
uv run scripts/run_demos.py ingestion-append-only
```

**Data Generator 配置**：`data_generator/examples/append_only_writer_config.yaml`

---

### 5. Change Data Capture (CDC)（變更資料捕獲）

**問題**：您需要從源系統捕獲並複製變更（插入、更新、刪除）到目標系統，接近即時。輪詢變更效率低下，您需要確保一致性和順序。

**解決方案**：使用變更資料捕獲（CDC）來捕獲變更，通常通過讀取事務日誌或使用資料庫觸發器。變更表示為事件（插入/更新/刪除）並流式傳輸到目標。

**使用場景**：
- 資料庫之間的即時資料複製
- 保持資料倉庫與操作資料庫同步
- 事件溯源和事件驅動架構
- 構建即時分析管道
- 維護所有變更的審計追蹤

**檔案**：`change_data_capture.py`

**Demo**：
```bash
uv run scripts/run_demos.py ingestion-cdc
```

**Data Generator 配置**：`data_generator/examples/change_data_capture_config.yaml`

---

## 執行所有 Demo

要執行所有資料擷取模式的 demo：

```bash
uv run scripts/run_demos.py all
```

## 使用 Data Generator

每個模式都包含一個 data generator 配置範例，位於 `data_generator/examples/` 目錄。您可以使用這些配置來生成測試資料：

```bash
uv run data_generator/generate_dataset.py --config_file data_generator/examples/idempotent_writer_config.yaml
```

## 實作細節

所有模式都使用 `patterns.utils.simple_db.SimpleDB` 作為演示用的記憶體資料庫。在實際應用中，這些模式應該適配到真實的資料庫系統（如 PostgreSQL、MySQL、BigQuery 等）。

每個模式都包含：
- 清晰的模式描述和問題說明
- 協議定義（Protocol）用於抽象化
- SimpleDB 實現用於演示
- 高階 Writer 類別封裝 ETL 流程
- Demo 函數展示模式的使用

## 參考資料

- Data Engineering Design Patterns (2025), Chapter 2: Data Ingestion

