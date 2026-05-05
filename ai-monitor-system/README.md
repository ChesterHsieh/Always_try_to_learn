# AI Monitor System

**一個以監控為優先的參考框架，協助 DataOps 團隊驗證可觀測性工具是否能有效應對真實的 PySpark Pipeline 故障。**

本專案是一個可直接執行的最佳實踐範本，回答一個核心問題：
**「我的可觀測性堆疊，真的能偵測、分類、關聯並告警真實的 Pipeline 故障嗎？」**

專案內含一個刻意設計為簡單的 PySpark Batch Pipeline，搭配 10 個可重現的故障情境，以及一個 Probe 驗證框架，用以斷言監控堆疊對每個故障的反應是否正確。可直接使用來評估堆疊，或 Fork 作為自有 Pipeline 監控基準的起點。

---

## 提供的能力

| 能力 | 實現方式 |
|---|---|
| **本地 Kubernetes 上的端對端可觀測性堆疊** | 上游 Helm Charts（Prometheus / Grafana / OpenTelemetry Collector / Marquez / Tempo）整合至單一 Chart，並附合理預設值 |
| **跨指標 / 追蹤 / 血緣的 Run 層級關聯** | 共用 `run_id`，傳播至 Prometheus Exemplars、OTel Span Attributes 與 OpenLineage Events |
| **10 個可重現的故障情境** | `scenarios/*.yaml` — 每個情境宣告預期的分類、告警與 Probe；可單獨執行或批次執行 |
| **Probe 驅動驗證** | `scripts/probe.py` 查詢 Prometheus / Tempo / Marquez，並輸出單行 PASS/FAIL 結論與提示 |
| **分層測試套件** | Contract（≤ 5 秒，無需叢集）→ Integration（Stub 叢集）→ Smoke（實際叢集） |
| **Coverage CLI** | `utils.coverage` 產生 JSON 報告，將 Chart 版本、告警規則、儀表板與血緣狀態整合為單一發布驗收成品 |
| **運維 Runbook** | 每個故障類別的症狀、重現指令、預期告警與處置路徑 |

---

## 必要堆疊

| 元件 | 角色 | 預設版本 |
|---|---|---|
| **OpenLineage** | Pipeline 端血緣發送（Spark Listener + Shadow Emitter） | 內建於 Pipeline Image |
| **Marquez** | OpenLineage Backend；可查詢 Run 狀態與資料集版本 | upstream chart `6.7.0` |
| **Prometheus** | 指標收集與告警評估 | upstream chart `25.27.0` |
| **OpenTelemetry Collector** | 追蹤攝入 → Tempo | upstream chart `0.78.0` |
| **Grafana Tempo** | 追蹤儲存與查詢 | upstream chart |
| **Grafana** | 儀表板（`pipeline-health`、`lineage-view`）與告警視覺化 | upstream chart `8.5.1` |
| **Helm 3** | 部署協調器 | 由主機提供 |
| **本地 Kubernetes** | kind / minikube / Rancher Desktop / Docker Desktop K8s | 由主機提供 |

> 此 Chart 以設定為優先。專案自有的 Template 極少 —
> 幾乎所有設定皆來自上游 Chart Values。詳見
> [docs/chart-version-matrix.md](docs/chart-version-matrix.md)。

---

## 專案結構

### 核心模組（Pipeline Logic）

| 路徑 | 用途 |
|---|---|
| [`pipeline/`](pipeline/) | **核心業務邏輯與編排** — Job 執行器 + 故障分類器 + 故障注入 + 執行時上下文 |
| [`telemetry/`](telemetry/) | **可觀測性工具集** — Prometheus 指標 + OpenTelemetry 追蹤 + OpenLineage 血緣 + Grafana 設定 |
| [`utils/`](utils/) | **共用工具** — 檔案 I/O 適配器 + 情境 Schema 驗證 + 監控覆蓋檢查 |

### 測試與部署

| 路徑 | 用途 |
|---|---|
| [`scenarios/`](scenarios/) | 宣告式故障情境 YAML 檔，由執行器讀取 |
| [`scripts/`](scripts/) | `run-scenario.sh`、`run-all-failure-scenarios.sh`、`probe.py` |
| [`deploy/helm/`](deploy/helm/) | Helm Chart、上游 Chart 相依、Values 覆蓋、專案膠合 Template |
| [`deploy/scripts/`](deploy/scripts/) | `bootstrap-local.sh`、`run-pipeline.sh`、`run-smoke-test.sh`、`check-monitoring-coverage.sh`、`nuke-local.sh` |
| [`monitoring/`](monitoring/) | Grafana 儀表板 JSON + Prometheus 告警規則 YAML（透過 Chart 掛載） |
| [`tests/contract/`](tests/contract/) | 純 Python 不變量（無需叢集）— 每次變更皆執行 |
| [`tests/integration/`](tests/integration/) | 使用 Stub 叢集元件的配線測試 |
| [`tests/smoke/`](tests/smoke/) | 針對實際本地叢集的端對端 Smoke 測試 |
| [`docs/`](docs/) | Onboarding、Runbook（每類別處置） |

---

## 快速開始

```bash
# 1. 啟動堆疊（建立 Namespace、建置 Pipeline Image、Helm Install）
./deploy/scripts/bootstrap-local.sh

# 2. 觸發一次正常執行
./deploy/scripts/run-pipeline.sh

# 3. 驗證堆疊確實觀察到了執行結果
./scripts/run-scenario.sh success-baseline

# 4. 執行所有故障情境並更新驗證帳本
./scripts/run-all-failure-scenarios.sh --update-report
```

Namespace 預設為 `ai-monitor-system`；Release Name 預設為 `monitor`。

### 本地 UI（NodePort — 無需 port-forward）

| 服務 | URL |
|---|---|
| Grafana | http://localhost:30300 |
| Prometheus | http://localhost:30090 |
| Marquez Web | http://localhost:30444 |
| Marquez API | http://localhost:30555 |
| Grafana Tempo | http://localhost:30318 |

---

## 故障情境目錄

每個情境是一個 YAML 檔，宣告（a）要注入的故障、（b）預期的生命週期結果、（c）必須觸發的告警，以及（d）證明堆疊偵測到故障的 Probe 查詢。使用 `./scripts/run-scenario.sh <name>` 單獨執行。

| 情境 | 類別 | 測試偵測對象 |
|---|---|---|
| [`success-baseline.yaml`](scenarios/success-baseline.yaml) | _(成功)_ | 健康執行基準；無告警觸發 |
| [`input-not-found.yaml`](scenarios/input-not-found.yaml) | `input_not_found` | 輸入缺失導致的 `FileNotFoundError` |
| [`invalid-path.yaml`](scenarios/invalid-path.yaml) | `invalid_path` | 路徑設定錯誤導致的 `IsADirectoryError` |
| [`permission-denied.yaml`](scenarios/permission-denied.yaml) | `permission_denied` | RBAC / 掛載導致的 `PermissionError` |
| [`spark-task-failed.yaml`](scenarios/spark-task-failed.yaml) | `spark_task_failed` | Py4JJavaError Task 失敗 |
| [`spark-driver-error.yaml`](scenarios/spark-driver-error.yaml) | `spark_driver_error` | Driver 端 SparkException |
| [`schema-drift.yaml`](scenarios/schema-drift.yaml) | `spark_driver_error` | **兩次執行的 Schema Drift** — 基準執行寫入 `value: STRING`，Drift 執行以不匹配 Schema 讀取 → `AnalysisException`（`UNRESOLVED_COLUMN`）；驗證 Marquez 在 OpenLineage Spark Listener 的 Plan 階段盲點下仍能記錄 `state=FAILED` 並附帶 `errorMessage` facet |
| [`lineage-emission-failed.yaml`](scenarios/lineage-emission-failed.yaml) | `lineage_emission_failed` | Marquez 無法連線 / OpenLineage 錯誤 |
| [`telemetry-unavailable.yaml`](scenarios/telemetry-unavailable.yaml) | `telemetry_unavailable` | OTel Collector / Prometheus 無法連線 |
| [`timeout.yaml`](scenarios/timeout.yaml) | `timeout` | `TimeoutError` / `socket.timeout` |
| [`runtime-error.yaml`](scenarios/runtime-error.yaml) | `runtime_error` | 未分類例外的 Catch-all |

每個情境的 Probe 同時斷言**三條訊號路徑**：

1. Prometheus 指標標籤（`pipeline_failures_total{failure_category="..."}`）
2. 告警狀態（`ALERTS{alertname="...",alertstate="firing"}`）
3. 血緣 / 追蹤關聯（適用時）

> **Schema Drift 情境的特殊性**：PySpark Plan 分析器故障（例如 `UNRESOLVED_COLUMN`）在 Spark 啟動 Job 之前就觸發 `AnalysisException`，因此 OpenLineage Spark Listener 永遠觀察不到這次執行。Pipeline 因此從 [`telemetry/lineage_emitter.py`](telemetry/lineage_emitter.py) 發送 Shadow `START`+`FAIL` OpenLineage Event，使用自身的 `run_id` — 以保留三路關聯。這是所有「Spark 引擎啟動前即失敗」類型 Bug 的通用模式。

### 情境檔案結構

```yaml
name: input-not-found
description: Input file missing; pipeline raises FileNotFoundError
pipeline:
  input_records: 0
  inject_failure: input_not_found     # pipeline.failure_injection.SUPPORTED_INJECTIONS 之一
  schema_version: v1                  # 選填 — 驅動 schema-drift 模式
  pre_runs:                           # 選填 — 多次執行情境（基準 → Drift）
    - schema_version: v1
      inject_failure: none
expected_run_status: failed           # succeeded | failed
expected_failure_category: input_not_found  # KNOWN_CATEGORIES 之一，或 null
expected_alerts:
  - PipelineRunFailed
probes:                               # 針對實際監控堆疊驗證
  - id: failure_category_metric
    cmd: prom-query
    args:
      query: 'pipeline_failures_total{failure_category="input_not_found"}'
      gte: 1
      within: 60
  - id: alert_firing
    cmd: prom-query
    args:
      query: 'ALERTS{alertname="PipelineRunFailed",alertstate="firing"}'
      gte: 1
      within: 60
```

### Probes

定義於 [`scripts/probe.py`](scripts/probe.py)。可內嵌於情境中或從 CLI 單次執行。

| `cmd` | Backend | 斷言內容 |
|---|---|---|
| `prom-query` | Prometheus | PromQL 表達式在指定時間窗口內評估為 ≥ N / ≤ N / == N |
| `otel-trace` | Tempo | 來自指定服務的最新追蹤存在，且包含指定屬性（後綴匹配，`--has-attr run_id` 可匹配 `pipeline.run_id`） |
| `lineage-run-state` | Marquez (OpenLineage) | 指定 `run_id` 在時間窗口內達到目標狀態（例如 `FAILED`） |

每個 Probe 輸出單行 JSON 結論，包含 `verdict`、`actual`、`latency_ms`，以及失敗時的 `hint`。

---

## 測試層次

| 層次 | 執行時機 | 時間 | 需要叢集？ | 斷言內容 |
|---|---|---|---|---|
| **Contract**（[`tests/contract/`](tests/contract/)） | 每次程式碼變更 | < 5 秒 | 否 | 型別層級不變量：KNOWN_CATEGORIES 穩定、情境 YAML Schema 有效、分類器確定性、指標標籤凍結、情境 / Runbook / 告警三路對齊 |
| **Integration**（[`tests/integration/`](tests/integration/)） | 宣告程式碼變更完成前 | ~10 秒 | 否（Stub） | 配線：故障注入 → 指標累加、OTel Span 屬性設置、告警 YAML 結構、生命週期 Payload 形狀、血緣 Run 狀態 Probe 行為 |
| **Smoke**（[`tests/smoke/`](tests/smoke/)） | 提交 Pipeline / Helm / Script 變更前 | 數分鐘 | **是** | 端對端：啟動 → Pipeline → Probe → Coverage；Nuke + 重建冪等性 |
| **Live 情境執行框架**（`./scripts/run-all-failure-scenarios.sh`） | 發布驗收；定期監控健康檢查 | 數分鐘 | **是** | 每個故障類別在真實 Marquez / Prometheus / Tempo 中產生對應的指標標籤、告警與血緣狀態 |

```bash
# 內迴圈（最快）
uv run ruff format . && uv run ruff check . --fix
uv run pytest -q tests/contract

# 提交前
uv run pytest -q tests/contract tests/integration

# 發布前
./deploy/scripts/run-smoke-test.sh
./scripts/run-all-failure-scenarios.sh --update-report
./deploy/scripts/check-monitoring-coverage.sh
```

---

## Coverage CLI — 發布驗收成品

```bash
python -m pipeline.coverage \
  --namespace ai-monitor-system \
  --marquez-url http://ai-monitor-system-upstream-marquez:9555 \
  --prometheus-url http://ai-monitor-system-upstream-prometheus-server:80 \
  --grafana-url http://ai-monitor-system-upstream-grafana:80 \
  --output .local-data/coverage/release.json
```

| 退出碼 | 含義 |
|---|---|
| `0` | 所有檢查通過 |
| `1` | 警告（血緣過期、Datasource 延遲） |
| `2` | 嚴重（Prometheus 無法連線、規則未載入、Grafana / Marquez 離線） |

JSON 報告包含所有四個上游 Chart 的版本、各驗證檢查的結果（每項 `pass`/`warn`/`fail` 含細節）以及 `last_verified_at` — 設計上應於每次發布時歸檔。

---

## 調整為自有 Pipeline

1. **替換 [`pipeline/job.py`](pipeline/job.py)** 為你的 Spark Job。保留生命週期封裝 — `record_run_started`、`record_run_succeeded` / `record_run_failed`、`start_run_span`、`maybe_shadow_emit` — 以確保三條訊號路徑共用同一個 `run_id`。
2. **直接重用 [`pipeline/failure_classifier.py`](pipeline/failure_classifier.py)**。9 個類別涵蓋大多數通用 Batch 故障模式；僅在某個類別需要獨立告警路由時才擴展 `KNOWN_CATEGORIES`。
3. **為你的團隊關注的故障模式撰寫情境** — YAML Schema 定義於 [`pipeline/scenario_schema.py`](pipeline/scenario_schema.py)。
4. **在 CI 中針對臨時本地叢集執行框架**（`./scripts/run-all-failure-scenarios.sh --update-report`），並將任何未全數通過的結果視為發布阻斷條件。
5. **自訂告警前閱讀 [`docs/runbook.md`](docs/runbook.md)** — 告警 / 儀表板 / Runbook 三路對齊由 [`tests/contract/test_coverage_alignment_contract.py`](tests/contract/test_coverage_alignment_contract.py) 強制執行，新增項目必須同步更新三者。

---

## 設計原則（為何如此設計）

- **設定優先整合。** 上游 Helm Charts 是堆疊元件的事實來源；專案 Template 僅為膠合層。
- **訊號間共用單一 `run_id`。** Pipeline 產生一個 UUID；指標 Exemplars、OTel Span Attributes、OpenLineage Events 全部帶有此 ID。這是讓告警 → 追蹤 → 血緣三跳處置得以運作的關鍵。
- **基數管控。** `run_id` 與 `failure_message` 存於 Prometheus _Exemplars_，絕不進入指標標籤 — 確保 `pipeline_failures_total` 的基數隨情境增長仍受控。
- **Probe 斷言人類真正關心的事。** Probe 查詢實際 Backend 而非內部 Mock，因此綠燈執行是堆疊能在生產中捕捉該故障的真實佐證。
- **三路對齊是一份契約。** 一個故障類別必須同時存在於 `scenarios/<x>.yaml`、`monitoring/alerts/...` 與 `docs/runbook.md` 中。三者之間的漂移是 Contract 測試失敗。
- **Plan 階段故障需要 Shadow Emission。** OpenLineage Spark Listener 無法觀察 Spark Job 啟動前的故障（例如 Schema Drift）。Pipeline 透過在故障路徑上自行發送 OpenLineage Events 來補足這個盲點 — 保留血緣偵測覆蓋率。

---

## 驗證帳本

`./scripts/run-all-failure-scenarios.sh --update-report` 自動更新 `docs/validation-report.md`（若存在），追蹤每個情境的 `last_run_at` 與結果，作為發布驗收的見證文件。

---

## 運維備註

- Pipeline Image 由 `bootstrap-local.sh` 在本地建置為 `local/ai-monitor-pyspark:latest`（使用 `imagePullPolicy: IfNotPresent` 避免 Registry 拉取）。
- Helm 使用 `--create-namespace`；若 Namespace 已存在則重用。
- `nuke-local.sh` 會刪除 Namespace + PV/PVC；視為破壞性操作，執行前請確認。
- 更深入的 Agent 上下文（CLI 捷徑、常見故障、Probe 使用方式），詳見 [`CLAUDE.md`](CLAUDE.md)。
