# Gap Analysis — monitor-error-case-coverage

> **語言**: 繁體中文（依 spec.json 設定）
> **分析日期**: 2026-05-04
> **規格狀態**: tasks-generated / ready_for_implementation

---

## 執行摘要

| 項目 | 結論 |
|------|------|
| 分析方式 | 對現有 brownfield codebase 進行深度掃描 |
| 整體完成度 | **高度完成**（約 95%）— 多數任務已實作完畢 |
| 現存唯一缺口 | Alert rule 缺少 `increase()` 包裝（1 個測試失敗） |
| 剩餘任務 | Task 6.3（一鍵彙總執行本地驗收）+ alert rule 修正 |
| 建議方式 | **Option A（Extend Existing）**：直接修正既有告警 YAML，完成驗收任務 |

---

## 1. 現有代碼庫現況調查

### 已完成的資產

#### scenarios/ 目錄（完整）
全部 9 個 `KNOWN_CATEGORIES` 情境檔均已存在：

| 情境檔 | 對應分類 | 狀態 |
|--------|----------|------|
| `input-not-found.yaml` | `input_not_found` | ✅ 存在，含完整 probes |
| `invalid-path.yaml` | `invalid_path` | ✅ 存在，含完整 probes |
| `permission-denied.yaml` | `permission_denied` | ✅ 存在，含完整 probes |
| `spark-task-failed.yaml` | `spark_task_failed` | ✅ 存在，含完整 probes |
| `spark-driver-error.yaml` | `spark_driver_error` | ✅ 存在，含完整 probes |
| `lineage-emission-failed.yaml` | `lineage_emission_failed` | ✅ 存在，含完整 probes |
| `telemetry-unavailable.yaml` | `telemetry_unavailable` | ✅ 存在，含完整 probes |
| `timeout.yaml` | `timeout` | ✅ 存在，含完整 probes |
| `runtime-error.yaml` | `runtime_error` | ✅ 存在，含完整 probes |
| `schema-mismatch.yaml` | `spark_driver_error`（schema 觸發） | ✅ 存在，含 lineage probe |
| `success-baseline.yaml` | `null`（成功基準） | ✅ 已補齊新欄位 |

#### pipeline/ 模組
| 模組 | 現況 |
|------|------|
| `failure_classifier.py` | ✅ 完整，9 種分類，邏輯穩定 |
| `failure_injection.py` | ✅ 完整，`SUPPORTED_INJECTIONS`、`STAGE_FOR_CATEGORY`、`maybe_inject()` |
| `scenario_schema.py` | ✅ 完整，含必填欄位驗證、`ScenarioValidationError` |
| `metrics.py` | ✅ `pipeline_failures_total` 含 `failure_category` label |
| `job.py` | ✅ 已接入三個注入點（`pre_spark`、`during_spark`、`post_spark`） |

#### 測試層
| 測試文件 | 覆蓋 | 狀態 |
|----------|------|------|
| `tests/contract/test_failure_classifier_contract.py` | 全 9 類分類 + Py4J 退化偵測 | ✅ 通過 |
| `tests/contract/test_failure_injection_contract.py` | 全注入介面 + round-trip | ✅ 通過 |
| `tests/contract/test_scenario_schema_contract.py` | schema 驗證、必填欄位、拒絕非法輸入 | ✅ 通過 |
| `tests/contract/test_coverage_alignment_contract.py` | 三方對齊（scenarios / runbook / alerts） | ✅ 通過 |
| `tests/integration/test_failure_injection_integration.py` | 全 9 類 lifecycle payload + metric | ✅ 通過 |
| `tests/integration/test_failure_scenario_integration.py` | 全 9 類 + schema-mismatch 三路徑 | ✅ 通過 |
| `tests/integration/test_failure_alerts.py` | alert rule 結構 / annotations | **⚠️ 1 test FAIL** |
| `tests/integration/test_trace_attributes.py` | span attributes + error span | ✅ 通過 |
| `tests/integration/test_lineage_correlation.py` | run_id 跨 metrics/lineage 關聯 | ✅ 通過 |

**整體測試狀態**: `1 failed, 153 passed`

#### 文件
| 文件 | 現況 |
|------|------|
| `docs/runbook.md` | ✅ 含全 9 個 `## failure-<category>` 章節（「症狀 / 重現 / 預期告警 / 處置」結構） |
| `docs/validation-report.md` | ✅ 含 ledger 表格（10 情境全部列出，result=fail 因尚未在 live cluster 執行） |

#### 腳本
| 腳本 | 現況 |
|------|------|
| `scripts/run-scenario.sh` | ✅ 完整，含 expected vs actual 比對 + probes |
| `scripts/run-all-failure-scenarios.sh` | ✅ 完整，含 `--update-report` 旗標 |
| `scripts/probe.py` | ✅ 含 `lineage-run-state` 子命令 |
| `deploy/scripts/check-monitoring-coverage.sh` | ✅ 委派 `pipeline.coverage` CLI |

---

## 2. 需求可行性分析

### 需求對應資產對應表

| 需求 | 關鍵技術需求 | 現有資產 | 缺口 |
|------|-------------|---------|------|
| R1: 失敗情境資產覆蓋 | 9 個 scenario YAML、schema 驗證、scenario runner | scenarios/ 目錄齊全；`scenario_schema.py` 已實作 | 無 |
| R2: 失敗分類正確性驗證 | classifier round-trip、metric label、contract test | `failure_classifier.py` + `failure_injection.py`；153/154 test 通過 | 無 |
| R3: 遙測訊號與 run 關聯 | 同 run_id 的 metric/trace/lineage、error span、degraded path | `job.py` 已接入；`test_trace_attributes.py`、`test_lineage_correlation.py` 通過 | 無 |
| R4: 告警與儀表板驗證 | `increase()` 包裝 PromQL、alert 在 30s 內 firing | alert rule 存在但**缺少 `increase()` 包裝**；1 個 integration test FAIL | ⚠️ **Alert rule 缺口** |
| R5: 文件與可重複性 | 每類 runbook 章節、validation ledger、一鍵執行腳本 | runbook 9 章節已存在；ledger 已初始化；`run-all-failure-scenarios.sh` 已完成 | ⚠️ Task 6.3（live cluster 驗收）尚未執行 |

### 已識別缺口

#### 缺口 1：`pipeline-failure-rules.yaml` 告警表達式缺少 `increase()` 包裝（**高優先**）

- **位置**: `monitoring/alerts/pipeline-failure-rules.yaml`，全部 5 條 alert
- **問題**: 目前 expr 使用 `pipeline_failures_total{...} > 0`（gauge 比較），而非 `increase(pipeline_failures_total{...}[5m]) > 0`
- **失敗測試**: `test_failure_alerts.py::test_failure_alert_uses_increase_expr`
- **影響**: 設計文件 task 2.3 明確要求使用 `increase()`；不符合 Requirement 4 的 alert 觸發語義（counter 重啟後不會誤觸發）
- **修正工作量**: XS（< 30 分鐘）— 純 YAML 修改

#### 缺口 2：Task 6.3 本地 live cluster 驗收尚未執行（**低優先，非代碼問題**）

- **問題**: 需要 bootstrap 本地 K8s cluster，執行 `run-all-failure-scenarios.sh`，觀察 Prometheus alert 實際 firing
- **影響**: validation-report.md ledger 中所有情境 `result=fail`（因為從未在 live cluster 執行過）
- **修正工作量**: M（依 cluster 啟動時間而定，1-3 小時）— 操作驗收而非代碼問題

---

## 3. 實作方式選項

### Option A：修正現有元件（**建議選項**）

**適用原因**: 代碼庫已高度完整，唯一代碼缺口是告警 YAML 中的表達式語法

**需修改的文件**:
| 文件 | 變更 | 影響 |
|------|------|------|
| `monitoring/alerts/pipeline-failure-rules.yaml` | 將 5 條 alert expr 從 `counter > 0` 改為 `increase(counter[5m]) > 0` | 修復 1 個 FAIL 測試；符合設計規格 |

**相容性評估**:
- ✅ 不破壞其他測試（`test_failure_alert_for_duration_is_30s` 等仍通過）
- ✅ 不改變 alert 名稱或標籤結構
- ✅ `test_coverage_alignment_contract.py` 三方對齊不受影響
- ✅ Prometheus rule 語義更正確（counter reset 不觸發誤報）

**Trade-offs**:
- ✅ 最小修改，高確定性
- ✅ 直接對齊設計文件 task 2.3 的規格
- ❌ 仍需手動 live cluster 驗收（非代碼，見缺口 2）

### Option B：新建獨立 alert rule 文件

**適用原因**: 若希望保留舊版 rule 作比對或需要分層部署

**Trade-offs**:
- ✅ 舊版本保留，方便回滾
- ❌ 增加文件管理複雜度
- ❌ `test_coverage_alignment_contract.py` 可能需同步調整

**結論**: 不建議，缺口只是 YAML 一行修改，不值得增加間接性。

### Option C：混合方式

不適用於本規格，缺口過小，無需混合策略。

---

## 4. 複雜度與風險評估

| 維度 | 評估 | 理由 |
|------|------|------|
| **整體工作量** | S（1-3 天） | 代碼層幾乎完整；唯一代碼修改是 alert rule YAML；主要工作是 live cluster 操作驗收 |
| **技術風險** | **低** | 所有模組已實作並有 153 個測試通過；`increase()` 修正是純語法問題 |
| **整合風險** | **低** | 既有 scenario harness 完整；probe 工具已測試；runbook 已對齊 |
| **驗收風險** | **中** | Task 6.3 需要 live cluster，依賴本地環境是否正常啟動（`bootstrap-local.sh`） |

---

## 5. 設計階段建議

### 優先修正（代碼層）

1. **修正 alert rule expr**：將 `pipeline-failure-rules.yaml` 中所有 5 條 alert 的 expr 改為 `increase(...[5m]) > 0`，使 `test_failure_alert_uses_increase_expr` 通過。

### 後續驗收（操作層）

2. **Task 6.3 本地驗收**：
   - 啟動 cluster：`./deploy/scripts/bootstrap-local.sh`
   - 執行一鍵驗收：`./scripts/run-all-failure-scenarios.sh --update-report`
   - 驗證 validation-report.md ledger 更新為最新結果

### 研究需求

無。所有技術面均已充分實作，不需要進一步研究。

---

## 6. 分析結論

本規格的實作**高度完整**。代碼層唯一的實質缺口是 `monitoring/alerts/pipeline-failure-rules.yaml` 中告警表達式缺少 `increase()` 包裝，這是一個 XS 規模的 YAML 修改。

其餘 Task 6.3 為操作驗收任務（需要 live cluster），不是代碼缺口。建議：
1. 先修正 alert rule（< 30 分鐘）
2. 確認全 154 個 contract + integration 測試通過
3. 視環境可用性執行 live cluster 驗收
