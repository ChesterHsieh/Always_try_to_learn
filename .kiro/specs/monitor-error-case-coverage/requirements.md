# Requirements Document

## Project Description (Input)
In @ai-monitor-system add more error case to prove monitor system viable

## Introduction
本規格旨在驗證 `ai-monitor-system` 的「**錯誤偵測能力**」：當 pipeline 發生各種失敗時，監控系統（metrics、traces、lineage、alerts）是否能正確**偵測、分類、關聯、通報**。觀測對象是 pipeline；監控元件本身的穩定性 / 延遲 / SLO（meta-monitoring）不在本規格範圍。目前 `failure_classifier.py` 已定義九種失敗類別，但 `scenarios/` 僅提供 `success-baseline.yaml`，缺乏失敗端對端的驗證資產，無法讓平台操作員與資料工程師確信監控系統能抓到真實故障。本規格將補齊每種失敗類別的可執行情境、contract / integration 測試、Prometheus 告警驗證，以及 runbook 章節，使「監控系統能否抓到錯誤」獲得可重複證據。

## Boundary Context
- **In scope**:
  - 為 `KNOWN_CATEGORIES` 中每個分類提供至少一個可執行的情境檔（`scenarios/*.yaml`）。
  - 擴充 `tests/integration/` 與 `tests/contract/` 對失敗路徑的覆蓋。
  - 驗證 run lifecycle metrics、OpenTelemetry trace 屬性、OpenLineage 事件、Prometheus 告警規則在每種失敗模式下的行為。
  - 更新 `docs/runbook.md` 與 `docs/validation-report.md` 以反映新覆蓋。
- **Out of scope**:
  - 新增新的失敗分類（`KNOWN_CATEGORIES` 之外）。
  - 重構 pipeline 商業邏輯或更換監控元件。
  - 多叢集 / 雲端託管環境驗證。
- **Adjacent expectations**:
  - 必須與 `pyspark-monitoring-framework` 規格保持一致的 run 識別與遙測 schema。
  - 沿用既有 Helm chart、Grafana dashboard、Prometheus alert rule 結構。

## Requirements

### Requirement 1: 失敗情境資產覆蓋
**Objective:** As a 平台操作員, I want 每一種已知失敗類別都有可執行的情境檔, so that 我可以在本地 Kubernetes 環境重現故障並驗證監控行為。

#### Acceptance Criteria
1. The Monitor Error Case Coverage system shall 為 `KNOWN_CATEGORIES` 中的每一個分類（input_not_found、invalid_path、permission_denied、spark_task_failed、spark_driver_error、lineage_emission_failed、telemetry_unavailable、timeout、runtime_error）提供至少一個 `scenarios/*.yaml` 檔。
2. When 操作員執行 `deploy/scripts` 中的 scenario runner 並指定某一失敗情境, the Scenario Runner shall 觸發對應的 pipeline 失敗並以非零結束碼回報。
3. The Scenario Runner shall 在情境檔中宣告預期的 `failure_category`、預期 metrics 標籤與預期 alert 名稱。
4. If 情境檔缺少必要欄位（`name`、`expected_failure_category`、`expected_alerts`）, then the Scenario Runner shall 在啟動前以驗證錯誤中止並列出缺少欄位。

### Requirement 2: 失敗分類正確性驗證
**Objective:** As a 資料工程師, I want 失敗分類器在每種情境下回傳預期類別, so that 後續告警、儀表板與 runbook 對應動作不會錯置。

#### Acceptance Criteria
1. When 任一情境完成執行, the Failure Classifier shall 回傳與情境檔 `expected_failure_category` 完全一致的字串。
2. The Failure Classifier shall 將分類結果寫入 run lifecycle metrics 的 `failure_category` 標籤。
3. If 分類器回傳 `runtime_error` 而情境預期為更明確類別, then the Contract Test Suite shall 將該情境標記為失敗並輸出實際與預期分類差異。
4. The Contract Test Suite shall 對 `KNOWN_CATEGORIES` 全部分類執行至少一個對應的分類測試案例。

### Requirement 3: 遙測訊號與 run 關聯
**Objective:** As a 平台操作員, I want 在每個失敗情境中 metrics、traces、lineage 都能以同一 run id 關聯, so that 我能在 Grafana 中跨訊號進行根因分析。

#### Acceptance Criteria
1. When 失敗情境執行完成, the Pipeline Telemetry shall 發出帶有相同 `run_id` 的 lifecycle metric、OpenTelemetry trace 與 OpenLineage 事件。
2. The Integration Test Suite shall 驗證每個失敗情境產生的 trace 至少包含一個帶 `error=true` 屬性的 span。
3. While telemetry collector 不可達（退化情境）, the Pipeline Failure Detection shall 仍能透過 Prometheus lifecycle metric 偵測到 pipeline 失敗並標記 `failure_category=telemetry_unavailable`。
4. If lineage 後端拒絕事件（退化情境）, then the Pipeline Failure Detection shall 仍能透過 Prometheus lifecycle metric 偵測到 pipeline 失敗並標記 `failure_category=lineage_emission_failed`。
5. When schema 不匹配情境執行（例如欄位型別不符導致 Spark `AnalysisException`）, the Pipeline Failure Detection shall 將失敗分類為 `spark_driver_error` 並透過 metrics、traces、lineage 三條偵測路徑回報該 run 為失敗。
6. The Integration Test Suite shall 透過查詢 lineage 後端驗證 schema 不匹配情境的 run state 為 `FAILED`（不驗 facet 內容、不驗欄位數）。

### Requirement 4: 告警與儀表板驗證
**Objective:** As a 工程主管, I want 每種失敗情境都對應到可驗證的 Prometheus 告警與 Grafana 面板狀態, so that 我能確認監控配置在生產相似環境中可實際觸發。

#### Acceptance Criteria
1. When 失敗情境觸發, the Prometheus Alerting shall 在情境宣告的時間窗內進入 `firing` 狀態。
2. The Monitoring Coverage Check Script shall 對每個失敗情境驗證對應 alert rule 存在且 label 與情境一致。
3. Where Grafana dashboard 包含 pipeline-health 或 lineage-view 面板, the Dashboard shall 在失敗情境執行後顯示對應的失敗計數或狀態變更。
4. If 任一預期告警未在時間窗內觸發, then the Integration Test Suite shall 將該情境標記為失敗並輸出告警評估歷程。

### Requirement 5: 文件、Runbook 與可重複性
**Objective:** As a 新進工程師, I want 每個失敗情境在 runbook 與 validation report 中都有清楚的重現與對應動作說明, so that 我可以獨立重現並驗證監控系統。

#### Acceptance Criteria
1. The Documentation Set shall 在 `docs/runbook.md` 中為每一個 `KNOWN_CATEGORIES` 分類提供一節「症狀 / 重現 / 預期告警 / 處置」。
2. The Documentation Set shall 在 `docs/validation-report.md` 列出每個情境最近一次驗證結果（成功 / 失敗 / 未執行）與時間戳。
3. When 任一情境檔新增或修改, the Coverage Check Script shall 拒絕缺少對應 runbook 章節或 validation report 條目的變更。
4. The Bootstrap Scripts shall 提供單一指令以重新執行所有失敗情境並產出彙總報告。
