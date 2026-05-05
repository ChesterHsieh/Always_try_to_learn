# Requirements Document

## Introduction

本規格定義 `ai-monitor-system` 的 **PySpark 監控框架** 第一版（v1）。產品定位為「以監控為設計第一優先」的參考框架，透過在本機 Kubernetes 叢集上執行一條簡單批次型 PySpark 管線（local file → local file），驗證並標準化以下可觀測性堆疊：**OpenLineage、Prometheus、OpenTelemetry、Grafana**。

主要受眾為平台維運人員、資料工程師與工程主管，需求重點在於：
1. 快速偵測管線失敗並提供首要分流脈絡。
2. 以共享 `run_id` 串連 metrics / traces / lineage，加速根因分析。
3. 透過 upstream Helm charts 的標準化堆疊，確保新管線可重複地完成監控接入。

## Boundary Context

- **In scope**:
  - 本機 Kubernetes 叢集（kind / minikube / Docker Desktop K8s）部署。
  - 一條 local-file-to-local-file 的批次參考管線，於單一 namespace 執行。
  - 標準化堆疊四件套（OpenLineage backend、Prometheus、OTel Collector、Grafana），以 upstream Helm charts 為主、專案僅保留 pipeline 與整合 overlay。
  - run-level 狀態、失敗類別、lineage 來源/目標、跨訊號 `run_id` 關聯、儀表板與告警最低語意。
  - 自動化驗證（contract / integration / smoke）與接入引導文件。

- **Out of scope (v1)**:
  - 進階 streaming、結構化串流、多叢集編排。
  - 物件儲存或外部 SaaS observability 後端整合。
  - 進階管線業務邏輯與多管線排程協調。
  - 跨組織 RBAC / 法遵特殊政策（沿用既有組織標準）。

- **Adjacent expectations**:
  - 既有的 incident response 工作流可消費此框架的告警輸出。
  - 既有 Helm 與 Kubernetes 平台、以及命名空間存取權，已由平台團隊提供。
  - Steering 文件（`.kiro/steering/`）所述的觀測架構原則為本規格的設計約束。

## Requirements

### Requirement 1: Run-Level Pipeline Visibility on Kubernetes
**Objective:** As a 平台維運人員, I want 看見每一次 PySpark 管線在 Kubernetes 上的執行狀態與終態, so that 我能立即辨識健康度與服務影響面。

#### Acceptance Criteria
1. When 一個 PySpark 管線在 Kubernetes 上開始執行, the Pipeline Monitoring Framework shall 發出含 `run_id`、`pipeline_name`、`status=running`、`start_time`、`input_path`、`output_path` 的 lifecycle 事件。
2. When 一個管線執行成功完成, the Pipeline Monitoring Framework shall 發出含 `run_id`、`status=succeeded`、`end_time`、`duration_ms` 的 lifecycle 事件。
3. While 多個管線執行同時進行中, the Operator Dashboard shall 顯示每個 run 的當前狀態與識別欄位（`run_id`、`pipeline_name`、`k8s_namespace`、`k8s_job_name`），以便分辨健康與異常 run。
4. The Pipeline Monitoring Framework shall 將 run 狀態轉換 (`running` / `succeeded` / `failed`) 在正常本機叢集條件下於 2 分鐘內呈現於監控視圖中。

---

### Requirement 2: Actionable Failure Signals for First-Response Triage
**Objective:** As a 平台維運人員, I want 在管線失敗時取得明確的失敗類別與脈絡, so that 我可以在不查原始 log 的情況下完成首要分流。

#### Acceptance Criteria
1. If 一個管線執行進入終態失敗, then the Pipeline Monitoring Framework shall 發出含 `run_id`、`status=failed`、`end_time`、`failure_category`、`failure_message` 的 lifecycle 事件。
2. When 失敗 lifecycle 事件被發出, the Alerting Subsystem shall 觸發一筆 `severity=critical`、含 `summary`、`trigger_time`、`run_id`、`dashboard_link` 的 AlertEvent。
3. The Pipeline Monitoring Framework shall 對相同失敗類型套用一致且具決定性的 `failure_category` 對應，以便儀表板與告警呈現一致語意。
4. While 監控後端短暫不可用, the Pipeline Monitoring Framework shall 在後端恢復後仍可使下一次失敗事件被觀察到，並避免靜默吞錯。

---

### Requirement 3: Run-Scoped Lineage and Execution Context
**Objective:** As a 資料工程師, I want 對每一次 run 取得對應的 lineage 與執行脈絡, so that 我能理解處理了哪些資料以及失敗發生在工作流的哪一段。

#### Acceptance Criteria
1. When 一個管線 run 完成, the Lineage Subsystem shall 為該 run 產生含 `run_id`、`job_name`、`job_namespace`、`source_dataset`、`target_dataset`、`event_time` 的 LineageRecord。
2. When 工程師檢視一個 run 的 lineage, the Lineage Subsystem shall 顯示與該 `run_id` 對應的來源與目標 dataset 關係。
3. If 一個管線階段失敗, then the Operator Dashboard shall 將失敗狀態與該 run 的 lineage 路徑進行關聯呈現。
4. If lineage 事件相對 metrics / traces 延遲或亂序到達, then the Pipeline Monitoring Framework shall 仍能透過 `run_id` 完成跨訊號關聯而不需要時間戳猜測。

---

### Requirement 4: Cross-Signal Telemetry Correlation by `run_id`
**Objective:** As a 資料工程師, I want 以共享識別串連 metrics / traces / lineage, so that 我可以在單一查詢路徑完成根因分析。

#### Acceptance Criteria
1. The Pipeline Monitoring Framework shall 對每一個 run 採用單一 `run_id` 作為 metrics、traces、lineage、run-scoped alerts 之強制關聯鍵。
2. When 發出任何管線 trace span, the Pipeline Tracing Module shall 在 span 屬性中包含 `run_id`、`pipeline_name`、`k8s_namespace`，並於終態 span 額外包含 `status`。
3. The Pipeline Monitoring Framework shall 為 run-scoped MonitoringSignal 強制要求 `run_id` 欄位存在。
4. If 一筆 run-scoped signal 缺少 `run_id`, then the Pipeline Monitoring Framework shall 視為違反 contract 並可被 contract test 偵測。

---

### Requirement 5: Standardized Observability Stack via Upstream Helm Charts
**Objective:** As an 工程主管, I want 確認所有 in-scope 部署都使用核可的觀測堆疊與 upstream charts, so that 框架在多團隊間可維護、可升級且行為一致。

#### Acceptance Criteria
1. The Deployment Package shall 透過 Helm upstream charts 部署 Prometheus、Grafana、OpenTelemetry Collector、OpenLineage backend，並以 pinned 版本管理。
2. While 部署本機最小化 profile, the Deployment Package shall 使用 `values.local-minimal.yaml` 套用單副本與資源上下界，避免不可預期的資源占用。
3. The Deployment Package shall 對所有上游 chart 採用釘版 (`chart_version` 不可為浮動 latest)，並在 `chart_version_matrix` 文件中記錄版本綁定。
4. Where 專案需保留自有模板（pipeline-job、namespace、Spark/OpenLineage 配置橋接）, the Deployment Package shall 將其限於管線執行與整合 overlay，避免重新實作上游元件能力。

---

### Requirement 6: Operator Dashboarding for Health, Failure and Recent History
**Objective:** As a 平台維運人員, I want 一致的儀表板呈現健康度、失敗、趨勢與最近 run 歷史, so that 我能在單一視圖完成日常觀測與分流。

#### Acceptance Criteria
1. The Operator Dashboard shall 對所有 run 顯示一致的狀態語意（`running`、`succeeded`、`failed`、`recovering`）。
2. The Operator Dashboard shall 提供失敗 run 的 triage 視圖，至少包含 `run_id`、`failure_category`、`failure_message` 與對應 lineage 連結。
3. While 預期高峰執行量, the Operator Dashboard shall 維持可用性與回應性，符合 quality validation QV-004。
4. Where 啟用 lineage-focused 儀表板, the Operator Dashboard shall 提供以 `run_id` 為入口的根因分析瀏覽路徑。

---

### Requirement 7: Alerting Workflow with Sufficient Operator Context
**Objective:** As a 平台維運人員, I want 接收具備足夠脈絡的關鍵告警, so that on-call 可以立即採取行動。

#### Acceptance Criteria
1. The Alerting Subsystem shall 對每一筆 critical AlertEvent 提供可操作的 `summary`。
2. When 偵測到 telemetry freshness 超過設定閾值, the Alerting Subsystem shall 產生 `severity=warning` 的 AlertEvent。
3. When run-failure 條件觸發, the Alerting Subsystem shall 在告警 payload 中附上 `run_id` 與 `dashboard_link`。
4. If 監控後端暫時不可用, then the Alerting Subsystem shall 不靜默吞錯，且須在後端恢復後可被驗證的鏈路上重新觀察。

---

### Requirement 8: Repeatable Onboarding for New Pipelines
**Objective:** As an 工程主管, I want 提供可重複執行的接入步驟, so that 新的簡單 PySpark 管線可在 1 個工作日內採用此框架。

#### Acceptance Criteria
1. The Onboarding Documentation shall 描述以 upstream chart 為基礎的 bootstrap 流程，可由新成員依步驟完成本機部署。
2. When 新管線啟用監控, the Onboarding Process shall 不需任何客製化 one-off 工具即可符合標準觀測堆疊。
3. The Onboarding Documentation shall 提供 quickstart 與 runbook，並涵蓋失敗情境的回滾與排錯指引。
4. Where 環境為 local-minimal profile, the Onboarding Process shall 在常見開發者級別硬體上於合理時間內 (bootstrap ≤ 10 分鐘、樣本成功 run ≤ 5 分鐘) 完成驗證。

---

### Requirement 9: Minimum Monitoring Acceptance Checks (Coverage Profile)
**Objective:** As an 工程主管, I want 自動化的最小監控接受檢查, so that 任何釋出版本都能驗證觀測堆疊符合 production-ready 標準。

#### Acceptance Criteria
1. The Coverage Profile shall 列出必要元件（OpenLineage、Prometheus、OTel Collector、Grafana），且必要元件未到位時不可通過 readiness 檢查。
2. The Coverage Profile shall 將每一項 validation_check 對應到可執行的自動化測試或腳本。
3. When 執行 coverage check, the Coverage Subsystem shall 驗證所有上游元件可達且報告其當前綁定的 `chart_version`。
4. When 一個 release candidate 通過自動化監控驗證, the Coverage Subsystem shall 將最後驗證時間 (`last_verified_at`) 記錄於 CoverageProfile。

---

### Requirement 10: Required Operational Metrics
**Objective:** As a 平台維運人員, I want 標準化的運維指標, so that 健康度、資源行為、結果分布可長期被觀測。

#### Acceptance Criteria
1. The Pipeline Monitoring Framework shall 暴露指標族：`pipeline_run_total{status}`、`pipeline_run_duration_seconds`、`pipeline_records_processed_total`、`pipeline_failures_total{failure_category}`、`pipeline_telemetry_freshness_seconds`。
2. When 任一指標族缺失, the Coverage Subsystem shall 回報為 contract 違規。
3. The Pipeline Monitoring Framework shall 確保上述指標可在 Grafana 儀表板與 Prometheus 查詢中被一致使用。

---

### Requirement 11: Resilience Under Transient Adverse Conditions
**Objective:** As a 平台維運人員, I want 系統在常見邊界情境下仍可信賴, so that 監控本身不成為單點故障。

#### Acceptance Criteria
1. If 管線執行成功但因瞬時網路問題使遙測不完整, then the Pipeline Monitoring Framework shall 於 telemetry freshness 警示中可見此狀況，避免靜默成功假象。
2. If Kubernetes pod 在執行中重啟, then the Pipeline Monitoring Framework shall 不產生重複或誤導性的告警狀態。
3. If lineage 事件相對 metrics 與 traces 晚到或亂序, then the Pipeline Monitoring Framework shall 仍能依 `run_id` 正確關聯。
4. While 監控後端在管線高峰期短暫不可用, the Pipeline Monitoring Framework shall 在恢復後不丟失能驗證 `run_id` 對應關係的事件可見性。

---

### Requirement 12: Quality, Testability and Documentation Standards
**Objective:** As an 工程主管, I want 框架本身符合品質、可測試性與文件標準, so that 後續維護與擴充風險可控。

#### Acceptance Criteria
1. The Pipeline Monitoring Framework shall 為核心監控行為（失敗偵測、遙測關聯、lineage 完整性、stack 覆蓋）提供 contract / integration / smoke 自動化測試。
2. The Documentation shall 對所有對外輸出 artifact 標明擁有者、預期輸入/輸出與維護指引。
3. The Operator Dashboard 與 Alerting Subsystem shall 在儀表板與告警之間維持一致的狀態與嚴重度語意。
4. When 釋出版本前, the Validation Process shall 執行所有必要自動化監控驗證並通過。

---

## Success Criteria Mapping

> 為利後續設計與測試對齊，下列為 spec.md 中既有 SC 與本需求的對應參考。

- SC-001 ↔ Requirement 1.4 / Requirement 2.2 / Requirement 6.1
- SC-002 ↔ Requirement 5 / Requirement 9
- SC-003 ↔ Requirement 8
- SC-004 ↔ Requirement 1 / Requirement 2 / Requirement 4 / Requirement 6
