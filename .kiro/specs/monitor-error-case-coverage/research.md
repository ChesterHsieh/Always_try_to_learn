# Research & Design Decisions — monitor-error-case-coverage

## Summary
- **Feature**: `monitor-error-case-coverage`
- **Discovery Scope**: Extension（既有 `ai-monitor-system` 增補錯誤情境覆蓋與驗證資產）
- **Key Findings**:
  - Scenario runner（`scripts/run-scenario.sh` + `scripts/probe.py`）已具備可重複呼叫 PromQL probe 的骨架，目前僅有 `success-baseline.yaml`，可直接擴充失敗情境檔。
  - `pipeline/failure_classifier.py` 已定義 9 種 `KNOWN_CATEGORIES` 與分類規則，但 contract / integration 測試與 alert 規則並未對等覆蓋（現有 alert 多聚焦 stack-health，例如 `OTelCollectorDown`、`PipelineTelemetryStaleness`，缺少分類維度的 `PipelineRunFailed` 變體）。
  - run lifecycle、tracing、lineage 已有 helper（`run_context.py`、`tracing.py`、`lineage_emitter.py`），run_id 為現成相關性鍵；本規格只需擴充故障注入點與相應驗證，不需更動遙測 schema。

## Research Log

### 既有 scenario 與 probe 框架
- **Context**: 確認新增情境是否需要重寫 runner。
- **Sources Consulted**: `scripts/run-scenario.sh`、`scripts/probe.py`、`scenarios/success-baseline.yaml`、`deploy/scripts/run-pipeline.sh`。
- **Findings**:
  - runner 透過 env var（如 `INJECT_FAILURE`）驅動 `run-pipeline.sh`，YAML 之 `pipeline.inject_failure` 為已預留欄位。
  - probe 已支援 PromQL `gte` / `eq` / `within` 語意，可直接驗證 `ALERTS{alertname=...,alertstate="firing"}`。
- **Implications**: 新情境主要工作為 (a) 擴充 `inject_failure` 取值、(b) 新增 9 個 YAML、(c) 補對應 alert rule 與 runbook 章節，不需新建 runner。

### 失敗分類與遙測標籤
- **Context**: 失敗分類如何進入 metric label 以利告警分流。
- **Sources Consulted**: `pipeline/failure_classifier.py`、`pipeline/metrics.py`、`pipeline/run_context.py`。
- **Findings**:
  - `classify_failure` 回傳字串會被寫入 `pipeline_run_duration_seconds_count{status,failure_category}` 或等價 lifecycle metric。
  - lineage / tracing 不直接攜帶 `failure_category`，但共用 `run_id`，可透過跨訊號 join。
- **Implications**: alert rule 可以 `failure_category` label 為維度產生分類別告警；不需改 schema。

### 既有 alert 規則覆蓋面
- **Context**: 評估目前 alert 是否足以驗證失敗情境。
- **Sources Consulted**: `monitoring/alerts/stack-health-rules.yaml`。
- **Findings**:
  - 現有規則涵蓋 collector down、telemetry freshness、metrics endpoint 不可達；缺乏 run-level 失敗（`PipelineRunFailed{failure_category=...}`）規則。
- **Implications**: 需在 `monitoring/alerts/` 新增 `pipeline-failure-rules.yaml`（或併入現檔）以涵蓋 9 種分類；以 label matcher 控制 fan-out。

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| 集中型 alert（單規則 + label） | 用一條 `PipelineRunFailed` alert 並以 `failure_category` label 區分 | 規則少、易維護 | 嚴重度無法分級、annotation 難個別化 | 需要 Grafana 端做 label 篩選 |
| 分類別 alert（每類一條規則） | 每個 `failure_category` 一條 alert | 嚴重度／runbook 可個別 | 規則檔變大 | 與 runbook 章節 1:1 對齊 |
| 混合（嚴重類分開、其餘共用） | spark/lineage/telemetry 等關鍵分類獨立，其餘共用 | 平衡維護成本與可操作性 | 規則分群準則需明示 | **採用** |

## Design Decisions

### Decision: 採用「混合 alert 策略」
- **Context**: 9 種失敗分類嚴重度與處置動作不同。
- **Alternatives Considered**:
  1. 集中型 alert + label
  2. 每類一條 alert
- **Selected Approach**: 對 `spark_driver_error`、`lineage_emission_failed`、`telemetry_unavailable`、`timeout` 各自獨立 alert；其餘併入共用 `PipelineRunFailed` 規則並以 `failure_category` label 區分。
- **Rationale**: 嚴重類有獨立 runbook 處置，需要個別 annotation 與 severity；輕度類用共用規則減少維護成本。
- **Trade-offs**: 規則規劃需明確分組準則；新增分類時需決定歸屬。
- **Follow-up**: 在 `runbook.md` 列出每分類所對應 alert 名稱。

### Decision: 失敗注入透過 `INJECT_FAILURE` env var 集中分派
- **Context**: 9 種情境需可重現且互不污染。
- **Alternatives Considered**:
  1. 每類一支 entrypoint script
  2. 單一 env var 分派至 pipeline 內部 hook
- **Selected Approach**: 在 `pipeline/job.py`（或新增 `pipeline/failure_injection.py`）依 `INJECT_FAILURE` 取值執行對應錯誤路徑（例如 `permission_denied` → `chmod` 後讀檔）。
- **Rationale**: 與既有 `run-pipeline.sh` env 慣例一致；單一注入模組便於測試與擴充。
- **Trade-offs**: 注入邏輯集中於非生產路徑，需確保不在生產 image 啟用（透過 env gate）。
- **Follow-up**: contract test 驗證 `INJECT_FAILURE=none` 時注入模組為 no-op。

### Decision: 規格目的收斂為「驗證錯誤偵測能力」
- **Context**: 使用者澄清本規格目的是驗證監控系統能否抓到 pipeline 的錯誤，**不是**驗證觀測元件本身的穩定性 / SLO（meta-monitoring）。
- **Impact**:
  - 重寫 Introduction、Overview、Boundary 將觀測對象明確收斂為 pipeline。
  - `lineage_emission_failed`、`telemetry_unavailable` 兩情境定位重寫：驗證的是「監控後端不可達時，pipeline 失敗仍能被偵測」這一**韌性**面，而非後端本身的健康度。
  - 排除 meta-monitoring 議題（component latency、collector export latency、Marquez 寫入延遲等）。
- **Rationale**: 收斂 boundary 可避免規格膨脹；component latency 屬另一規格的議題。

### Decision: 加入 `schema-mismatch` 情境並收斂 lineage 驗證為「run 終態可達性」
- **Context**: OpenLineage 對 data schema 變動會在成功 read/write 時發出 `schemaDatasetFacet`；想驗證 schema 觸發的失敗能否被監控抓到。先前提案是查 facet 內容與欄位數，但這越界進入「驗證 lineage 後端行為正確性」。
- **Alternatives Considered**:
  1. 不動：完全不加 schema-mismatch；lineage 偵測路徑就無情境覆蓋。
  2. 加情境 + 驗 facet 內容（欄位數比對）：超出本規格 boundary（驗 lineage 後端而非 pipeline 偵測）。
  3. **採用**：加情境，但 lineage 驗證收斂為「lineage 後端是否收到該 run 的 `FAILED` 終態事件」。
- **Selected Approach**: 新增 `scenarios/schema-mismatch.yaml`，`probe.py` 加 `lineage-run-state` cmd（唯一斷言 `state_eq=FAILED`）。
- **Rationale**: 對齊「驗證偵測能力」的核心目的 — lineage 是三條偵測路徑之一（metrics、traces、lineage），驗其終態可達性即足夠；驗 facet 內容是越界。
- **Trade-offs**: 引入 lineage backend HTTP 為 probe 依賴（P1，僅 schema-mismatch 情境使用）；lineage 寫入非同步須以輪詢 `within` 處理。
- **Follow-up**: 確認本地 profile lineage 後端端點可用；若關閉，coverage check 應跳過 schema-mismatch 並警告。

### Decision: scenario YAML schema 擴充必填欄位
- **Context**: 新情境需自描述預期分類與告警。
- **Selected Approach**: 在每個 `scenarios/*.yaml` 增加 `expected_failure_category`（string）、`expected_alerts`（list of alertname）、`expected_run_status`（`failed` / `succeeded`）。
- **Rationale**: 讓 runner 與 coverage check 可以靜態驗證情境完整性。
- **Trade-offs**: 既有 `success-baseline.yaml` 需補欄位（`expected_run_status: succeeded`、`expected_failure_category: null`）。
- **Follow-up**: `check-monitoring-coverage.sh` 加入 schema 驗證步驟。

## Risks & Mitigations
- 風險：注入失敗的 pipeline 程式碼污染正式 image — 緩解：`INJECT_FAILURE` 預設 `none`，並於 helm values 不曝露此 env。
- 風險：alert 在本地時間窗（`for: 2m`）內不及觸發 — 緩解：情境檔 `within` 預設 180s，必要時對失敗類規則使用 `for: 30s`。
- 風險：lineage/telemetry backend 不可達情境本身會影響 probe — 緩解：probe 採 PromQL（Prometheus 仍可用），不依賴受測的後端。

## References
- 內部：`ai-monitor-system/pipeline/failure_classifier.py`、`scripts/run-scenario.sh`、`scenarios/success-baseline.yaml`、`monitoring/alerts/stack-health-rules.yaml`。
- Prometheus alerting rules: https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/
- OpenLineage Spark integration: https://openlineage.io/docs/integrations/spark/
