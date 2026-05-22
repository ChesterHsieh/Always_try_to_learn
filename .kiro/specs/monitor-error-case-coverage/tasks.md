# Implementation Plan

> 範圍：以「驗證 monitor system 能否抓到 pipeline 錯誤」為目的，補齊 9 種失敗分類情境 + `schema-mismatch` 情境，並擴充對應的注入點、metric label 規約、告警規則、probe、測試與 runbook 條目。
> 順序：foundation → core（分類資產，多項可並行）→ integration → validation。

- [ ] 1. Foundation：scenario schema、metric label 規約、failure injection 介面骨架
- [x] 1.1 擴充 scenario YAML schema 與解析驗證
  - 在 runner 與 coverage check 共用之 schema 模組中新增必填欄位（`expected_run_status`、`expected_failure_category`、`expected_alerts`、`probes`）。
  - `expected_failure_category` 限定為 `KNOWN_CATEGORIES` 之一或 `null`（success 情境）。
  - 缺欄位 / 類別不符 / 未知 alert 名稱時，schema 驗證以非零退出並列出具體錯誤項。
  - **Observable**: 對既有 `success-baseline.yaml` 補欄位後通過驗證；故意刪欄位的測試 fixture 失敗並列出缺項。
  - _Requirements: 1.3, 1.4_
  - _Boundary: ScenarioSchema_

- [x] 1.2 凍結 metric label contract（順應既有 schema）
  - **不**修改 `pipeline_run_duration_seconds` 與 `pipeline_run_total` 的 label set（保留既有 `{status, pipeline_name}`）。
  - 在 `record_run_failed` 路徑強制要求 `failure_category` 必為 `KNOWN_CATEGORIES` 之一，並以 `pipeline_failures_total.labels(failure_category=<x>, pipeline_name=<y>).inc(exemplar={"run_id": <id>})` 寫入。
  - 提供共用工具（如 module-level 常量 `from pipeline.failure_classifier import KNOWN_CATEGORIES`）給後續任務使用。
  - **Observable**: contract test 對既有 metric label set 不變、失敗 run 後 `pipeline_failures_total{failure_category=<x>}` 至少加 1、未知分類拋例外。
  - _Requirements: 2.2_
  - _Boundary: MetricLabelContract_
  - _Depends: 1.1_

- [x] 1.3 建立 failure injection 介面骨架（不接真實失敗路徑）
  - 定義 `SUPPORTED_INJECTIONS`、`InjectionStage`（`pre_spark` / `during_spark` / `post_spark`）。
  - 對每個分類在介面層登記固定階段（依 design.md 階段對應表）；階段不符即 no-op，分類未知即 fail fast。
  - **Observable**: 介面以單元測試覆蓋全部 `SUPPORTED_INJECTIONS`，`none` 在三階段皆 no-op；分類—階段對應表由單一表常量驅動。
  - _Requirements: 1.2_
  - _Boundary: FailureInjection_

- [x] 2. Core：分類資產（情境檔、注入路徑、告警規則、contract 測試）
- [x] 2.1 (P) 撰寫 9 種失敗分類情境檔
  - 為 `input_not_found`、`invalid_path`、`permission_denied`、`spark_task_failed`、`spark_driver_error`、`lineage_emission_failed`、`telemetry_unavailable`、`timeout`、`runtime_error` 各建立一份 YAML，宣告 `expected_run_status=failed`、預期分類、預期 alert、PromQL probes。
  - 補齊 `success-baseline.yaml` 的新欄位。
  - **Observable**: `check-monitoring-coverage.sh` 對 `KNOWN_CATEGORIES` 的清單匹配無缺漏。
  - _Requirements: 1.1, 1.3_
  - _Boundary: ScenarioAssets_
  - _Depends: 1.1_

- [x] 2.2 (P) 實作 failure injection 各分類觸發路徑
  - 依分類—階段對應表，於 pipeline 三個固定點接入注入點：(1) 輸入解析前；(2) DataFrame action / UDF 內；(3) telemetry / lineage flush 前。
  - 各分類拋出與 `failure_classifier.classify_failure` 規則一致的例外類型（`FileNotFoundError`、`PermissionError`、`Py4JJavaError` 模擬等）。
  - 注入路徑以 env gate 控制；helm values 預設不含此鍵。
  - **Observable**: 於本地以 `INJECT_FAILURE=<x>` 執行 pipeline，分類器對所有 9 類回傳對應字串並寫入 metric label。
  - _Requirements: 1.2, 2.1, 2.2_
  - _Boundary: FailureInjection, FailureClassifier_
  - _Depends: 1.2, 1.3_

- [x] 2.3 (P) 撰寫 `pipeline-failure-rules.yaml` 告警群（混合策略）
  - 4 條獨立 alert：`PipelineSparkDriverError`、`PipelineLineageEmissionFailed`、`PipelineTelemetryUnavailable`、`PipelineRunTimeout`，PromQL `increase(pipeline_failures_total{failure_category="<x>"}[5m]) > 0`。
  - 1 條共用 alert `PipelineRunFailed`，PromQL `increase(pipeline_failures_total{failure_category=~"input_not_found|invalid_path|permission_denied|spark_task_failed|runtime_error"}[5m]) > 0`。
  - `for:` 預設 30s。
  - **Observable**: `promtool check rules` 對該檔案通過；alert 名稱與情境檔 `expected_alerts` 對齊。
  - _Requirements: 4.1, 4.2_
  - _Boundary: PipelineFailureRules_
  - _Depends: 1.2_

- [x] 2.4 (P) 擴充 contract test 覆蓋分類器全 9 類
  - 對每個 `KNOWN_CATEGORIES` 構造代表性例外，斷言分類器回傳精確字串。
  - 新增退化偵測：可分類例外被歸為 `runtime_error` 即 fail。
  - **Observable**: pytest 報告每類至少一個 case 通過；故意傳入 spark `Py4JJavaError` 文字而期望 `runtime_error` 的反例會 fail。
  - _Requirements: 2.1, 2.3, 2.4_
  - _Boundary: FailureContractTests_
  - _Depends: 1.3_

- [x] 3. Schema-mismatch 情境與 lineage 偵測路徑驗證
- [x] 3.1 撰寫 `schema-mismatch.yaml` 情境檔
  - `inject_failure: schema_mismatch`、`expected_failure_category: spark_driver_error`。
  - probes 至少含：分類 metric、alert firing、`lineage-run-state` probe。
  - **Observable**: schema 檢查在情境執行時於 `during_spark` 階段拋 `AnalysisException`（或等義型別錯誤）。
  - _Requirements: 3.5, 3.6_
  - _Boundary: ScenarioAssets_
  - _Depends: 2.1, 2.2_

- [x] 3.2 為 probe 工具新增 `lineage-run-state` 子命令
  - 唯讀查詢 lineage 後端 run 終態；唯一斷言 `state_eq=FAILED`，二態結果（PASS / FAIL，無 SKIP）。
  - 端點透過 env 注入；`within` 預設 60s 並輪詢。
  - **Observable**: 對 mock lineage 後端的成功（`FAILED`）與失敗（404 / `RUNNING`）兩種回應產出對應 verdict。
  - _Requirements: 3.6_
  - _Boundary: LineageRunStateProbe_

- [x] 4. Integration：runner、彙總執行、coverage check 串接
- [x] 4.1 在 runner 加入 expected vs actual 比對
  - runner 結束時驗證：lifecycle metric `failure_category` 等於 `expected_failure_category`、`expected_alerts` 全部於 `within` 內進入 firing。
  - 失敗時輸出告警評估歷程與 metric 快照。
  - **Observable**: 對任一失敗情境執行後，stdout 含結構化 verdict 行；缺一即整體非零退出。
  - _Requirements: 1.2, 2.1, 4.1, 4.4_
  - _Boundary: ScenarioRunnerExt_
  - _Depends: 2.1, 2.2, 2.3_

- [x] 4.2 擴充 `check-monitoring-coverage.sh` 為三方對齊閘門
  - 對齊：scenario 檔 ↔ `KNOWN_CATEGORIES` ↔ runbook 章節 anchor ↔ `pipeline-failure-rules.yaml` alertname。
  - 偵測 `LINEAGE_BACKEND_URL` 健康端點；不可達即整體 FAIL（不引入 SKIP）。
  - 驗證 production overlay 不含 `INJECT_FAILURE` 鍵。
  - **Observable**: 故意刪除 runbook 一節或 alert 規則時 script 非零退出並列出缺項。
  - _Requirements: 1.4, 4.2, 5.3_
  - _Boundary: CoverageCheckExt_
  - _Depends: 2.1, 2.3, 5.1_

- [x] 4.3 提供 `run-all-failure-scenarios.sh` 一鍵彙總入口
  - 依序執行所有 `expected_run_status=failed` 情境；二態結果（pass / fail），無 skipped。
  - `--update-report` 旗標寫入 `docs/validation-report.md` 的 ledger 段（scenario / last_run_at / result / run_id）。
  - 任一失敗最終以非零退出。
  - **Observable**: 本地執行後 stdout 顯示彙總表；指定 `--update-report` 後 ledger 段可被 git diff 觀察到更新。
  - _Requirements: 5.4, 5.2_
  - _Boundary: RunAllFailureScenarios_
  - _Depends: 4.1_

- [x] 5. Documentation：runbook 與 validation report 條目
- [x] 5.1 為每個 `KNOWN_CATEGORIES` 分類撰寫 runbook 章節
  - 每分類一節，固定四欄：症狀 / 重現 / 預期告警 / 處置；anchor 命名 `failure-<category>`。
  - 新增 `schema-mismatch` 對應段落（沿用 `spark_driver_error` 章節 + lineage 偵測說明）。
  - **Observable**: `check-monitoring-coverage.sh`（4.2）以正則檢出每個分類 anchor 均存在。
  - _Requirements: 5.1_
  - _Boundary: RunbookFailureSections_

- [x] 5.2 初始化 validation report ledger 結構
  - 在 `docs/validation-report.md` 增加 ledger 表（scenario / last_run_at / result / run_id），首次填入 placeholder 行對應每個失敗情境。
  - **Observable**: `run-all-failure-scenarios.sh --update-report` 會更新該表中對應 row（4.3 完成後）。
  - _Requirements: 5.2_
  - _Boundary: ValidationReportLedger_

- [x] 6. Validation：端對端整合測試
- [x] 6.1 (P) 失敗告警與 metric label 整合測試
  - 為每個失敗情境執行 runner 子流程；斷言 metric `failure_category` 標籤、alert firing、trace span 至少一個 `error=true`。
  - 對 `telemetry_unavailable` 情境模擬 collector 不可達，仍可從 Prometheus 觀察 lifecycle metric。
  - 對 `lineage_emission_failed` 情境模擬 lineage 後端拒絕，pipeline 仍以該分類揭露失敗。
  - **Observable**: pytest 9 個失敗情境全綠；故意關閉某 alert rule 之反例 fail。
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.1, 4.4_
  - _Boundary: FailureAlertIntegrationTests_
  - _Depends: 4.1_

- [x] 6.2 (P) schema-mismatch 端對端測試
  - 透過 runner 執行 `schema-mismatch.yaml`；斷言三條偵測路徑：metric label `spark_driver_error`、trace `error=true` span、lineage 後端 run state `FAILED`。
  - **Observable**: pytest 通過，且關閉 lineage 後端時整體 FAIL（不會被 SKIP 隱藏）。
  - _Requirements: 3.5, 3.6_
  - _Boundary: FailureAlertIntegrationTests, LineageRunStateProbe_
  - _Depends: 3.1, 3.2, 4.1_

- [ ] 6.3 一鍵彙總執行的本地驗收
  - 在本地 kind cluster 執行 `run-all-failure-scenarios.sh`；確認 9 失敗 + schema-mismatch 全綠、ledger 自動更新、coverage check 通過。
  - **Observable**: 退出碼 0；`docs/validation-report.md` 的 ledger 行 timestamp 全部為當次執行。
  - _Requirements: 4.2, 5.2, 5.3, 5.4_
  - _Boundary: RunAllFailureScenarios, CoverageCheckExt_
  - _Depends: 4.2, 4.3, 6.1, 6.2_

## Requirements Coverage Map

| Requirement | Tasks |
|-------------|-------|
| 1.1 | 2.1 |
| 1.2 | 1.3, 2.2 |
| 1.3 | 1.1, 2.1 |
| 1.4 | 1.1, 4.2 |
| 2.1 | 2.2, 2.4, 4.1 |
| 2.2 | 1.2, 2.2 |
| 2.3 | 2.4 |
| 2.4 | 2.4 |
| 3.1 | 6.1 |
| 3.2 | 6.1 |
| 3.3 | 6.1 |
| 3.4 | 6.1 |
| 3.5 | 3.1, 6.2 |
| 3.6 | 3.1, 3.2, 6.2 |
| 4.1 | 2.3, 4.1, 6.1 |
| 4.2 | 2.3, 4.2, 6.3 |
| 4.3 | _（dashboard 顯示為手動觀察，無 code 任務；於 6.3 本地驗收時一併目視確認）_ |
| 4.4 | 4.1, 6.1 |
| 5.1 | 5.1 |
| 5.2 | 5.2, 4.3, 6.3 |
| 5.3 | 4.2, 6.3 |
| 5.4 | 4.3, 6.3 |
