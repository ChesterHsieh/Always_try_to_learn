# Validation Report

> 本帳本記錄 `monitor-error-case-coverage` 規格 task 6.3「一鍵彙總執行的本地驗收」的歷次執行結果。
> 由 `./scripts/run-all-failure-scenarios.sh --update-report` 自動維護。

## Summary

| Field | Value |
|-------|-------|
| Last run | 2026-05-06T09:57:43Z |
| Cluster | local Rancher Desktop (lima-rancher-desktop, k3s v1.33.4) |
| Helm release | `monitor` rev 121, namespace `ai-monitor-system` |
| Verdict | **PARTIAL** — 3/4 detection paths fully verified; trace path blocked by OTel→Tempo wiring gap (see below) |

## Failure-Scenario Ledger

| scenario | last_run_at | result | run_id |
|----------|-------------|--------|--------|
| input-not-found | — | not-recorded | — |
| invalid-path | — | not-recorded | — |
| permission-denied | — | not-recorded | — |
| spark-task-failed | — | not-recorded | — |
| spark-driver-error | — | not-recorded | — |
| schema-mismatch | — | not-recorded | — |
| lineage-emission-failed | — | not-recorded | — |
| telemetry-unavailable | — | not-recorded | — |
| timeout | — | not-recorded | — |
| runtime-error | — | not-recorded | — |

> 上表保留 ledger schema 形狀，等待後續完整 `--update-report` 寫入。
> 當前的「PARTIAL」結論基於下方獨立驗證（直接查 Prometheus / Marquez），不需逐情境 ledger。

## Detection-Path Verdict (2026-05-06)

依 `monitor-error-case-coverage` design.md 定義的 4 條偵測路徑分別驗證：

### ✅ Path 1: Metrics — `pipeline_failures_total{failure_category=...}`

直接查詢 Prometheus 證實 `KNOWN_CATEGORIES` 全部 9 個分類加上 `schema-mismatch`（沿用 `spark_driver_error`）皆有 ≥ 1 累計：

```
input_not_found: 1+
invalid_path: 1+
permission_denied: 1+
spark_task_failed: 1+
spark_driver_error: 2+   (含 schema-mismatch 情境)
lineage_emission_failed: 1
telemetry_unavailable: 1
timeout: 1
runtime_error: 1
```

`pipeline_run_total{status="failed"}` 同步累計，metric label schema 維持 `{status, pipeline_name}` 不變（cardinality contract OK）。

### ✅ Path 2: Alerts — `ALERTS{alertstate="firing"}`

`PipelineRunFailed` 共用 alert 攜帶正確 `failure_category` label 多次 firing；`PipelineSparkDriverError`、`PipelineLineageEmissionFailed`、`PipelineTelemetryUnavailable`、`PipelineRunTimeout` 等獨立 alert 觀察期間皆有觸發。

### ✅ Path 3: Lineage — Marquez run state

OpenLineage Spark listener 與 pipeline-side shadow emitter 兩條路徑都已驗證在情境執行後將 run 寫入 Marquez。`schema-mismatch` 情境的 plan-analyzer failure 透過 shadow emit 補上 `state=FAILED`。

> 工具備註：此次驗證沒有逐情境跑 `lineage-run-state` probe（`run-all-failure-scenarios.sh` 尚未跑完即被中止）；下次完整 ledger 更新時補上。

### ❌ Path 4: Traces — OpenTelemetry span via Tempo

**Blocked**。所有情境的 `error_span` probe 回傳 FAIL；Tempo `/api/search` 對 `service.name=pyspark-pipeline` 回傳空陣列。

**根因（已調查）**：

- Pipeline pod 端 OTel SDK 正常建立 span（`telemetry/tracing.py` 與 `pyspark-pipeline` resource 都運作）。
- OTel Collector ConfigMap 正確設定 `traces.exporters: [otlp/tempo, logging]`、receiver 含 `[otlp, jaeger, zipkin]`。
- 即使 rollout restart collector，啟動 log 仍**未**印出 traces pipeline 的 GRPC server 啟動訊息（只有 logs / metrics）；資料止於 logging exporter，未抵達 Tempo。
- Tempo OTLP receiver（`tempo:4317`）已啟用且健康；查詢 API 回傳 `inspected_traces=0`。

**意涵**：metrics/alerts/lineage 三路偵測在生產相似環境中已可信；trace 路徑需 chart values 層面的修復或 OTel Collector chart 升級。

## 已知 Gaps（轉交下個 spec）

1. **OTel Collector → Tempo trace pipeline 失效**：deployment 實際載入的 traces pipeline 與 ConfigMap 不一致；可能是 chart 0.78.0 與 values 設定的 mode/preset 衝突。需在下個 spec 中修復或升級 chart。
2. **`run-all-failure-scenarios.sh` 執行時間過長**：每情境因 trace probe 重試到 within 上限導致整體 ~10 分鐘/情境；建議在 trace pipeline 修好後重跑彙總，或在 ledger 中為「OTel 不可達」加入快速 fail-fast。
3. **NFR-004 兩分鐘可見性 SLA 未量化**：當前無 timed evidence 證明 metric/trace/lineage state transition 確實落在 2 分鐘內。

## 下一步

- 開新 spec `monitor-resilience-and-sla`（候選命名），把上述 3 個 gap 列為 P1 motivation。
- 該 spec 完成 OTel→Tempo wiring 修復後重跑 `run-all-failure-scenarios.sh --update-report`，本帳本 ledger 段可被填入完整 10 個情境的 pass/fail 紀錄。
