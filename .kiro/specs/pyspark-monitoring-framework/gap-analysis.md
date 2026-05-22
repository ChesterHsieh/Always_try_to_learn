# Gap Analysis: PySpark Monitoring Framework

**Spec**: `.kiro/specs/pyspark-monitoring-framework/`
**Source codebase**: `ai-monitor-system/`
**Reference requirements**: `requirements.md`（12 個 Requirement）
**Analysis date**: 2026-04-17
**Language**: zh-TW（依 `spec.json.language`）

---

## 摘要（Executive Summary）

- **整體實作狀態**：骨架（scaffolding）與工件（artifacts）多數齊全，但**執行期整合（runtime wiring）大量缺失**，估約 **40% 已交付實際可觀測行為**，60% 為「檔案存在但未串接後端」。
- **最大斷裂點**：lifecycle payload 在 `pipeline/job.py` 內建構但**從未送出**（無 `prometheus_client`、無 OTel span 真正建立、無對 Marquez 的 HTTP 發送驗證），導致 Requirement 1 / 2 / 4 / 7 / 10 的 acceptance criteria 在實際執行期無法被驗證。
- **配置斷層**：Prometheus 的 scrape target 指向 `pyspark-pipeline:8000`，但 pipeline 從未開啟該埠；告警規則與儀表板 JSON 雖存在於 repo，但**未被 mount 進 Prometheus/Grafana** 的上游 chart releases。
- **覆蓋與韌性**：`check-monitoring-coverage.sh` 漏檢 Marquez（OpenLineage backend），且無 `chart_version` 報告；Requirement 11 的韌性面向（pod 重啟、後端短暫離線、亂序事件）幾乎無對應實作或測試。
- **文件落差**：`tasks.md` 內 T001–T044 全部標記為 `[x]`，但其中至少 T017、T021、T026、T027、T029、T030、T036、T040 對應的「實際發送/載入/驗證」未完成；steering 與 spec 主軸已對齊 Option B，但實作層的「上游 chart 與專案 overlay」邊界仍有落差。

---

## 1. Current State Investigation

### 1.1 Repository 結構與實作邊界

實作對齊 `.kiro/steering/structure.md` 所定義的邊界：

| 區塊 | 路徑 | 角色 | 實作狀態 |
| --- | --- | --- | --- |
| Pipeline 執行期 | `ai-monitor-system/pipeline/` | run identity、I/O、telemetry envelope、tracing、lineage event、failure 分類、PySpark job | 7 個 .py 檔，run 結構完整；**但僅是 payload 構造，未實際送出** |
| 部署 | `ai-monitor-system/deploy/helm/` | upstream chart 組合、values 與 overlays、專案模板 | Chart.yaml 釘版 4 個 upstream chart；values + local-minimal 完整；templates 含一個重要矛盾（見 1.4） |
| 部署腳本 | `ai-monitor-system/deploy/scripts/` | bootstrap / run / coverage / smoke 入口 | 4 支腳本皆可執行，但 coverage 檢查浮淺、缺少對 Marquez 與 chart_version 的驗證 |
| 監控資產 | `ai-monitor-system/monitoring/` | dashboards、alert rules、otel/prometheus/grafana 配置 | 檔案皆存在，但**並未被 Helm chart 引入進 Prometheus / Grafana 的 provisioning 路徑** |
| 測試 | `ai-monitor-system/tests/` | contract / integration / smoke | 結構完整、皆有 assertion；**但多為 payload 結構與檔案存在性檢查，缺少 runtime 行為驗證** |
| 文件 | `ai-monitor-system/docs/` | onboarding、runbook、chart matrix、OL config、validation report | 多數完整；`validation-report.md` 為 stub |

### 1.2 已可重用的核心元件（Reusable Assets）

- `pipeline/run_context.py`：`RunContext` dataclass + `validate()`，已強制 `run_id` 必填與 status 列表；可作為 cross-signal 關聯的單一來源。
- `pipeline/telemetry.py`：`build_otel_attributes`、`build_signal_envelope`、`build_correlation_attributes`、`lifecycle_metric_payload` 已具備 envelope 構造能力；缺的是「emit / dispatch」。
- `pipeline/lineage.py`：`build_openlineage_event` 已能輸出符合 OpenLineage schema 的 JSON（含可選的 trace facet），可重用於 OL HTTP transport 或備援路徑。
- `pipeline/failure_classifier.py`：4 種類別的決定性對應，已被 `job.py` 與 contract test 引用，可向上擴展（見 §2.3）。
- `deploy/helm/Chart.yaml`：已釘版 4 個 upstream chart（`prometheus@25.27.0`、`grafana@8.5.1`、`opentelemetry-collector@0.78.0`、`ilum-marquez@6.7.0`），與 Option B 主軸一致。
- `monitoring/dashboards/*.json` 與 `monitoring/alerts/*.yaml`：已具雛形，可直接灌入 upstream Grafana 的 dashboard provisioning 與 Prometheus 的 rules 載入路徑。

### 1.3 慣例（Conventions）

- 命名空間單一：`ai-monitor-system`（global.namespace 統一）。
- run identity：`run_id` 為 UUID，於 `job.run_pipeline()` 內生成並貫穿所有 payload；OpenLineage namespace 與 Spark listener namespace 對齊為 `ai_monitor_system`。
- 測試分層：`contract/`（payload schema）、`integration/`（跨模組行為）、`smoke/`（artifact + 入口腳本存在性）；`conftest.py` 將 repo root 加入 `PYTHONPATH`。
- ruff 設定：`line-length=100`、`py311`、規則集 `E,F,I,B`。
- 部署腳本一致透過 `helm upgrade --install`，並以 `values.local-minimal.yaml` 為偏好 overlay。

### 1.4 整合面（Integration Surfaces）與一個關鍵矛盾

- **OpenLineage Spark listener**：在 `pipeline/job.py` 內以 `spark.extraListeners`、`spark.openlineage.transport.url` 等官方 key 配置；後端透過 `openlineage-configmap.yaml` 的 ConfigMap 指向 `ai-monitor-system-upstream-marquez:9555`。listener 已被「裝載」，但實際事件是否抵達 Marquez **沒有驗證鏈路**。
- **OTel Collector**：上游 chart 已啟用，並有專案自有 `monitoring/otel/collector-config.yaml` 描述 OTLP receiver → batch → `debug` exporter；但 collector 的實際部署仍同時有一份 `templates/monitoring-stack.yaml`（199 行，描述自有 Prometheus / Grafana / OTel Collector / 服務）。**這代表上游 chart 與專案自有的 monitoring-stack 模板存在功能重疊**，是 Option B 邊界尚未徹底執行的明確訊號（見 §3）。
- **Prometheus 抓取**：`monitoring/prometheus/prometheus.yml` 設定 `targets: ["pyspark-pipeline:8000"]`，但 pipeline pod 從未開啟 8000 埠；上游 Prometheus chart 也沒有將此檔灌入 `additionalScrapeConfigs`。
- **Grafana datasource & dashboard**：`monitoring/grafana/datasources.yaml` 為合法 provisioning 格式，但**無 ConfigMap / sidecar / `grafana.dashboards` values 將其灌入 release**；dashboards 同樣未透過 sidecar 或 ConfigMap 自動載入。
- **告警規則**：`monitoring/alerts/*.yaml` 為合法 Prometheus rules，但 chart values 並未指定 `serverFiles.alerting_rules.yml` 或等價的載入機制。

---

## 2. Requirements Feasibility Analysis

### 2.1 Requirement-to-Asset Map（精要表）

> 標記：✅ 已具備且可運作；⚠️ 結構存在但欠缺 runtime 串接；❌ 尚未實作；🔍 Research Needed

| Requirement | 主要對應實作 | 狀態 | 缺口分類 |
| --- | --- | --- | --- |
| R1 Run-Level Visibility | `pipeline/{run_context,job}.py`、`templates/pipeline-job.yaml` | ⚠️ | Missing: 指標未 expose、無 `/metrics` 端點 |
| R2 Failure Signals | `pipeline/{failure_classifier,job}.py`、`monitoring/alerts/pipeline-failure-rules.yaml` | ⚠️ | Missing: 告警規則未載入；alert payload 缺 `failure_category` / `run_id` |
| R3 Lineage & Run Context | `pipeline/lineage.py`、Spark listener config、Marquez chart | ⚠️ | Unknown: 事件實際送達率（🔍） |
| R4 Cross-Signal Correlation | `pipeline/{telemetry,tracing,lineage}.py` | ⚠️ | Missing: OTel span 未建立；query-time 關聯未驗證 |
| R5 Standardized Stack via Upstream Charts | `deploy/helm/Chart.yaml`、`Chart.lock`、`charts/` | ⚠️ | Constraint: 自有 `monitoring-stack.yaml` 模板與上游 chart 功能重疊 |
| R6 Operator Dashboards | `monitoring/dashboards/*.json` | ⚠️ | Missing: 未經 Grafana provisioning 自動載入；缺失敗 triage 視圖 |
| R7 Alerting Workflow | `monitoring/alerts/*.yaml` | ⚠️ | Missing: 規則未載入；告警 payload 欄位不足 |
| R8 Repeatable Onboarding | `deploy/scripts/bootstrap-local.sh`、`docs/onboarding-monitoring.md`、`docs/runbook.md` | ✅ | 強化：troubleshooting 與 SLA 計時 |
| R9 Coverage Profile | `deploy/scripts/check-monitoring-coverage.sh`、`docs/chart-version-matrix.md` | ⚠️ | Missing: Marquez 檢查、`chart_version` 回報、`last_verified_at` 紀錄 |
| R10 Required Metrics | （無）`prometheus_client` 整合不存在 | ❌ | Missing: 全部 5 個指標族未實際 expose |
| R11 Resilience | （無對應） | ❌ | Missing: 後端離線/亂序/重啟下的可觀測行為 |
| R12 Quality / Tests / Docs | `tests/{contract,integration,smoke}/`、`docs/` | ⚠️ | 強化：runtime 端到端、validation-report 補完 |

### 2.2 主要缺失能力（Missing Capabilities）

1. **指標 expose 端點**：缺 `prometheus_client`（或對等）整合，pipeline pod 沒有任何 `/metrics` HTTP 服務；Prometheus 因此無法抓取 R10 列出的所有指標族。
2. **OTel SDK 真正的 span 建立**：`tracing.py` 僅產生 attribute dict，未呼叫 `trace.get_tracer().start_as_current_span(...)`；OTel Collector 因此沒有 span 可接收，也無法跨 metrics / lineage 進行 trace context 傳播。
3. **OpenLineage 事件送達驗證**：listener 已配置 HTTP transport，但缺少 smoke 測試或 health probe 確認事件抵達 Marquez（例如查 `/api/v1/namespaces/{ns}/jobs` 或 `/runs`）。
4. **告警規則與儀表板的 chart-side wiring**：upstream Prometheus chart 預期透過 `serverFiles.alerting_rules.yml` 或 `extraConfigmapMounts` 載入；upstream Grafana chart 透過 `dashboardProviders` + `dashboardsConfigMaps`/sidecar。目前 values 都未啟用這些路徑。
5. **scrape job 設定**：上游 Prometheus chart 的 `extraScrapeConfigs` / `additionalScrapeConfigs` 未指定 pipeline job 的 scrape；同時當前自有的 `prometheus.yml` 中 target port 與 pipeline 實際暴露的 port 不一致。
6. **Coverage check 缺項**：`check-monitoring-coverage.sh` 缺少對 Marquez service / `/api/v1/health` 的檢查、缺對 4 個 chart 的版本回報、缺對 alert rules 是否實際被 Prometheus 載入（`/-/reload` 後 `groups` 是否非空）的探測。
7. **Failure 分類擴展**：當前僅 4 類，缺 Spark task / driver / 後端離線等 Spark 特定分類，造成 R2.3 的 “一致語意” 在現實情境下分辨度不足。
8. **韌性測試與行為**：R11 對應的「後端瞬時不可用、pod 重啟、lineage 晚到」皆無實作或自動化測試；`run-pipeline.sh` 的 timeout 為固定 300s，無 SLA 計時與失敗回報。

### 2.3 約束與重要決定（Constraints）

- **Steering 規範**：`tech.md` 明確要求「以 upstream chart 為核心，專案模板僅為 pipeline 與整合 overlay」；`structure.md` 要求「監控資產獨立於 pipeline 程式」；本分析的選項評估必須遵循這兩條。
- **本機資源預算**：`values.local-minimal.yaml` 已為 4 個 upstream 元件設定保守的 requests/limits；任何新增 sidecar、agent 或進程都需與此預算協同。
- **Python 3.11、PySpark 3.5、OL Spark listener 1.45.0**：版本鏈已綁定於 Dockerfile 與 chart matrix，新增整合須與此版本相容。
- **無外部 SaaS**：v1 不引入 SaaS observability 後端；指標、trace、lineage 全部需在本機叢集自洽。

### 2.4 Research Needed（轉交設計階段）

- 🔍 **OTel Collector 在本機 minimal profile 下的最小可信 exporter pipeline**：是否以 `prometheus` exporter 將收到的 OTel metrics 暴露給 Prometheus（pull），或以 `prometheusremotewrite` 推到 Prometheus；trace 端是否需要 Tempo / Jaeger，或在 v1 僅以 OTel Collector logging exporter 取得 SLA 可見度？
- 🔍 **OpenLineage 事件健康檢查 API**：Marquez 在 6.7.0 chart 提供的 `/api/v1/health` 與 `/api/v1/namespaces/...` 端點是否足以作為 coverage check 與 R3 acceptance 的驗證依據？回應結構是否穩定？
- 🔍 **Prometheus chart 的告警規則載入路徑**：在 25.27.0 中，`serverFiles.alerting_rules.yml`、`additionalPrometheusRulesMap`、與獨立 `prometheus-rules` ConfigMap 三條路徑哪一條最符合 Option B 邊界、且不需大幅改動 Helm 值結構？
- 🔍 **Grafana chart 的 dashboard provisioning 模式**：sidecar（`sidecar.dashboards.enabled=true` + `searchNamespace`）與顯式 `dashboardConfigMaps` 兩種模式，何者在 local-minimal 下成本最低？
- 🔍 **指標暴露策略**：在 PySpark driver process 中 `prometheus_client.start_http_server()` 與 Spark JMX exporter / OL Spark listener 的 metrics facet，何者更貼近 R10 的指標族與 R4 的 run_id 標籤？
- 🔍 **韌性測試最小可行集**：在不引入完整 chaos 框架前，能否以 `kubectl delete pod` + 後端短暫 scale 0 兩種腳本場景，覆蓋 R11.1 / R11.2 / R11.4 的最低證據？

### 2.5 Complexity Signals

- 整體偏向「**多元件配置整合 + 既有 payload 構造已就位**」，非演算法問題。
- 主要複雜度來源：(1) 上游 chart 的 values 模板與 sidecar 行為差異；(2) PySpark driver process 內以同一進程暴露 `/metrics` 並啟動 OTel SDK 的相依與生命週期；(3) 多訊號的 query-time 關聯（query 設計 + label model）。

---

## 3. Implementation Approach Options

下列三案皆「以遵循 Option B（上游 chart 為主）為前提」，差異在於專案內現有資產（dashboards、alerts、`monitoring-stack.yaml`、coverage 腳本、pipeline 程式）的處置策略。

### Option A — 擴展既有元件（Extend）

**何時適用**：希望以最小變動修補現有 wiring，避免新增大量檔案。

- 擴展檔案重點
  - `deploy/helm/values.yaml`：補上 `prometheus.serverFiles.alerting_rules.yml` 內含告警；以 `additionalScrapeConfigs` 增加對 pipeline job 的 scrape；於 `grafana.dashboardConfigMaps` 與 `grafana.dashboardProviders` 顯式列出既有 dashboard JSON 對應的 ConfigMap。
  - `deploy/helm/templates/pipeline-job.yaml`：在 pipeline pod 暴露 `metrics` containerPort，並透過 annotation `prometheus.io/scrape=true`（或對應 chart 的 service-monitor 路徑）。
  - `pipeline/job.py` + `pipeline/telemetry.py`：在 driver 起 HTTP `start_http_server(port)`，註冊 R10 列出的指標族，於 lifecycle 點寫入 `Counter` / `Histogram`。
  - `pipeline/tracing.py`：補上 OTel SDK `TracerProvider` + OTLP exporter 的初始化，並於 `run_pipeline()` 包覆 `start_as_current_span()`。
  - `deploy/scripts/check-monitoring-coverage.sh`：補入對 Marquez `/api/v1/health` 的 curl 檢查、四個 release 的 `helm get metadata` 版本回報、以及對 Prometheus `/api/v1/rules` 的非空驗證。
  - `monitoring/alerts/pipeline-failure-rules.yaml`：在 annotations 中加入 `run_id`、`failure_category`、`failure_message` 模板。

- 相容性
  - 既有 contract / integration / smoke 測試仍可保留；對 payload 結構不破壞。
  - **風險**：`templates/monitoring-stack.yaml`（199 行）與上游 chart 功能重疊，若未一併處理，會有重複資源（兩套 prometheus/grafana/otel 服務名衝突或競合）。

- 複雜度與可維護性
  - 變動集中於既有檔案；認知成本中等。
  - 風險於 `pipeline/job.py` 同檔案中堆積（指標、trace、lineage 三件事），需注意單一職責；若行數迅速增加，建議在 Option C 中拆成多模組。

**Trade-offs**
- ✅ 變動少、最快可看到 R1/R2/R10 的 runtime 結果。
- ✅ 完全沿用既有測試骨架。
- ❌ 若不一併刪除/縮減 `monitoring-stack.yaml`，會違反 Option B 的「以 upstream chart 為單一來源」邊界。
- ❌ `job.py` 易膨脹。

---

### Option B — 建立新元件（Create new modules）

**何時適用**：希望明確切出新責任邊界、提升 testability 與長期演進性。

- 新增/重構責任
  - 新模組 `pipeline/metrics.py`：負責所有 `prometheus_client` 物件的宣告、HTTP server 啟停、與 lifecycle 點的紀錄 API（`record_run_started / record_run_succeeded / record_run_failed`）。
  - 新模組 `pipeline/otel_setup.py`：集中 OTel Tracer / Meter Provider 初始化、resource attributes、與 OTLP exporter 設定；`tracing.py` 改為純 span helper。
  - 新模組 `pipeline/lineage_emitter.py`：以 OL HTTP client 包裝 `lineage.py` 的 event；提供與 Spark listener 並行的事後驗證或備援送出（後者僅用於測試/驗收）。
  - 新模組 `pipeline/coverage.py`（CLI 入口）：以 Python 實作 coverage 檢查，覆蓋 Marquez health、Prometheus rules、Grafana datasource health、四個 chart 的版本；shell 腳本退化為入口呼叫。
  - 新模板 `deploy/helm/templates/monitoring-config-bundle.yaml`：將 `monitoring/alerts/*.yaml` 與 `monitoring/dashboards/*.json` 包成 ConfigMap，並對應 upstream chart values 的引用。
  - 廢止/縮減 `templates/monitoring-stack.yaml`：保留必要的 service alias 或刪除，避免與上游 release 衝突。

- 整合點
  - Coverage Python 模組可直接被 contract test 引用，產出 R9.4 `last_verified_at` 的記錄檔（例如 `.kiro/specs/.../coverage-runs/<ts>.json`）。
  - 新 emitter 模組為韌性測試（R11）提供注入點。

- 責任邊界
  - `pipeline/job.py` 只負責 orchestration；指標、trace、lineage、failure 分類各自獨立檔案，符合 steering 的 “小檔高內聚” 原則。

**Trade-offs**
- ✅ 邊界清晰、單元測試容易（可 mock emitter / metrics registry）。
- ✅ 為韌性與覆蓋驗證提供結構化注入點。
- ❌ 檔案數增加；變更面更廣，影響回歸風險。
- ❌ 需重寫部分既有 integration test（mocking 模式改變）。

---

### Option C — 混合（Hybrid，**建議優先評估**）

**何時適用**：當前狀態為「結構大半就位，但缺 emitter 與 chart-side wiring」，混合做法能在 1–2 週內推進 Option B 主軸並控制重寫面。

- 組合策略（建議）
  - **Phase H1（最小可運行可見性）**：採 Option A 的 `values.yaml` chart-side wiring（rules、dashboards、scrape config），同時在 pipeline 端拆出 **新模組** `pipeline/metrics.py`（Option B）；首要打通 R1 / R2 / R10 / R7 的 runtime 鏈路。
  - **Phase H2（trace 與 query-time 關聯）**：拆出 `pipeline/otel_setup.py`（Option B），補上 Tempo/Jaeger 或暫以 Collector logging exporter；新增 query-time correlation 的 integration test，以 `run_id` 在 Prometheus / OL backend 同時被觀察到為通過條件。
  - **Phase H3（Option B 邊界完成）**：縮減/刪除 `templates/monitoring-stack.yaml`（Option A 的清理面），改以 upstream chart values 與一個 thin `monitoring-config-bundle.yaml` ConfigMap（Option B）為單一來源；補完 `coverage.py` 與 `validation-report.md`。

- Phased Implementation
  - 每個 Phase 都有獨立 smoke 測試入口，能單獨回退（H2/H3 可暫退至 H1 狀態而不破壞 R1/R2/R10 的 MVP 可見性）。
  - 採 feature flag（chart values 中 `enableSelfManagedMonitoringStack=false` / `enableUpstreamWiring=true`）切換新舊路徑。

- 風險緩解
  - 新增可關閉旗標：在 H1 不立即移除 `monitoring-stack.yaml`，避免一次性破壞既有 demo。
  - 將「事件確實抵達後端」的健康檢查腳本提早到 H1 完成，作為 H2/H3 的迴歸守門。

**Trade-offs**
- ✅ 風險可控：MVP 可在 H1 落地，後續 Phase 不阻擋發行。
- ✅ 同時實踐 Option B 邊界與 “新模組分擔職責” 兩個目標。
- ❌ 計畫複雜度與多階段協調成本較高。
- ❌ 若不嚴格控管 chart 值的相容性，會出現 H1/H3 的 schema 變動。

---

## 4. Out-of-Scope（Defer to Design）

以下項目應於設計階段（`/kiro:spec-design`）展開，本次 gap 分析僅以「Research Needed」記入：

- 對 Tempo / Jaeger / Loki 的選型與是否納入 v1。
- 是否引入 ServiceMonitor / PodMonitor（kube-prometheus-stack）作為替代方案。
- 韌性測試框架選型（litmus / chaos-mesh / 純 kubectl 腳本）。
- 與後續可能的 streaming pipeline / 多叢集場景之相容性（非 v1）。

---

## 5. Implementation Complexity & Risk

| 工作叢集 | Effort | Risk | 一句話理由 |
| --- | --- | --- | --- |
| chart-side wiring（rules / dashboards / scrape） | M | Medium | 上游 chart 路徑已知，但 4 個 release 的 values schema 需逐一驗證 |
| pipeline 端 `prometheus_client` 與 R10 指標族 | S | Low | 既有 lifecycle hook 已就位，新增 metrics 模組為標準工程任務 |
| OTel SDK Tracer/Meter Provider 與 OTLP exporter | M | Medium | 與 Spark driver 生命週期、Collector pipeline 設定相依 |
| OL 事件抵達驗證 + Marquez health 檢查 | S | Low | 端點熟悉、curl 即可，但需與 chart 服務名相依 |
| `templates/monitoring-stack.yaml` 縮減/移除 | M | Medium | 涉及與既有 demo 的相容性，需 feature flag 漸進化 |
| Coverage CLI 化 + `chart_version` 報告 | S | Low | 替換為 Python CLI，邏輯直觀 |
| Failure 分類擴展（Spark/後端） | S | Low | 在 `failure_classifier.py` 內擴充對應；需與測試同步 |
| query-time correlation integration test | M | High | 牽涉 PromQL、OL API、Spark driver 啟停的同步測試環境 |
| 韌性測試（pod 重啟、後端離線、亂序） | L | High | 需建立可重入的腳本場景與穩定的等待/收斂判讀 |
| `validation-report.md` 補完與 SLA 計時 | S | Low | 文件 + 測試聚合，工程量低 |

---

## 6. Recommendations for Design Phase

### 6.1 偏好方向

- **採 Option C（Hybrid）**，並以 `enableSelfManagedMonitoringStack` / `enableUpstreamWiring` 兩個 chart-level flag 切割三個 Phase；H1 為 MVP 可見性，H2 為 trace 與 query-time correlation，H3 完成 Option B 邊界與覆蓋報告。
- **指標端**：在 Spark driver process 內以 `prometheus_client.start_http_server(port)` 暴露 R10 五個指標族；Helm values 透過 `additionalScrapeConfigs` 指定 pipeline 的 service/port，並在 alert rules 的 annotations 注入 `run_id` / `failure_category` / `failure_message`。
- **Trace 端**：在 v1 採「OTel Collector + logging exporter」作為最小可信骨架，但 Pipeline 端需真正建立 span（含 `run_id`、`pipeline_name`、`k8s_namespace`、`status`），確保 R4 的契約在 SDK 層落地；Tempo / Jaeger 列為設計階段選項。
- **Lineage 端**：保留 OL Spark listener 為主送出路徑，新增 Marquez `/api/v1/health` 與「最近 N 筆 runs 內含特定 `run_id`」兩條健康檢查，作為 R3 / R9 的客觀驗證。
- **Dashboards/Alerts wiring**：以 ConfigMap + 上游 chart 的 dashboardProviders / serverFiles 路徑載入，**移除** `templates/monitoring-stack.yaml` 中與上游重疊的部分，明確化 Option B 邊界。
- **Coverage**：將 shell 腳本退化為 Python `pipeline/coverage.py` 的 thin entry，輸出 `coverage-report.json`（含 `chart_version`、`last_verified_at`、各檢查通過/失敗），與 `tasks.md` 後續 release 流程串接。

### 6.2 設計階段需先決定的關鍵點

1. 指標 expose 的 port、service exposure 與 `additionalScrapeConfigs` 的最終格式（決定 chart values 結構）。
2. Trace pipeline 是否在 v1 引入 Tempo/Jaeger，抑或僅以 Collector exporter 留證據鏈。
3. `templates/monitoring-stack.yaml` 的縮減幅度（完全移除 vs 退化為 service alias）與 feature flag 命名。
4. Coverage 報告的存放路徑與保留策略（repo 內 vs 跨 release 累計）。
5. `failure_classifier` 擴展類別清單（與 Spark 例外實際命名對齊）。

### 6.3 帶到設計階段的 Research 項

- 🔍 OTel Collector 對 Prometheus 的 metrics 出口策略（`prometheus` vs `prometheusremotewrite`）。
- 🔍 Marquez 6.7.0 的健康與查詢端點穩定性與最小可信 contract。
- 🔍 Prometheus 25.27.0 的 rules / scrape config 載入慣例與 values schema。
- 🔍 Grafana 8.5.1 的 dashboard provisioning 路徑（sidecar vs 顯式 ConfigMap）。
- 🔍 PySpark driver 行內 `prometheus_client` HTTP server 與 Spark 驅動進程的生命週期相容性。
- 🔍 R11 韌性測試的最小腳本場景（不引入完整 chaos 框架）。

---

## 7. 與 `tasks.md` 完成度的對照

`/specs/001-pyspark-monitoring-framework/tasks.md` 中 T001–T044 全部標記為 `[x]`，但 gap 分析發現以下任務在 “runtime 行為” 層仍未完成，建議於設計與後續 tasks 重新切片：

- T017（lifecycle 發送）：payload 構造完成，**未實際發送**。
- T021 / T030（dashboard 擴展）：JSON 存在，**未經 Grafana provisioning 載入**；缺失敗 triage 與根因瀏覽路徑。
- T026（Spark listener 接線）：listener 已配置，**事件抵達 Marquez 未驗證**。
- T027（trace span 屬性）：屬性建構完成，**未實際建立 span**。
- T029（OTel Collector 上游 chart 遷移）：values 已備，**exporter 為 debug**，無有效 outbound。
- T036（coverage check）：腳本可執行，但**漏 Marquez、漏 chart_version、漏 rules/dashboards 載入驗證**。
- T040（validation report）：`docs/validation-report.md` 為 stub。

> 注意：上述觀察 **不否定** `tasks.md` 的設計切割合理性；它反映的是「將任務的 Definition of Done 從 ‘檔案存在’ 升級為 ‘可被觀測到’」的需要，將在設計階段轉換為更嚴格的 acceptance。
