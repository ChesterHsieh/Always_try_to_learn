# Research & Design Decisions

## Summary

- **Feature**: `pyspark-monitoring-framework`
- **Discovery Scope**: Extension（既有 `ai-monitor-system/` 之 brownfield 強化；採 light discovery）
- **Key Findings**:
  - **Prometheus chart 25.27.0**：alerting rules 寫入 `serverFiles."alerting_rules.yml"` 即被預設 `rule_files` 引入；新增 scrape job 應使用 `extraScrapeConfigs`（YAML 字串），避免覆蓋 chart 預設的 kubernetes 自動發現。`additionalPrometheusRulesMap` 屬於 `kube-prometheus-stack`，**不適用**本 chart。
  - **Grafana chart 8.5.1**：sidecar 模式（`sidecar.dashboards.enabled` / `sidecar.datasources.enabled`）每類各加一個 `kiwigrid/k8s-sidecar` container（≈ 50–100 Mi），**對 local-minimal profile 過重**；改採顯式 provisioning（`datasources` + `dashboardProviders` + `dashboardsConfigMaps`）以維持單 pod。
  - **OTel Collector chart 0.78.0**：必須顯式設定 `mode`（`deployment` 為本 v1 預設）；以 `prometheusremotewrite` exporter 將 metrics 推到上游 Prometheus（並啟用 `--web.enable-remote-write-receiver`），比 `prometheus` exporter（pull）少一條 scrape 鏈，符合 local-minimal 邊界。
  - **Marquez via `ilum-marquez@6.7.0`**：API service 為 `ai-monitor-system-upstream-marquez:9555`；`/healthcheck` 與 `/ping` 在 admin port 5001（pod-internal、Service 不暴露）；coverage script 應改用 `GET /api/v1/namespaces` 與 `GET /api/v1/events/lineage?limit=...` 作為 API-port 的健康/活性訊號。
  - **Pipeline 端的 emission 缺口**：lifecycle/trace/lineage payload 已就位，但缺 `prometheus_client` HTTP 端點、缺 OTel SDK Tracer/Meter 初始化、缺 OL 事件 fallback emitter；屬「runtime wiring」型缺口，可藉由新增小模組與 chart values 完成。
  - **Option B 邊界落差**：`templates/monitoring-stack.yaml`（199 行）與上游 chart 功能重疊，需以 feature flag 漸進化縮減。

---

## Research Log

### Topic 1 — Prometheus chart 25.27.0：rules 與 scrape config 載入路徑

- **Context**：Requirement 7 / 10 需要將 `monitoring/alerts/*.yaml` 載入 Prometheus、並讓 Prometheus 抓取 pipeline pod 的 `/metrics`，且不違反「Option B 上游 chart 為主」邊界。
- **Sources Consulted**:
  - `prometheus-community/helm-charts` tag `prometheus-25.27.0` 的 `values.yaml`（lines 754–800、1207–1210）
- **Findings**:
  - `serverFiles."alerting_rules.yml"`（內含 `groups:`）為 chart 預設掛入路徑，`server.config` 預設的 `prometheus.yml.rule_files` 已包含 `/etc/config/alerting_rules.yml`，因此**只要把 rule groups 寫到此 key 即會被自動評估**。
  - `extraScrapeConfigs` 是「字串」型欄位，會被附加到 `prometheus.yml.scrape_configs`，**不覆蓋**預設的 kubernetes-apiservers / nodes / cadvisor / pods / services；若改寫 `serverFiles."prometheus.yml".scrape_configs` 則會**整段取代**預設，較不建議。
  - 在 umbrella chart 中所有 keys 都需放在 alias 名稱底下，例如 `upstream-prometheus.serverFiles."alerting_rules.yml"`、`upstream-prometheus.extraScrapeConfigs`。
  - `additionalPrometheusRulesMap` 為 `kube-prometheus-stack` 才有的 key，本 chart **不可用**。
  - 啟用 remote write receiver：`server.extraArgs."web.enable-remote-write-receiver": null`（無值意指純 flag）。
- **Implications**:
  - Option B 的 rules wiring **不需新增專案 ConfigMap**，只需將 `monitoring/alerts/*.yaml` 內容以 Helm `tpl + Files.Glob` 灌入 `upstream-prometheus.serverFiles."alerting_rules.yml"`。
  - scrape job 走 `extraScrapeConfigs`；對 pipeline pod 的 service discovery 可採 `kubernetes_sd_configs` 配合 namespace + label selector，避免 hardcode service:port。
  - 需在 design 中明確：**廢止** `monitoring/prometheus/prometheus.yml` 此檔（因為 chart 自帶 `prometheus.yml`），改為 alert rules 與 scrape config 兩段獨立的 YAML。

### Topic 2 — Grafana chart 8.5.1：dashboard / datasource provisioning

- **Context**：Requirement 6 / 7.3 需要 dashboards 與 datasource 自動載入，且不增加 local-minimal profile 的資源負擔。
- **Sources Consulted**:
  - `grafana/helm-charts` tag `grafana-8.5.1` 的 `values.yaml`（lines 615 / 724 / 743 / 780 / 859）
- **Findings**:
  - 兩個方案 **互斥**（values.yaml 中明示 sidecar 與 `dashboardProviders / datasources / dashboards` 不可同時使用）。
  - sidecar 模式：每類各加一個 `kiwigrid/k8s-sidecar` container（datasources + dashboards 共 +2 containers，約 100–200Mi 額外開銷）。
  - 顯式模式：
    - `datasources."datasources.yaml".datasources: [...]`
    - `dashboardProviders."dashboardproviders.yaml".providers: [{name, folder, type: file, options.path: /var/lib/grafana/dashboards/<name>}]`
    - `dashboardsConfigMaps.<provider_name>: <existing-cm-name>`（指向 repo 內 `monitoring/dashboards/*.json` 編譯的 ConfigMap）
- **Implications**:
  - 在 local-minimal 採顯式模式（**不啟用 sidecar**），於 umbrella chart 內新增一個 `templates/grafana-dashboards-configmap.yaml`，用 `tpl (.Files.Glob "monitoring/dashboards/*.json").AsConfig` 集成，並用 `dashboardsConfigMaps.default` 引用。
  - datasource 設定直接 inline 寫入 `upstream-grafana.datasources`，避免額外 file。

### Topic 3 — OTel Collector chart 0.78.0：mode 與 metrics 出口策略

- **Context**：Requirement 4 需要 trace 與 metrics 真正流通；既有 collector exporter 為 `debug`。本機環境希望最少元件達成 R10 指標族可被 Prometheus 查詢。
- **Sources Consulted**:
  - `open-telemetry/opentelemetry-helm-charts` tag `opentelemetry-collector-0.78.0` 的 `values.yaml`（lines 8 mode、95 config、269 ports.metrics）
  - OTel `prometheusremotewriteexporter` 文件（contrib repo）
- **Findings**:
  - `mode` 必填；本 v1 採 `deployment`（無需 daemonset 級別的節點覆蓋）。
  - `config:` 為 deep-merge：本專案僅需覆寫 `service.pipelines` 與 `exporters`。
  - 兩個 metrics 出口選項：
    - `prometheus`（pull）：collector 暴露 `/metrics`，需在 Prometheus 端新增 scrape job → 多一條依賴。
    - `prometheusremotewrite`（push）：collector 主動推 `/api/v1/write` → Prometheus 啟用 receiver flag 即可，**少一條 scrape**。
  - `presets.*.enabled` 預設皆為 false；保持不啟用以避免引入額外 receiver。
  - Service DNS：`<release>-prometheus-server.<ns>.svc:80`。
- **Implications**:
  - Design 採 `prometheusremotewrite`，endpoint `http://ai-monitor-system-prometheus-server.ai-monitor-system.svc:80/api/v1/write`。
  - Trace 在 v1 仍以 OTel Collector `debug` exporter 留存於 stdout（保持最小依賴），`logging` exporter 作為 SLA 證據；後續若引入 Tempo/Jaeger 屬於下一個 spec。
  - 需明示「OTel SDK 端」初始化責任：在 `pipeline/otel_setup.py` 新增 module（見 design 的 Components 區）。

### Topic 4 — Marquez (ilum-marquez 6.7.0)：健康端點與服務命名

- **Context**：Requirement 3 / 9 需要在 coverage script 中以 HTTP 客觀證明 lineage backend 可用且最近接收到事件。
- **Sources Consulted**:
  - `ilum-marquez-6.7.0.tgz`：`values.yaml`、`templates/marquez/service.yaml`、`templates/marquez/deployment.yaml`、`templates/_helpers.tpl`
  - Marquez OpenAPI 文件 (`marquezproject.github.io/marquez/openapi.html`)
- **Findings**:
  - Marquez 為 Dropwizard：API port 5000、admin port 5001；chart 的 Service **僅暴露 API port 5000 → 9555**；`/healthcheck`、`/ping` 在 admin port 5001（pod-internal）。
  - 在 cluster 內以 Service 進行 HTTP 探測時，建議改用以下 API：
    - `GET /api/v1/namespaces`：200 即視為 alive。
    - `GET /api/v1/events/lineage?limit=N`：取得最近 N 筆 lineage 事件，可以用來證明 “最近接收”。
    - `GET /api/v1/namespaces/{ns}/jobs` 與 `.../jobs/{job}/runs`：列出最近 runs，搭配 `run_id` 對齊。
  - Web UI Service 為 `ai-monitor-system-upstream-marquez-web`（port 9444），與 API service 分離。
  - Service 命名：`{Release.Name}-{alias}` → `ai-monitor-system-upstream-marquez`。
- **Implications**:
  - `check-monitoring-coverage` 應於 `coverage.py` 內透過 `requests.get("http://ai-monitor-system-upstream-marquez:9555/api/v1/namespaces")` 與 `/api/v1/events/lineage?limit=20` 進行檢查；不可仰賴 `/healthcheck`。
  - design 中應載明「Marquez admin port 在 v1 不開放於 Service」此邊界，避免 future tasks 嘗試錯誤路徑。

---

## Architecture Pattern Evaluation

| Option | 描述 | 優點 | 風險 / 限制 | 備註 |
|--------|------|------|--------------|------|
| A. Extend in-place | 直接在 `pipeline/job.py` 與既有 helm values 內補 emitter / wiring | 變動少、最快可見 | `job.py` 易膨脹；不易移除 `monitoring-stack.yaml`；違反 Steering 的 “小檔高內聚” | 適合 spike，不適合 v1 release |
| B. Refactor into new modules | 新增 `pipeline/{metrics,otel_setup,lineage_emitter,coverage}.py` 與 `templates/monitoring-config-bundle.yaml`，廢止 `monitoring-stack.yaml` | 邊界清楚、可單元測試、符合 Option B 規範 | 一次性變動範圍大、回歸風險較高 | 適合 long-term，但與 v1 時程有張力 |
| C. Hybrid（**Selected**） | 採三段式 (H1 chart-side wiring + 新增 metrics module；H2 OTel Tracer/Meter + correlation；H3 縮減 monitoring-stack 與補完 coverage CLI) | MVP 早期可見；保留可關閉旗標；與 Steering 對齊 | 計畫複雜度與多階段協調成本高 | gap-analysis §3 已論述，本 design 採此案 |

---

## Design Decisions

### Decision: 採 Option C（Hybrid）三階段交付

- **Context**：gap-analysis 顯示骨架完整、emission 全缺；同時存在 `monitoring-stack.yaml` 與上游 chart 功能重疊問題。
- **Alternatives Considered**:
  1. Option A：純擴展既有檔案（失敗於 Option B 邊界與 `job.py` 膨脹）。
  2. Option B：完整重構（失敗於 v1 時程與回歸面）。
- **Selected Approach**：H1（chart-side rules/dashboards/scrape + `pipeline/metrics.py`）→ H2（`pipeline/otel_setup.py` 真實建立 span + query-time correlation 測試）→ H3（縮減 `monitoring-stack.yaml`、`pipeline/coverage.py`、補完 `validation-report.md`）。
- **Rationale**：H1 即解鎖 R1/R2/R7/R10 的 runtime 行為，H2 解鎖 R3/R4 的 cross-signal 證據，H3 完成 Option B 邊界。每階段皆可單獨回退而不破壞 MVP。
- **Trade-offs**：需要兩個 Helm values 旗標（`monitoring.enableSelfManagedStack` / `monitoring.enableUpstreamWiring`）以隔離舊新路徑；多階段協調成本高。
- **Follow-up**：在 H3 移除 `monitoring-stack.yaml` 前，於 smoke test 加入「兩個旗標互斥」的契約檢查。

### Decision: pipeline metrics 採 `prometheus_client` 直接暴露

- **Context**：R10 要求 5 個指標族；R4 要求所有 metrics 帶 `run_id` label。
- **Alternatives Considered**:
  1. 透過 OTel SDK 的 Meter Provider → OTel Collector → `prometheusremotewrite`（多一段相依）。
  2. 在 driver process 內以 `prometheus_client.start_http_server(port)` 暴露 `/metrics`，由 Prometheus pull。
- **Selected Approach**：採 (2)。
- **Rationale**：PySpark driver 為單進程 long-running，內嵌 HTTP server 開銷低；且 Prometheus pull 模型與既有 alert rules / dashboards 直接對齊，不需經過 collector。trace 仍走 OTel SDK（見下一決定）。
- **Trade-offs**：metrics 與 trace 不共用同一 SDK；好處是 v1 更穩定，缺點是需在 design 中明示兩條 emission 路徑。
- **Follow-up**：H1 須提供 `pipeline/metrics.py` 的 5 個 collector 物件（Counter/Histogram/Gauge）並包含 `run_id` label；契約測試需斷言 `/metrics` 端點輸出包含這 5 個 family 名稱。

### Decision: trace 採 OTel SDK + Collector logging exporter（v1）

- **Context**：R4 要求 trace span 帶 `run_id / pipeline_name / k8s_namespace / status`；steering 要求最少元件。
- **Alternatives Considered**:
  1. 引入 Tempo / Jaeger（多一個元件）。
  2. 僅以 collector `debug`/`logging` exporter 留存於 stdout，作為 SLA 證據鏈。
- **Selected Approach**：(2)。
- **Rationale**：v1 主要驗證 trace span 確實被建立並抵達 collector；視覺化非首要。
- **Trade-offs**：缺乏 trace UI；以 collector log + correlation test（query 時驗證 `run_id` 一致）為證據。
- **Follow-up**：design 中標註 trace UI 為 v2 候選；在 `runbook.md` 補上「以 `kubectl logs` 觀察 collector trace 訊息」的步驟。

### Decision: lineage 採 Spark listener 為主、Python emitter 為驗證輔

- **Context**：R3 / R11.3 要求 lineage 不仰賴時間戳即可關聯；既有 `lineage.py` 已能輸出 OL JSON。
- **Alternatives Considered**:
  1. 完全依賴 Spark listener（黑盒）。
  2. 在 `pipeline/job.py` 收尾處主動 POST 一筆 `COMPLETE` 事件給 Marquez，作為「事件抵達」的驗證輔助。
- **Selected Approach**：(1) 為主送出路徑，(2) 為**測試環境** opt-in 的驗證輔助（H2）。
- **Rationale**：避免 production 重複事件造成 Marquez 端的 idempotency 問題；同時保留 smoke 測試需要的「明確 fingerprint」。
- **Trade-offs**：driver/listener 失敗時仍須重試；contract test 須區分「兩條來源」的 run_id 一致性。
- **Follow-up**：在 `lineage_emitter.py` 內以環境變數 `LINEAGE_SHADOW_EMIT=true` 作為 opt-in；H2 smoke 測試開啟此旗標。

### Decision: dashboard / alerts wiring 走顯式 ConfigMap

- **Context**：R6 / R7 要求自動載入；local-minimal 有資源預算限制。
- **Alternatives Considered**：sidecar vs 顯式 ConfigMap（見 Topic 2）。
- **Selected Approach**：顯式 ConfigMap（不啟用 sidecar）。
- **Rationale**：節省 ≈ 100–200Mi 與 2 個 sidecar container，符合 local-minimal。
- **Trade-offs**：新增 dashboards 需修改 chart values（`dashboardsConfigMaps`）；可接受。
- **Follow-up**：在 design 中說明 future 若採 sidecar，僅需切換 values（不需動 dashboards JSON）。

### Decision: coverage 從 shell 升級為 Python CLI

- **Context**：R9 要求記錄 `chart_version` 與 `last_verified_at`；shell 缺乏結構化輸出。
- **Alternatives Considered**：shell + jq vs Python CLI。
- **Selected Approach**：Python CLI（`pipeline/coverage.py`），shell 退化為 thin entry。
- **Rationale**：可於 contract test 直接 import；JSON 輸出便於日後彙總。
- **Trade-offs**：增加一個 Python module；可接受。
- **Follow-up**：CLI 需輸出 `coverage-report.json` 至 `.local-data/coverage/<ts>.json`，並支援 `--exit-code` 模式於 CI 使用。

### Decision: failure_classifier 擴展 Spark / 後端類別

- **Context**：R2.3 要求一致語意；既有僅 4 類無法分辨基礎設施問題。
- **Alternatives Considered**：以 Spark `Py4JJavaError` 字串解析 vs 以例外型別 + message 模式分類。
- **Selected Approach**：以例外型別為主，輔以特定 message 模式（regex）對 Spark / OL / OTel 後端錯誤分類。
- **Rationale**：保留既有 4 類為快速路徑，新增 `spark_task_failed` / `spark_driver_error` / `lineage_emission_failed` / `telemetry_unavailable` / `timeout`。
- **Trade-offs**：regex 維護成本；以單元測試覆蓋。
- **Follow-up**：fixture 中加入典型 Spark error message 供測試。

---

## Risks & Mitigations

- **R-1：`monitoring-stack.yaml` 與上游 chart 同時部署造成資源命名衝突** — 以 `monitoring.enableSelfManagedStack` flag 預設 false，於 H3 完全移除；H1/H2 期間於 `bootstrap-local.sh` 驗證 flag 互斥。
- **R-2：driver 內 `prometheus_client` HTTP server 與 Spark UI port 衝突** — Spark UI 預設 4040；本框架使用 9095 作為 metrics port，並於 chart values 設為可調。
- **R-3：OTel Collector remote-write 端點未啟用 receiver** — `upstream-prometheus.server.extraArgs."web.enable-remote-write-receiver": null` 列為強制 values；coverage check 主動驗證。
- **R-4：Marquez 事件延遲導致 smoke test 不穩** — `pipeline/coverage.py` 對 `/api/v1/events/lineage` 採用 polling + timeout（預設 30s），並於失敗時回報 `chart_version` + 最近一筆事件的 `eventTime`。
- **R-5：`tasks.md` 既有 `[x]` 與真實狀態不一致** — design 不重用既有 task 流水號；產生新的 tasks 時將 acceptance 強化為 “runtime observable”。
- **R-6：Python emitter 與 Spark listener 雙寫產生 lineage 重複** — emitter 預設關閉，僅在 H2 smoke test opt-in；Marquez 端以 `runId` upsert 自然去重。

---

## References

- [prometheus-community/helm-charts (prometheus-25.27.0) values.yaml](https://github.com/prometheus-community/helm-charts/blob/prometheus-25.27.0/charts/prometheus/values.yaml) — alerting rules、`extraScrapeConfigs`、`extraArgs`
- [grafana/helm-charts (grafana-8.5.1) values.yaml](https://github.com/grafana/helm-charts/blob/grafana-8.5.1/charts/grafana/values.yaml) — datasources / dashboardProviders / sidecar / dashboardsConfigMaps
- [opentelemetry-helm-charts (opentelemetry-collector-0.78.0) values.yaml](https://github.com/open-telemetry/opentelemetry-helm-charts/blob/opentelemetry-collector-0.78.0/charts/opentelemetry-collector/values.yaml) — `mode`、`config` deep-merge、`presets`
- [OTel prometheusremotewriteexporter](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/exporter/prometheusremotewriteexporter) — endpoint 與 retry 設定
- [Marquez OpenAPI](https://marquezproject.github.io/marquez/openapi.html) — namespaces / jobs / runs / events 端點
- ilum-marquez 6.7.0 chart：`charts.ilum.cloud/index.yaml`、`ilum-marquez-6.7.0.tgz`（已存於 `ai-monitor-system/deploy/helm/charts/`）
- `.kiro/steering/{product,tech,structure}.md` — 既有產品 / 技術 / 結構規範
- `.kiro/specs/pyspark-monitoring-framework/{requirements,gap-analysis}.md` — 上游契約
