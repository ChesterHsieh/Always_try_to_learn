# Implementation Plan

> **Spec**: `pyspark-monitoring-framework`
> **Companion docs**: `requirements.md`、`design.md`、`research.md`、`gap-analysis.md`
> **Deployment model**: 單堆疊直切（上游 Helm chart 為唯一部署路徑；無 H1/H2/H3 漸進、無 self-managed fallback、無 feature flag mutex）
> **Boundary 對齊**：依 `.kiro/steering/structure.md` 之 `pipeline/` / `deploy/helm/` / `deploy/scripts/` / `monitoring/` / `tests/` / `docs/` 邊界
> **Path map (post-restructure, 2026-05-05, commit `6ab73f1`)**：本 plan 中所有 `pipeline.<name>` / `pipeline/<name>.py` 引用已遷移至 `telemetry/`（telemetry 與 lineage 模組）與 `utils/`（io_adapter、scenario_schema、coverage）。完整對照表見 `design.md` 開頭的 Path Map 區塊。模組行為與契約皆未變動。

---

## 1. Foundation：依賴、image、測試骨架

- [x] 1.1 加入 Python 執行期依賴並更新 image
  - 在 `pyproject.toml` 新增 `prometheus_client>=0.20.0`、`requests>=2.31.0`、`kubernetes>=29.0.0`，並對齊既有 lock 機制
  - 更新 `Dockerfile` 之 pip install 指令包含上述三個套件；明確**不**安裝 `helm` CLI
  - 完成觀察條件:`docker build` 成功,且 `python -c "import prometheus_client, requests, kubernetes"` 在 image 內可正常 import
  - _Requirements: 5.4, 9.3, 12.1_

- [x] 1.2 建立測試 fixtures 與共用模擬層
  - 在 `tests/conftest.py` 新增 `InMemorySpanExporter` fixture（OTel）、mock Marquez HTTP server fixture、mock Kubernetes Helm release Secret fixture
  - 提供 `prometheus_registry` fixture,於每個測試結束時重置 `prometheus_client.REGISTRY` 以避免跨測試污染
  - 完成觀察條件:`pytest --collect-only` 列出新 fixtures,且 `pytest tests/conftest.py` smoke import 通過
  - _Requirements: 4.4, 12.1_

- [x] 1.3 確認 Helm umbrella alias 與 `Chart.lock` 一致性
  - 對照 `Chart.yaml` 的四個 `dependencies`（`upstream-prometheus`、`upstream-grafana`、`upstream-otel-collector`、`upstream-marquez`）與 `Chart.lock` 釘版號是否與 `docs/chart-version-matrix.md` 相符
  - 若漂移則執行 `helm dependency update`，並在 `Chart.lock` 提交對齊後的 SHA
  - 完成觀察條件：`helm dependency list deploy/helm/` 顯示四個 alias 為 `ok` 狀態，版本與 design.md `Technology Stack` 一致
  - _Requirements: 5.1, 5.3_

## 2. Failure classifier 擴展

- [x] 2.1 擴增 failure 類別至 9 類
  - 於 `pipeline/failure_classifier.py` 之 `KNOWN_CATEGORIES` 新增 `spark_task_failed`、`spark_driver_error`、`lineage_emission_failed`、`telemetry_unavailable`、`timeout`
  - `classify_failure` 透過例外型別（含 `Py4JJavaError`、`requests.exceptions.ConnectionError`、`socket.timeout`）為主、訊息 regex 為輔；regex 集中於模組常數區
  - 永遠回傳非空字串；未知例外回 `runtime_error`
  - 完成觀察條件：對 9 種預期例外的 fixture 呼叫 `classify_failure` 各回傳對應 category，且 `KNOWN_CATEGORIES` 為 frozenset 不可變
  - _Requirements: 2.1, 2.3, 3.3_
  - _Boundary: pipeline.failure_classifier_

- [x] 2.2 (P) Contract test：KNOWN_CATEGORIES 與分類決定性
  - 新增 / 強化 `tests/contract/` 內 failure_classifier 測試：對每個 category 提供至少一筆代表性例外 fixture，斷言 `classify_failure(exception)` 返回值落在 `KNOWN_CATEGORIES`
  - 同一例外型別重複呼叫返回相同 category（決定性檢查）
  - 完成觀察條件：`pytest -k failure_classifier` 全綠，且 `KNOWN_CATEGORIES` 列為對外穩定 list
  - _Requirements: 2.3, 12.1_
  - _Boundary: pipeline.failure_classifier_
  - _Depends: 2.1_

## 3. Driver-side metrics 模組與 `/metrics` 端點

- [x] 3.1 建立 5 個指標族與 exemplar-only 高基數承載
  - 於 `pipeline/metrics.py` 建立 `pipeline_run_total`、`pipeline_run_duration_seconds`、`pipeline_records_processed_total`、`pipeline_failures_total`、`pipeline_telemetry_freshness_seconds` 五個 collector
  - Label set 限定為 design.md 指定之低基數欄位；於模組初始化時驗證若任何 collector 包含 `run_id` / `failure_message` 則 raise（contract 守門）
  - 對 Counter / Histogram 之 `inc(exemplar=...)` / `observe(amount, exemplar=...)` 包裝 helper，將 `run_id`、`trace_id`、（失敗時）截斷後的 `failure_message` 寫入 exemplar
  - 完成觀察條件：以 `prometheus_client.generate_latest(REGISTRY)`（OpenMetrics 格式）輸出可看到 5 個 family、且 sample 行後跟 `# {run_id="..."} ...` exemplar 行
  - _Requirements: 1.1, 1.2, 2.1, 4.1, 4.3, 10.1, 10.3_
  - _Boundary: pipeline.metrics_

- [x] 3.2 (P) 啟動 OpenMetrics HTTP 端點於 9095
  - 以 `prometheus_client.start_http_server` 啟動 HTTP server，content type 設為 `application/openmetrics-text; version=1.0.0; charset=utf-8`，否則 exemplars 不會被 Prometheus 寫入
  - port 由 env `METRICS_PORT` 覆寫（預設 9095）；確認與 Spark UI（4040）不衝突；server 啟動失敗則 driver 以 exit 78 結束（fail-fast,不允許 silent 觀測缺失）
  - 完成觀察條件：driver 啟動後 `curl http://localhost:9095/metrics` 回應 200 並含 `# TYPE pipeline_run_total counter` 與至少一筆 exemplar 行
  - _Requirements: 1.4, 7.4, 11.1_
  - _Boundary: pipeline.metrics_
  - _Depends: 3.1_

- [x] 3.3 Contract test：metric label set 與 exemplar 守門
  - 新增 `tests/contract/test_metrics_contract.py`，以 OpenMetrics 格式輸出斷言：(a) 5 個 family 名稱完整；(b) label set 嚴格符合 design.md Data Contracts 表（`run_id` / `failure_message` 不在 label）；(c) Counter / Histogram 至少一筆 sample 含 `run_id` exemplar；(d) `pipeline_telemetry_freshness_seconds` 為 Gauge 且不附 exemplar
  - 任何 family 缺漏或 label 違反皆使測試失敗（R10.2 contract 違規）
  - 完成觀察條件：`pytest tests/contract/test_metrics_contract.py -v` 全綠，且故意污染 label 之 fixture 會被偵測
  - _Requirements: 4.4, 10.1, 10.2_
  - _Depends: 3.1_

## 4. OpenTelemetry SDK 初始化與 trace 串接

- [x] 4.1 (P) 建立 Tracer/Resource 設定與 provider 單例
  - 於 `pipeline/otel_setup.py` 實作 `configure_tracer(otlp_endpoint=...)`：以 `TracerProvider` + `OTLPSpanExporter`（gRPC）；endpoint 由 env `OTEL_EXPORTER_OTLP_ENDPOINT` 取得
  - Resource attributes 至少含 `service.name=pipeline-job`、`k8s.namespace=ai-monitor-system`
  - Tracer 為 process-global；重複呼叫 `configure_tracer` 不會重新建立 provider；啟動失敗則 driver 以 exit 78 結束（對齊 metrics fail-fast 規則）
  - 完成觀察條件：unit test 驗證 `configure_tracer()` 後 `trace.get_tracer_provider()` 為已設定的 provider；resource attributes 完整
  - _Requirements: 4.1, 4.2_
  - _Boundary: pipeline.otel_setup_

- [x] 4.2 實作 `start_run_span` context manager 與終態屬性
  - 於 `pipeline/tracing.py`（既有檔案）對接 `otel_setup`，提供 `start_run_span(run_id, pipeline_name, k8s_namespace)`；span attributes 含 `pipeline.run_id`、`pipeline.name`、`k8s.namespace`
  - 終態 span 額外設定 `status` attribute（succeeded / failed）；失敗時 `record_exception` 並設定 span status code
  - 完成觀察條件：以 `InMemorySpanExporter` 取得 span 後可看到上述四個 attributes 完整且 status 正確
  - _Requirements: 4.1, 4.2_
  - _Boundary: pipeline.tracing_
  - _Depends: 4.1_

- [x] 4.3 Integration test：`InMemorySpanExporter` 驗證 span 屬性
  - 強化 `tests/integration/test_trace_attributes.py`，以 `InMemorySpanExporter` 取代純 dict 斷言
  - 對成功與失敗 run 分別斷言終態 span attributes 含 `status`、`pipeline.run_id`、`pipeline.name`、`k8s.namespace`
  - 完成觀察條件：`pytest tests/integration/test_trace_attributes.py -v` 全綠
  - _Requirements: 4.2, 12.1_
  - _Depends: 4.2_

## 5. Lineage shadow emitter（opt-in）

- [x] 5.1 (P) 實作 `maybe_shadow_emit` 與 retry 政策
  - 由 env `LINEAGE_SHADOW_EMIT=true` opt-in；以 `pipeline.lineage.build_openlineage_event` 取得 payload
  - `POST {marquez_url}/api/v1/lineage`；指數退避 retry（最多 3 次，base=1s）
  - 失敗時透過 `failure_classifier.classify_failure` 對應為 `lineage_emission_failed` 並結構化 log（含 `run_id`），不 raise 回呼叫端
  - 完成觀察條件：mock `requests.post` 模擬 503 三次後 200，回傳 `True`；模擬連續 503 五次回傳 `False` 並 log 對應 category
  - _Requirements: 3.1, 3.4, 11.3_
  - _Boundary: pipeline.lineage_emitter_
  - _Depends: 2.1_

## 6. 主流串接：`pipeline.job` lifecycle 組裝

- [x] 6.1 將 metrics recorder 串接至 `pipeline.job` lifecycle
  - 在 `run_pipeline()` 入口建立 recorder、呼叫 `start_endpoint(port=...)`；在 run started / succeeded / failed 三個轉折呼叫對應 `record_*`
  - 失敗路徑於 `except` 區塊呼叫 `failure_classifier.classify_failure` 取得 category 後傳入 `record_run_failed`；任何 metrics 寫入失敗以 `try/except` 包覆並結構化 log（含 `run_id`），不中斷 pipeline 主流
  - `update_freshness` 於 run 終態呼叫，將最近一次 telemetry 事件距今秒數寫入 Gauge
  - 完成觀察條件：本機觸發一次 success run 後，`/metrics` 端點 `pipeline_run_total{status="succeeded"}` 增量為 1 且帶 `run_id` exemplar
  - _Requirements: 1.1, 1.2, 2.1, 4.3, 7.4, 11.1_
  - _Depends: 3.2, 2.1_

- [x] 6.2 將 tracer 接入 `pipeline.job` 主流並設定 shutdown
  - 在 `run_pipeline()` 入口呼叫 `configure_tracer()` + `start_run_span(...)`，以 context manager 包覆主處理段
  - 將 `run_id` 與 `trace_id` 傳入 metrics exemplar helper（由 task 3.1 提供）
  - process exit 前呼叫 `tracer_provider.force_flush(timeout_millis=2000)`，確保 trace 不丟失
  - 完成觀察條件：執行一次完整 run 後 OTel Collector logging exporter stdout 顯示對應 span，且 attributes 含 `pipeline.run_id`
  - _Requirements: 4.1, 4.2_
  - _Depends: 6.1, 4.2_

- [x] 6.3 將 shadow emitter 串接至 `pipeline.job` 終態
  - 於 `run_pipeline()` 終態呼叫 `lineage_emitter.maybe_shadow_emit(run_id=..., job_name=..., namespace=..., source_dataset=..., target_dataset=...)`
  - 在 `tests/integration/conftest.py` 新增 fixture 設定 `LINEAGE_SHADOW_EMIT=true` + mock Marquez svc
  - 完成觀察條件：以 mock Marquez 接收 POST 後可見對應 OL `RunEvent` JSON，且 `runId` 與 driver run_id 一致
  - _Requirements: 3.1, 11.3_
  - _Depends: 6.1, 5.1_

## 7. Chart values 整合（rules / scrape / dashboards / OTel exporter）

> **共享檔案注意**：7.1–7.5 皆修改 `deploy/helm/values.yaml` 與 `values.local-minimal.yaml`，視為單一 boundary；序列執行避免 merge 衝突。單堆疊部署，無 feature flag / mutex。

- [x] 7.1 設定 Prometheus values：scrape、rules、remote-write receiver
  - 在 `upstream-prometheus.extraScrapeConfigs` 加入 `kubernetes_sd_configs`，以 label `app.kubernetes.io/name=pyspark-pipeline` + port name `metrics`（9095）為抓取目標；scrape interval 15s
  - 加入 `upstream-prometheus.serverFiles."alerting_rules.yml"`，由 `monitoring-config-bundle` ConfigMap 灌入（task 7.5 串接）
  - 加入 `upstream-prometheus.server.extraArgs."web.enable-remote-write-receiver": null` 啟用 remote-write receiver；同時啟用 `--enable-feature=exemplar-storage`
  - 完成觀察條件：`helm template deploy/helm/ | yq '.spec.template.spec.containers[0].args'` 包含上述 flags
  - _Requirements: 5.1, 10.3_

- [x] 7.2 設定 Grafana values：datasources、dashboards、disable sidecar
  - `upstream-grafana.datasources` inline 設定 Prometheus 為預設來源；URL 使用 chart 預設 svc DNS（`http://ai-monitor-system-upstream-prometheus-server`）
  - 設定 `upstream-grafana.dashboardsConfigMaps.default: "{{ .Release.Name }}-pipeline-dashboards"`；`sidecar.dashboards.enabled: false`、`sidecar.datasources.enabled: false` 以降低 local-minimal memory budget
  - 完成觀察條件：`helm template deploy/helm/ | grep -A2 dashboardsConfigMaps` 顯示對應 ConfigMap 名稱；`kubectl get cm -l grafana_dashboard=1` 在部署後可見
  - _Requirements: 5.1, 6.3, 10.3_

- [x] 7.3 設定 OpenTelemetry Collector values：exporters 與 pipeline
  - `upstream-otel-collector.mode: deployment`
  - `config.exporters.prometheusremotewrite.endpoint: http://ai-monitor-system-upstream-prometheus-server.ai-monitor-system.svc:80/api/v1/write`
  - `config.exporters.logging.verbosity: detailed`；`config.service.pipelines.metrics.exporters: [prometheusremotewrite]`、`config.service.pipelines.traces.exporters: [logging]`
  - 完成觀察條件：`helm template ... | yq '.data["config.yaml"]'` 顯示上述 exporters / pipelines；Collector 啟動後 `/metrics` 端點顯示 `otelcol_exporter_sent_metric_points` 增長
  - _Requirements: 4.2, 5.1, 10.3_

- [x] 7.4 確認 Marquez alias、port 與 namespace
  - 確認 `upstream-marquez` alias 對應之 svc 為 `ai-monitor-system-upstream-marquez:9555`（API port，non-admin）；`coverage` CLI 將透過該 svc 呼叫 `/api/v1/namespaces`、`/api/v1/events/lineage`
  - 在 `values.yaml` 將 marquez namespace 對齊主 release namespace（避免 cross-namespace 解析）
  - 完成觀察條件：`kubectl exec` 進入任一 pod 內執行 `curl http://ai-monitor-system-upstream-marquez:9555/api/v1/namespaces` 回應 200 JSON
  - _Requirements: 3.1, 5.1_

- [x] 7.5 建立 monitoring-config-bundle ConfigMap 模板 + `metricsPort` 對齊
  - 新增 `deploy/helm/templates/monitoring-config-bundle.yaml`，包含：(a) `{{ .Release.Name }}-pipeline-dashboards`（label `grafana_dashboard=1`）；(b) `{{ .Release.Name }}-pipeline-alert-rules`
  - 內容以 `tpl (.Files.Glob "../../monitoring/dashboards/*.json").AsConfig`、`tpl (.Files.Glob "../../monitoring/alerts/*.yaml").AsConfig` 灌入；不 hardcode 檔案清單
  - 將 `upstream-prometheus.serverFiles."alerting_rules.yml"` 內容由此 ConfigMap 取得（透過 `tpl` lookup 或 helm `lookup`）
  - `values.yaml` 與 `values.local-minimal.yaml` 對齊 `pyspark.metricsPort: 9095`
  - 完成觀察條件：`helm template deploy/helm/` 輸出兩個 ConfigMap，且 `data` 區塊含對應 dashboard JSON 與 alert YAML 鍵；`helm template -f values.local-minimal.yaml` 可正常渲染
  - _Requirements: 5.2, 5.4, 6.1, 7.1_

## 8. Helm 模板、bootstrap 與 teardown

- [x] 8.1 更新 `pipeline-job.yaml` 暴露 metrics 與 RBAC
  - 加入 `containerPort: 9095`（name=`metrics`）、annotations `prometheus.io/scrape: "true"`、`prometheus.io/port: "9095"`、`pipeline-monitoring/run-id: "{{ "" }}"`（由 entrypoint 動態注入）
  - 補上 ServiceAccount + namespaced Role（`secrets:get,list` 限定 `selector "owner=helm,name=upstream-*"`）+ RoleBinding，使 `pipeline.coverage` 可讀 Helm release Secret
  - 完成觀察條件：`kubectl describe pod -l app.kubernetes.io/name=pyspark-pipeline` 顯示 9095 port 與 service account；`kubectl auth can-i get secrets --as=system:serviceaccount:ai-monitor-system:pyspark-pipeline -n ai-monitor-system` 回 `yes`
  - _Requirements: 1.1, 5.4, 9.3_

- [x] 8.2 強化 `nuke-local.sh` 做為 dev teardown 唯一路徑
  - 既有 `deploy/scripts/nuke-local.sh` 需驗證三個行為：(a) `helm uninstall` 失敗不 propagate；(b) namespace stuck 時以 `/finalize` 強制釋放；(c) 逐一刪除 `claimRef.namespace==ai-monitor-system` 之 PV
  - 加上 `set -euo pipefail` 下仍容錯的 `|| true` 分支；對 `jq` 未安裝時印出明確安裝指引
  - 完成觀察條件：在 dirty 狀態（namespace 有卡死 finalizer + released PV）執行後，`kubectl get ns ai-monitor-system` 回 `NotFound` 且 `kubectl get pv` 不再看到對應 PV
  - _Requirements: 5.2, 8.1_
  - _Boundary: deploy/scripts_

- [x] 8.3 更新 `bootstrap-local.sh` 計時與 nuke 串接
  - 支援 `NUKE_BEFORE_BOOTSTRAP=true`：在 helm 部署前呼叫 `nuke-local.sh` 清場
  - 以 `date +%s` 計時整體 bootstrap 流程；輸出 `BOOTSTRAP_DURATION_SECONDS=...` 至 stdout（給 R8.4 smoke test 比對）
  - 移除任何殘留 `enableSelfManagedStack` / `enableUpstreamWiring` flag 引用；只保留上游 chart 單一路徑
  - 完成觀察條件：本機執行 `NUKE_BEFORE_BOOTSTRAP=true bash deploy/scripts/bootstrap-local.sh` 在 minimal profile 下 ≤10 分鐘完成，並印出 `BOOTSTRAP_DURATION_SECONDS=...`
  - _Requirements: 5.2, 8.1, 8.4_
  - _Depends: 8.2_

- [x] 8.4 更新 `run-pipeline.sh` 計時與 exit code
  - 包覆 `kubectl create job ...` 與 `kubectl wait --for=condition=complete --timeout=5m`
  - 輸出 `RUN_DURATION_SECONDS=...`；失敗時 propagate 非 0 exit code（區分超時 / 失敗 / 後端不可達）
  - 完成觀察條件：success run 與 fail run 分別印出時程與正確退出碼，且本機叢集條件下成功 run ≤5 分鐘
  - _Requirements: 1.4, 8.4_

## 9. Alert rules 與 Grafana dashboards

- [x] 9.1 (P) 重寫 `pipeline-failure-rules.yaml` annotation 結構
  - `expr` 改用 `increase(pipeline_failures_total[1m]) > 0`；`for: 1m`（避免 pod 重啟造成重複告警）
  - annotations 必含 `summary`、`pipeline_name`、`failure_category`、`dashboard_link`（含 `var-pipeline_name` + `from`/`to` 時間參數）、`runbook_link`；**不**直接引用 `$labels.run_id`
  - 完成觀察條件：`promtool check rules monitoring/alerts/pipeline-failure-rules.yaml` 通過；`yq '.groups[0].rules[0].annotations'` 不含 `run_id` 鍵
  - _Requirements: 2.2, 7.1, 7.3, 11.2, 12.3_
  - _Boundary: monitoring/alerts_

- [x] 9.2 (P) 更新 `stack-health-rules.yaml` 增補 `PipelineMetricsMissing`
  - 加入 `up{job="pyspark-pipeline"} == 0 for: 2m` 之 warning rule
  - 加入 `pipeline_telemetry_freshness_seconds > 300` warning，annotation 含 `pipeline_name` 與 dashboard 連結
  - 完成觀察條件：`promtool check rules monitoring/alerts/stack-health-rules.yaml` 通過；測試 fixture 模擬 freshness=600s 觸發告警
  - _Requirements: 7.2, 7.4, 11.1_
  - _Boundary: monitoring/alerts_

- [x] 9.3 (P) 更新 `pipeline-health.json` dashboard
  - 新增 `Recent Failed Runs` 表格 panel（資料來源為 Marquez API + Histogram exemplar 抽樣），欄位含 `run_id`、`pipeline_name`、`failure_category`、`failure_message`、`start_time`
  - 新增 `pipeline_telemetry_freshness_seconds` 趨勢圖 panel
  - 保留既有 stat panels；新增 `var-pipeline_name` 變數與時間範圍與 alert dashboard_link 對齊
  - 完成觀察條件：載入 dashboard 後可在 Grafana 看到對應 panels；模擬一筆失敗 run 後表格內出現 `run_id`
  - _Requirements: 1.3, 6.1, 6.2, 6.3, 10.3, 12.3_
  - _Boundary: monitoring/dashboards_

- [x] 9.4 (P) 更新 `lineage-view.json` dashboard
  - 以 `run_id` 為 panel 變數入口，顯示 source / target dataset 圖；提供「Open in Marquez」外連按鈕指向 `/lineage/jobs/...`
  - 失敗 run 之 lineage panel 標示對應 `failure_category`
  - 完成觀察條件：在 dashboard 中輸入任一 `run_id` 可看到對應 lineage 圖與 Marquez 連結
  - _Requirements: 3.2, 3.3, 6.4_
  - _Boundary: monitoring/dashboards_

## 10. Coverage CLI 與 chart_version 報告

- [x] 10.1 實作 coverage 健康檢查序列（fail-fast）
  - 於 `pipeline/coverage.py` 實作 `run_coverage(namespace, marquez_url, prometheus_url, grafana_url)`，依序檢查：Prometheus reachable → `/api/v1/rules` 含 `pipeline-failure` 群組 → Grafana datasource health → Marquez `/api/v1/namespaces` → Marquez 最近 N 筆 lineage
  - 每次 HTTP 呼叫 timeout=5s；總體 polling 上限 30s；其中 `STALE_LINEAGE` 為 warn（exit 1，避免 cold-start 誤報），其餘 backend 不可達為 critical（exit 2）
  - 完成觀察條件：以 mock 後端執行 `python -m pipeline.coverage --output /tmp/r.json` 印出 summary 並寫 JSON；exit code 與檢查結果一致
  - _Requirements: 7.4, 9.1, 9.2, 11.4_
  - _Boundary: pipeline.coverage_

- [x] 10.2 透過 Kubernetes API 解析 Helm release `chart_version`
  - 使用 `kubernetes.client.CoreV1Api.list_namespaced_secret(label_selector="owner=helm,name=<release>")` 取最新 revision；解碼 base64+gzip JSON 後讀 `chart.metadata.version`
  - 對 4 個 release（`upstream-prometheus`、`upstream-grafana`、`upstream-otel-collector`、`upstream-marquez`）逐一解析；`in-cluster` 不可用時 fallback `~/.kube/config`；皆失敗時退化為解析 `docs/chart-version-matrix.md` 並標 `chart_version_lookup` 為 `warn`
  - 完成觀察條件：以 mock kubernetes 客戶端 fixture 餵入 4 個 fake secret，`coverage.report.components` 取得對應 semver 字串
  - _Requirements: 5.3, 9.3_
  - _Boundary: pipeline.coverage_

- [x] 10.3 實作 CoverageReport JSON writer 與 exit code policy
  - 將 `validation_checks`（list of `CheckResult`）、`components`（dict）、`profile_name`、`profile_version`、`last_verified_at`（UTC ISO 8601）寫入 `--output PATH`（預設 `.local-data/coverage/<ts>.json`）
  - 統整 exit code：任一 critical → 2；無 critical 但有 warn → 1；皆 pass → 0
  - 完成觀察條件：`cat .local-data/coverage/<ts>.json | jq '.last_verified_at'` 回傳 ISO 8601 字串；模擬 mixed result 時 exit code 對應規則正確
  - _Requirements: 9.4, 12.4_
  - _Depends: 10.1, 10.2_

- [x] 10.4 將 `check-monitoring-coverage.sh` 退化為 thin wrapper
  - shell 僅組裝 svc URL（in-cluster service DNS）並 `exec python -m pipeline.coverage --output ...`
  - 移除 shell 內任何模仿型 health-check 邏輯
  - 完成觀察條件：`bash deploy/scripts/check-monitoring-coverage.sh` 行為與直接執行 CLI 等價
  - _Requirements: 9.1, 9.2_
  - _Depends: 10.3_

- [x] 10.5 Contract test：CoverageReport 結構與 chart_version 內容
  - 新增 / 強化 `tests/contract/test_monitoring_coverage_contract.py`，import `pipeline.coverage.run_coverage` with mocks；斷言 `CoverageReport.components` 涵蓋 4 個 component 且每個 `chart_version` 為非空 semver 字串
  - 斷言 `last_verified_at` 為 ISO 8601；`validation_checks` 至少含 5 個 check（Prometheus、rules、Grafana、Marquez、recent lineage）
  - 完成觀察條件：`pytest tests/contract/test_monitoring_coverage_contract.py -v` 全綠
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 12.1_
  - _Depends: 10.3_

## 11. 跨訊號 correlation 驗證

- [x] 11.1 新增 `tests/integration/test_query_time_correlation.py`
  - 以 `prometheus_client.REGISTRY` + `InMemorySpanExporter` + mock Marquez 三方共構，呼叫 `pipeline.job.run_pipeline`
  - 斷言：(a) Prometheus exemplar 中含特定 `run_id`；(b) 至少一筆 span attribute `pipeline.run_id` 等於該值；(c) Marquez 端收到 `RunEvent.runId` 等於該值
  - 完成觀察條件：`pytest tests/integration/test_query_time_correlation.py -v` 全綠；故意污染任一條訊號之 run_id 即測試失敗
  - _Requirements: 4.1, 4.4, 11.3, 12.1_
  - _Depends: 6.2, 6.3_

- [x] 11.2 (P) 強化 `test_failure_alerts.py` 守門 alert annotation 不依賴 `$labels.run_id`
  - 解析 `pipeline-failure-rules.yaml`，斷言 `annotations.dashboard_link`、`annotations.failure_category`、`annotations.pipeline_name` 模板字串完整
  - 額外斷言任何 annotation 不含 `$labels.run_id`（避免 cardinality 回退）
  - 完成觀察條件：`pytest tests/integration/test_failure_alerts.py -v` 全綠；故意改回 `$labels.run_id` 即測試失敗
  - _Requirements: 7.3, 12.3_
  - _Boundary: tests/integration_
  - _Depends: 9.1_

## 12. Resilience 與端到端 smoke

- [x] 12.1 新增 `tests/smoke/test_resilience_min.py`
  - 場景 A：`kubectl delete pod -l app.kubernetes.io/name=pyspark-pipeline`，斷言 `pipeline-failure` alert `for: 1m` 不重複觸發；於 Job 重新排程後可見 metric series 連續性
  - 場景 B：`kubectl scale deploy ai-monitor-system-upstream-opentelemetry-collector --replicas=0` 持續 90s 後恢復；斷言 freshness alert 觸發後在恢復後 3 個 evaluation cycle 內 resolved
  - 完成觀察條件：本機叢集執行該檔可在 ≤5 分鐘內完成兩個場景驗證並全綠
  - _Requirements: 2.4, 11.2, 11.4_

- [x] 12.2 (P) 強化 `tests/smoke/test_end_to_end_local_cluster.py`
  - 以 `time.monotonic()` 包覆 `bash deploy/scripts/bootstrap-local.sh` 與 `bash deploy/scripts/run-pipeline.sh`，斷言：bootstrap ≤600s、success run ≤300s
  - 取得 run_id 後呼叫 `coverage.run_coverage` 並斷言 `chart_version` 內容完整
  - 完成觀察條件：smoke 全綠並印出兩個 SLA 對照值（實測 vs 上限）
  - _Requirements: 1.4, 5.3, 8.4, 12.4_
  - _Boundary: tests/smoke_
  - _Depends: 8.3, 8.4, 10.3_

- [x] 12.3 (P) 改寫 `tests/smoke/test_us3_monitoring_coverage.py` 直接呼叫 `run_coverage`
  - 不再經 shell；以 in-cluster URLs 直接 import 並呼叫 `pipeline.coverage.run_coverage`
  - 斷言 `status="pass"` 之 check 數量 ≥5、`components` 含 4 個 chart_version
  - 完成觀察條件：smoke 全綠且輸出 JSON 與 10.x 之 thin wrapper 等價
  - _Requirements: 9.1, 9.2_
  - _Boundary: tests/smoke_
  - _Depends: 10.4_

- [x] 12.4 (P) 新增 `tests/smoke/test_nuke_and_rebuild.py`
  - 執行 `bash deploy/scripts/nuke-local.sh` 後斷言：`kubectl get ns ai-monitor-system` 返回 `NotFound`、沒有屬於該 ns 的 PV、`helm list -n ai-monitor-system` 為空
  - 接著 `NUKE_BEFORE_BOOTSTRAP=true bash deploy/scripts/bootstrap-local.sh` 重建；斷言四個 upstream release 全部 `deployed`
  - 完成觀察條件：smoke 全綠，且總時程 ≤15 分鐘（涵蓋完整 teardown + bootstrap）
  - _Requirements: 5.2, 8.1, 8.4_
  - _Boundary: tests/smoke_
  - _Depends: 8.2, 8.3_

## 13. 文件交付（onboarding / runbook / validation report）

> 本框架以 onboarding 為核心使用者價值；下列文件屬交付契約而非次要工作。

- [x] 13.1 (P) 更新 `docs/onboarding-monitoring.md`
  - 補上單堆疊部署模型說明（上游 chart 為唯一路徑、無 feature flag mutex）、`/metrics` 端點 9095、exemplar 模型與 `run_id` 由 dashboard drilldown 解析的查詢路徑
  - 補上 `NUKE_BEFORE_BOOTSTRAP=true` 的 dev 清場流程與警告（只限本機叢集）
  - 標明 owner、預期 input（local k8s 叢集、CPU/RAM 預算）、預期 output（bootstrap 完成的 4 個 release + pipeline Job）
  - 完成觀察條件：新成員依該文件可在 ≤10 分鐘 bootstrap、≤5 分鐘觸發成功 run（本機 minimal）
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 12.2_
  - _Boundary: docs_

- [x] 13.2 (P) 更新 `docs/runbook.md`
  - 補上 alert → Grafana panel → Marquez lineage 三跳追蹤路徑示意
  - 補上 coverage CLI 用法、退出碼語意、常見錯誤對應 `failure_category`
  - 補上 trace 觀察方式（OTel Collector logging exporter stdout）
  - 補上 `nuke-local.sh` 使用時機與風險（namespace stuck / 測試資料污染）
  - 完成觀察條件：runbook 章節覆蓋 design.md 9 個 failure category、且每個 category 對應一段排錯步驟
  - _Requirements: 8.3, 12.2, 12.3_
  - _Boundary: docs_

- [x] 13.3 (P) 改寫 `docs/validation-report.md` 模板
  - 以 placeholder 形式列出本 release 的 contract / integration / smoke 結果與 `coverage-report.json` 摘要欄位
  - 標明 owner、執行步驟、最近一次驗證時間欄位（對應 `last_verified_at`）
  - 完成觀察條件：執行 task 14 之後此文件可被填入實測值
  - _Requirements: 9.4, 12.2, 12.4_
  - _Boundary: docs_

## 14. Release validation gate

- [x] 14.1 全鏈路 release validation 執行
  - 依序執行：`pytest tests/contract`、`pytest tests/integration`、`bash deploy/scripts/run-smoke-test.sh`、`python -m pipeline.coverage --output .local-data/coverage/release.json`
  - 將四步驟結果與 `coverage-report.json` 摘要寫入 `docs/validation-report.md`（手動或腳本填入）
  - 任一步驟非 0 退出碼則 release gate fail
  - 完成觀察條件：本地完整流程綠燈時 `validation-report.md` 含實測時間與 `last_verified_at`；`coverage-report.json` 之 `chart_version` 全部為 semver 字串
  - _Requirements: 9.4, 12.1, 12.3, 12.4_
  - _Depends: 2.2, 3.3, 4.3, 9.1, 9.2, 10.5, 11.1, 11.2, 12.1, 12.2, 12.3, 12.4_

---

## Requirements Coverage Map

| Requirement | Tasks |
|-------------|-------|
| 1.1 | 3.1, 6.1, 8.1 |
| 1.2 | 3.1, 6.1 |
| 1.3 | 9.3 |
| 1.4 | 3.2, 7.1, 8.4, 12.2 |
| 2.1 | 2.1, 3.1, 6.1 |
| 2.2 | 9.1 |
| 2.3 | 2.1, 2.2 |
| 2.4 | 12.1 |
| 3.1 | 5.1, 6.3, 7.4 |
| 3.2 | 9.4 |
| 3.3 | 2.1, 9.4 |
| 3.4 | 5.1 |
| 4.1 | 3.1, 4.1, 4.2, 6.2, 11.1 |
| 4.2 | 4.1, 4.2, 4.3, 6.2, 7.3 |
| 4.3 | 3.1, 6.1 |
| 4.4 | 3.3, 11.1 |
| 5.1 | 1.3, 7.1, 7.2, 7.3, 7.4 |
| 5.2 | 7.5, 8.2, 8.3, 12.4 |
| 5.3 | 1.3, 10.2, 12.2 |
| 5.4 | 1.1, 7.5, 8.1 |
| 6.1 | 7.5, 9.3 |
| 6.2 | 9.3 |
| 6.3 | 7.2, 9.3 |
| 6.4 | 9.4 |
| 7.1 | 7.5, 9.1 |
| 7.2 | 9.2 |
| 7.3 | 9.1, 11.2 |
| 7.4 | 3.2, 6.1, 9.2, 10.1 |
| 8.1 | 8.2, 8.3, 12.4, 13.1 |
| 8.2 | 13.1 |
| 8.3 | 13.1, 13.2 |
| 8.4 | 8.3, 8.4, 12.2, 12.4, 13.1 |
| 9.1 | 10.1, 10.4, 10.5, 12.3 |
| 9.2 | 10.1, 10.4, 10.5, 12.3 |
| 9.3 | 1.1, 8.1, 10.2, 10.5 |
| 9.4 | 10.3, 10.5, 13.3, 14.1 |
| 10.1 | 3.1, 3.3 |
| 10.2 | 3.3 |
| 10.3 | 3.1, 7.1, 7.2, 7.3, 9.3 |
| 11.1 | 3.2, 6.1, 9.2 |
| 11.2 | 9.1, 12.1 |
| 11.3 | 5.1, 6.3, 11.1 |
| 11.4 | 10.1, 12.1 |
| 12.1 | 1.1, 1.2, 2.2, 4.3, 10.5, 11.1, 14.1 |
| 12.2 | 13.1, 13.2, 13.3 |
| 12.3 | 9.1, 9.3, 11.2, 13.2, 14.1 |
| 12.4 | 10.3, 12.2, 13.3, 14.1 |

## Parallelization Notes

- `(P)` markers 反映「同一 parent 下不同 boundary 之 sub-task 可並行」；跨 major 之並行（例如 task 5 與 task 9）由 ordering 與 boundary 隔離自然成立。
- `values.yaml` 屬 shared file → task 7.1–7.5 不標 `(P)`，需序列執行避免合併衝突。
- `pipeline.job` 為 integration-critical orchestrator → task 6.1–6.3 不標 `(P)`，序列串接 metrics / tracer / shadow emitter。
- task 14.1 為 release gate，依設計依賴近全部 validation 任務，明確列於 `_Depends:_`。
