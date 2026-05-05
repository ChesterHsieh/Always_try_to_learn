# Runbook: PySpark Monitoring Framework

**Owner**: Platform team  
**Scope**: Local Kubernetes deployment of PySpark monitoring framework

---

## Operational Commands

```bash
# Bootstrap stack
bash deploy/scripts/bootstrap-local.sh

# Run pipeline
bash deploy/scripts/run-pipeline.sh

# Verify coverage
bash deploy/scripts/check-monitoring-coverage.sh
# OR: python -m pipeline.coverage --output .local-data/coverage/report.json

# Run smoke tests
bash deploy/scripts/run-smoke-test.sh

# Teardown (dev only)
bash deploy/scripts/nuke-local.sh
NUKE_BEFORE_BOOTSTRAP=true bash deploy/scripts/bootstrap-local.sh
```

---

## Coverage CLI

```bash
python -m pipeline.coverage \
  --namespace ai-monitor-system \
  --marquez-url http://ai-monitor-system-upstream-marquez:9555 \
  --prometheus-url http://ai-monitor-system-upstream-prometheus-server:80 \
  --grafana-url http://ai-monitor-system-upstream-grafana:80 \
  --output .local-data/coverage/$(date -u +%Y%m%dT%H%M%SZ).json
```

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All checks pass |
| `1` | Warning (stale lineage, helm secret fallback, datasource latency) |
| `2` | Critical (Prometheus unreachable, rules not loaded, Grafana down, Marquez down) |

---

## Failure Categories

All 9 categories are defined in `pipeline.failure_classifier.KNOWN_CATEGORIES`:

| Category | Trigger | Troubleshooting |
|----------|---------|-----------------|
| `input_not_found` | `FileNotFoundError` | Check `INPUT_PATH` env var and hostPath volume mount |
| `invalid_path` | `IsADirectoryError` | Verify `INPUT_PATH` points to file, not directory |
| `permission_denied` | `PermissionError` | Check pod ServiceAccount + volume mount permissions |
| `spark_task_failed` | `Py4JJavaError` (task failure) | Check Spark logs for task exception; reduce data skew |
| `spark_driver_error` | `Py4JJavaError` (driver/SparkException) **or** `pyspark.errors.AnalysisException` (plan-analyzer failure such as schema drift) | Check JVM logs; verify OL JAR path. For plan-time failures see `failure-spark_driver_error` § *Schema drift sub-case* |
| `lineage_emission_failed` | `ConnectionError` to Marquez / OL error | Verify Marquez service DNS; check `OPENLINEAGE_URL` |
| `telemetry_unavailable` | OTel/Prometheus `ConnectionError` | Verify OTel Collector and Prometheus services |
| `timeout` | `TimeoutError` / `socket.timeout` / requests timeout | Check service latency; increase timeout env vars |
| `runtime_error` | Unknown exception | Inspect full pipeline logs; add specific handler |

---

## Alert → Trace → Lineage (3-hop triage path)

1. **Alert fires**: `PipelineRunFailed` in Prometheus Alertmanager  
   - `annotations.pipeline_name` → identify pipeline  
   - `annotations.failure_category` → narrow failure type  
   - `annotations.dashboard_link` → open Grafana panel  

2. **Grafana panel** (`pipeline-health` dashboard, `var-pipeline_name` filter):  
   - "Recent Failed Runs" table shows `run_id` from exemplar sampling + Marquez  
   - Click row to open Histogram exemplar → trace  

3. **Marquez** (`lineage-view` dashboard or direct UI):  
   - Enter `run_id` in panel variable  
   - View source/target dataset graph + "Open in Marquez" link  

---

## Trace Observation (OTel Collector logging exporter)

```bash
kubectl logs -n ai-monitor-system \
  -l app.kubernetes.io/name=opentelemetry-collector \
  --tail=100 | grep pipeline_run
```

Span attributes: `pipeline.run_id`, `pipeline.name`, `k8s.namespace`, `status`

---

## `nuke-local.sh` — Usage and Risks

**Purpose**: Brute-force teardown of the `ai-monitor-system` namespace for dev/local environments.  
**Risks**: Irreversible — deletes namespace, all resources, and lingering PVs.  
**Never use on**: Shared, staging, or production clusters.

**When to use**:
- Namespace is stuck (finalizers preventing deletion)
- PVs/PVCs left over from failed deployments
- Rebuilding from scratch after config test pollution

---

## Chart Version Reference

See [chart-version-matrix.md](chart-version-matrix.md) for current pinned versions.

---

## Rollback Checklist

1. Pin back to known-good versions in `deploy/helm/Chart.yaml`
2. Update `deploy/helm/values.local-minimal.yaml` if needed
3. Run `NUKE_BEFORE_BOOTSTRAP=true bash deploy/scripts/bootstrap-local.sh`
4. Run `bash deploy/scripts/check-monitoring-coverage.sh`
5. Run smoke suite: `bash deploy/scripts/run-smoke-test.sh`

---

## Troubleshooting Matrix

| Symptom | Likely Cause | First Check |
|---------|-------------|-------------|
| No metrics on `/metrics` | Endpoint not started | Pod logs for `SystemExit(78)` |
| No traces in Collector | OTLP endpoint wrong | `OTEL_EXPORTER_OTLP_ENDPOINT` env; OTel Collector logs |
| No lineage events | Marquez endpoint mismatch | `OPENLINEAGE_URL` in ConfigMap; Marquez pod logs |
| Dashboards empty | Datasource or scrape config | Grafana datasource; Prometheus targets page |
| Alert not firing | Rules not loaded | Prometheus UI → Status → Rules |
| `chart_version` fallback in report | RBAC missing | `kubectl auth can-i list secrets -n ai-monitor-system` |
| Bootstrap > 10 min | Image pull or resource pressure | Docker local image cache; node resources |

---

## Failure Category Runbook {#failure-categories}

Each section below corresponds to one entry in `KNOWN_CATEGORIES` and one scenario file in `scenarios/`.
Use `INJECT_FAILURE=<category> ./deploy/scripts/run-pipeline.sh` to reproduce locally.

---

### failure-input_not_found {#failure-input_not_found}

**症狀 (Symptoms)**: `PipelineRunFailed` alert fires; `pipeline_failures_total{failure_category="input_not_found"}` increments.

**重現 (Reproduce)**:
```bash
INJECT_FAILURE=input_not_found ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh input-not-found
```

**預期告警 (Expected Alerts)**: `PipelineRunFailed`

**處置 (Resolution)**:
1. Check `INPUT_PATH` env var in the pipeline Job spec.
2. Verify the PVC / hostPath volume is mounted and the source file exists.
3. Re-run pipeline after fixing the path.

---

### failure-invalid_path {#failure-invalid_path}

**症狀**: `PipelineRunFailed` fires; `failure_category="invalid_path"`.

**重現**:
```bash
INJECT_FAILURE=invalid_path ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh invalid-path
```

**預期告警**: `PipelineRunFailed`

**處置**:
1. Verify `INPUT_PATH` points to a regular file, not a directory.
2. Correct the path in the pipeline Job env or ConfigMap.

---

### failure-permission_denied {#failure-permission_denied}

**症狀**: `PipelineRunFailed` fires; `failure_category="permission_denied"`.

**重現**:
```bash
INJECT_FAILURE=permission_denied ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh permission-denied
```

**預期告警**: `PipelineRunFailed`

**處置**:
1. Check the pod ServiceAccount RBAC and volume mount `readOnly` flag.
2. Fix file permissions on the source path or update the SecurityContext.

---

### failure-spark_task_failed {#failure-spark_task_failed}

**症狀**: `PipelineRunFailed` fires; `failure_category="spark_task_failed"`; Py4JJavaError with TaskFailed in message.

**重現**:
```bash
INJECT_FAILURE=spark_task_failed ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh spark-task-failed
```

**預期告警**: `PipelineRunFailed`

**處置**:
1. Examine Spark executor logs for the root task exception.
2. Check for data skew or OOM in executor JVM logs.
3. Reduce partition count or increase executor memory if needed.

---

### failure-spark_driver_error {#failure-spark_driver_error}

**症狀**: `PipelineSparkDriverError` alert fires; `failure_category="spark_driver_error"`; SparkException in driver logs.

**重現**:
```bash
INJECT_FAILURE=spark_driver_error ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh spark-driver-error
```

**預期告警**: `PipelineSparkDriverError`

**處置**:
1. Check driver JVM logs for `SparkException` stack trace.
2. Verify the OpenLineage JAR path (`OPENLINEAGE_JAR_PATH`).
3. For schema mismatch sub-cases, also check lineage run state via:
   ```bash
   uv run python scripts/probe.py lineage-run-state --run-id <run_id> --state-eq FAILED
   ```

#### Schema drift sub-case

PySpark plan-analyzer errors (`UNRESOLVED_COLUMN`, type mismatch, missing
column) are surfaced as `pyspark.errors.exceptions.captured.AnalysisException`
and classified as `spark_driver_error`. The `schema-drift` scenario
reproduces this end-to-end with two sequential runs (baseline schema v1 →
drift schema v2 fails analysis).

**Reproduce**:
```bash
./scripts/run-scenario.sh schema-drift
```

**OpenLineage Spark listener blind spot**: when plan analysis fails, no
Spark job is launched, so the listener never observes the run. The
pipeline therefore emits a shadow OpenLineage `START`+`FAIL` pair from
[lineage_emitter.py](../pipeline/lineage_emitter.py) so Marquez records
the run with state `FAILED` and an `errorMessage` facet (gated by the
`LINEAGE_SHADOW_EMIT` configmap key). The shadow event uses the
pipeline-side `run_id`, identical to the metric exemplar and trace
`run_id`, so the three-way correlation still holds.

**Triage at Marquez**:
```bash
curl http://<marquez>/api/v1/jobs/runs/<run_id>
# state=FAILED, facets.errorMessage.message contains the offending column
```

---

### failure-lineage_emission_failed {#failure-lineage_emission_failed}

**症狀**: `PipelineLineageEmissionFailed` alert fires; `failure_category="lineage_emission_failed"`.

**重現**:
```bash
INJECT_FAILURE=lineage_emission_failed ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh lineage-emission-failed
```

**預期告警**: `PipelineLineageEmissionFailed`

**處置**:
1. Verify Marquez pod is running: `kubectl get pod -n ai-monitor-system -l app=marquez`.
2. Check `OPENLINEAGE_URL` / `OPENLINEAGE_TRANSPORT` in the pipeline ConfigMap.
3. Confirm Marquez health: `curl http://ai-monitor-system-upstream-marquez:9555/api/v1/namespaces`.

---

### failure-telemetry_unavailable {#failure-telemetry_unavailable}

**症狀**: `PipelineTelemetryUnavailable` alert fires; `failure_category="telemetry_unavailable"`. Prometheus lifecycle metric still present (metric path independent of OTel).

**重現**:
```bash
INJECT_FAILURE=telemetry_unavailable ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh telemetry-unavailable
```

**預期告警**: `PipelineTelemetryUnavailable`

**處置**:
1. Check OTel Collector pod: `kubectl logs -n ai-monitor-system -l app.kubernetes.io/name=opentelemetry-collector`.
2. Verify `OTEL_EXPORTER_OTLP_ENDPOINT` env var in pipeline Job.
3. Restart collector if config drift is suspected.

---

### failure-timeout {#failure-timeout}

**症狀**: `PipelineRunTimeout` alert fires; `failure_category="timeout"`.

**重現**:
```bash
INJECT_FAILURE=timeout ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh timeout
```

**預期告警**: `PipelineRunTimeout`

**處置**:
1. Identify which external call timed out from the pipeline logs.
2. Check Marquez / OTel Collector response times.
3. Increase the relevant timeout env var (e.g. `REQUESTS_TIMEOUT_SECONDS`) or address the slow dependency.

---

### failure-runtime_error {#failure-runtime_error}

**症狀**: `PipelineRunFailed` fires; `failure_category="runtime_error"` (catch-all for unclassified exceptions).

**重現**:
```bash
INJECT_FAILURE=runtime_error ./deploy/scripts/run-pipeline.sh
# or: ./scripts/run-scenario.sh runtime-error
```

**預期告警**: `PipelineRunFailed`

**處置**:
1. Inspect full pipeline logs for the exception type and stack trace.
2. If the exception is recurrent, add a specific `classify_failure` rule to `failure_classifier.py` to promote it to a named category.
3. Consider adding a dedicated alert rule if routing to a specific team is needed.
