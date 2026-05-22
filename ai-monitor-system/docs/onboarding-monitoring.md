# Onboarding: PySpark Monitoring Framework

**Owner**: Platform team
**Input**: Local Kubernetes cluster (kind / minikube / Rancher Desktop / Docker Desktop), ≥4 CPU, ≥8 GB RAM
**Output**: Upstream observability stack deployed + pipeline Job runnable in ≤10 min bootstrap, ≤5 min success run, ≤15 min full failure-scenario verification

This guide takes a new operator from a clean cluster to a complete
release-gate witness. Skim §1–4 to get green; read §5 before declaring the
monitoring stack production-ready for your pipeline.

> Looking for the project overview, design principles, and "adapt to your
> own pipeline" guide? See [README.md](../README.md). This document is the
> hands-on operator path.

---

## Deployment Model

This framework uses the **single-stack upstream chart model**: all
observability components are deployed via upstream Helm chart dependencies.
There is no self-managed stack, no migration flag, and no feature flag mutex.
The only deployment path is:

```
bootstrap-local.sh → helm upgrade --install (Prometheus + Grafana + OTel Collector + Marquez + Tempo)
```

---

## 1. Prerequisites

```bash
# Ensure a local k8s cluster is running
kubectl cluster-info

# Ensure Helm 3 is installed
helm version

# Ensure uv is installed (project uses pyproject.toml + uv.lock)
uv --version
```

---

## 2. Bootstrap the stack (≤10 min)

```bash
cd ai-monitor-system
./deploy/scripts/bootstrap-local.sh
```

Expected output ends with:
```
Local stack bootstrapped in namespace: ai-monitor-system
BOOTSTRAP_DURATION_SECONDS=<n>
```

The script builds `local/ai-monitor-pyspark:latest` locally then runs
`helm upgrade --install` against the `ai-monitor-system` namespace
(`--create-namespace`; reuses existing namespace if present).

---

## 3. Trigger a happy-path run (≤5 min)

```bash
./deploy/scripts/run-pipeline.sh
```

Expected output ends with:
```
RUN_DURATION_SECONDS=<n>
```

After the run completes, the pushgateway holds the run's metrics:
- `pipeline_run_total{status="succeeded"}` → 1
- `pipeline_records_processed_total` → ≥ 1

---

## 4. Verify the stack saw it

```bash
./scripts/run-scenario.sh success-baseline
```

Expected: `VERDICT: 4/4 PASS`. The probes assert that Prometheus scraped
the metrics, the OTel Collector forwarded spans, and no failure alerts are
firing — the minimum proof that all four signal paths are live.

---

## 5. Verify the stack catches failures (≤15 min)

This is the step most observability checklists skip. A green happy-path run
is **necessary but not sufficient** — the stack also has to catch failures.
Run the full failure catalog end-to-end:

```bash
./scripts/run-all-failure-scenarios.sh --update-report
```

This iterates every failure scenario in [`scenarios/`](../scenarios/),
triggers the corresponding pipeline failure, runs each scenario's probes
against the live stack, and writes the verdict into the
[Failure Scenario Validation Ledger](validation-report.md#failure-scenario-ledger).
Treat anything less than full pass as a release block.

The 10 failure scenarios cover every category in
`pipeline.failure_classifier.KNOWN_CATEGORIES` plus one **two-run schema
mismatch** case (`schema-mismatch`) — see [README.md § Failure Scenario Catalog](../README.md#failure-scenario-catalog) for the full list.

---

## 6. Run the coverage CLI (release-gate artifact)

```bash
./deploy/scripts/check-monitoring-coverage.sh
# OR directly:
python -m utils.coverage --output .local-data/coverage/latest.json
```

Exit codes:

| Code | Meaning |
|---|---|
| `0` | All checks pass |
| `1` | Warning (stale lineage, helm secret fallback, datasource latency) |
| `2` | Critical (backend unreachable, rules not loaded) |

The JSON report archives pinned chart versions for all four upstream charts,
each validation check's status (`pass` / `warn` / `fail`) with detail, and
`last_verified_at` — meant to be archived per release.

---

## Local UIs

All services are exposed as NodePort on `localhost`. No `kubectl port-forward` required.

| Service | URL | Notes |
|---|---|---|
| Prometheus | http://localhost:30090 | Query metrics + alerts |
| Grafana | http://localhost:30300 | Anonymous admin; `pipeline-health` and `lineage-view` dashboards auto-loaded |
| Grafana Tempo | http://localhost:30318 | HTTP query API (`/api/search`, `/api/traces/{id}`) |
| Marquez API | http://localhost:30555 | OpenLineage-compatible lineage backend |
| Marquez Web | http://localhost:30444 | DAG lineage viewer |

In-cluster service URLs (used by pipeline pods, scrape targets, dashboards):

| Service | In-cluster URL |
|---|---|
| Pushgateway | `http://pushgateway:9091` |
| OTel Collector | `http://otel-collector:4317` (OTLP gRPC) |
| Marquez | `http://ai-monitor-system-upstream-marquez:9555` |

---

## Metrics Endpoint

The pipeline driver exposes an **OpenMetrics** endpoint on port `9095` and
also pushes final metrics to Pushgateway at run termination:

```bash
curl http://localhost:9095/metrics
```

Content-Type: `application/openmetrics-text; version=1.0.0; charset=utf-8`

### 5 Metric Families

| Family | Type | Labels |
|---|---|---|
| `pipeline_run_total` | Counter | `status`, `pipeline_name` |
| `pipeline_run_duration_seconds` | Histogram | `status`, `pipeline_name` |
| `pipeline_records_processed_total` | Counter | `pipeline_name` |
| `pipeline_failures_total` | Counter | `failure_category`, `pipeline_name` |
| `pipeline_telemetry_freshness_seconds` | Gauge | `pipeline_name` |

> **Cardinality discipline**: `run_id` and `failure_message` are **not**
> metric labels — they ride OpenMetrics **exemplars** on Counter/Histogram
> observations. Adding a new failure category is a 9-cardinality decision;
> keeping `run_id` out of labels prevents per-run cardinality explosion.

### Three-hop drilldown path

```
Alert (Prometheus) → Grafana panel (pipeline_name filter) → Marquez lineage-view (run_id) → root cause
```

The shared `run_id` is what keeps all three hops connected.

---

## Trace Observation

Traces are exported via OTLP gRPC from the pipeline pod to the OTel Collector,
then forwarded to Grafana Tempo for storage and query.

```bash
# Search recent traces by service
curl 'http://localhost:30318/api/search?tags=service.name%3Dpyspark-pipeline&limit=5' | jq

# Fetch a full trace by ID (includes all span attributes)
curl http://localhost:30318/api/traces/<trace_id> | jq
```

Each span carries:
- `pipeline.run_id` — shared with metrics exemplars and OpenLineage runId
- `pipeline.name` — pipeline identifier
- `k8s.namespace`
- `status` — `succeeded` / `failed`
- `error=true` (set on failure spans)

> Tempo's `/api/search` only echoes attributes that match the search query;
> `--has-attr` checks must hydrate via `/api/traces/{id}`. The
> [`scripts/probe.py otel-trace`](../scripts/probe.py) probe handles this
> two-step lookup automatically.

---

## Lineage emission

Two lineage event sources cooperate to give Marquez complete coverage:

| Source | Driven by | Run ID | Covers |
|---|---|---|---|
| **OpenLineage Spark listener** | `spark.extraListeners` config in pipeline | listener-generated UUID | normal Spark execution (job submitted, plan analyzed, tasks run, COMPLETE/FAIL emitted) |
| **Pipeline shadow emitter** ([`telemetry/lineage_emitter.py`](../telemetry/lineage_emitter.py)) | `LINEAGE_SHADOW_EMIT=true` (configmap default) | pipeline's own `run_id` | failures **before** Spark launches a job (e.g. plan-analyzer failures like a schema mismatch between runs); preserves three-way correlation |

**Why both are needed**: PySpark plan-analyzer failures
(`AnalysisException`, `UNRESOLVED_COLUMN`, type mismatch) raise before
`SparkContext` submits any job, so the Spark listener never observes them.
Without the shadow emitter, those failures would be invisible at Marquez —
exactly the kind of bug operators most need lineage for. The shadow
emitter sends a `START` event followed by the terminal event (`COMPLETE`
or `FAIL` with an `errorMessage` facet) using the pipeline's own `run_id`,
so Grafana → Marquez drilldown still works.

To inspect a specific run at Marquez:

```bash
curl http://localhost:30555/api/v1/jobs/runs/<run_id> | jq '{state, startedAt, endedAt, facets}'
```

---

## Failure Categories at a Glance

All 9 categories are defined in
[`pipeline.failure_classifier.KNOWN_CATEGORIES`](../pipeline/failure_classifier.py).
Each has at least one `scenarios/<name>.yaml`, one runbook section, and one
matching alert rule — three-way alignment is enforced by
[`tests/contract/test_coverage_alignment_contract.py`](../tests/contract/test_coverage_alignment_contract.py).

| Category | Trigger (Python exception → classifier output) | Alert |
|---|---|---|
| `input_not_found` | `FileNotFoundError` | `PipelineRunFailed` |
| `invalid_path` | `IsADirectoryError` | `PipelineRunFailed` |
| `permission_denied` | `PermissionError` | `PipelineRunFailed` |
| `spark_task_failed` | `Py4JJavaError` (task-level) | `PipelineRunFailed` |
| `spark_driver_error` | `Py4JJavaError` (driver/SparkException) **or** `pyspark.errors.AnalysisException` (plan-analyzer) | `PipelineSparkDriverError` |
| `lineage_emission_failed` | `ConnectionError` to Marquez / OL error | `PipelineLineageEmissionFailed` |
| `telemetry_unavailable` | OTel/Prometheus `ConnectionError` | `PipelineTelemetryUnavailable` |
| `timeout` | `TimeoutError` / `socket.timeout` / requests timeout | `PipelineRunTimeout` |
| `runtime_error` | Unknown exception (catch-all) | `PipelineRunFailed` |

For per-category symptoms, reproduction commands, and triage steps see
[`docs/runbook.md`](runbook.md).

---

## Test Suites

```bash
# Inner loop — every code change (< 5 s, no cluster)
uv run ruff format . && uv run ruff check . --fix
uv run pytest -q tests/contract

# Pre-commit — before declaring code change "done" (~10 s, stubbed)
uv run pytest -q tests/contract tests/integration

# Pre-release — before pushing pipeline / Helm / script changes (live cluster)
./deploy/scripts/run-smoke-test.sh
./scripts/run-all-failure-scenarios.sh --update-report
./deploy/scripts/check-monitoring-coverage.sh
```

| Layer | Time | Cluster | Asserts |
|---|---|---|---|
| Contract | < 5 s | No | KNOWN_CATEGORIES stable, scenario YAML schema, classifier determinism, metric labels frozen, three-way alignment |
| Integration | ~10 s | No (stubbed) | Failure injection wiring, OTel span attrs, alert YAML structure, lifecycle payloads |
| Smoke | minutes | Yes | Bootstrap → pipeline → probes → coverage; nuke + rebuild idempotency |
| Live scenario harness | minutes | Yes | Every failure category produces matching metric label, alert, and lineage state in real backends |

---

## Dev Teardown and Rebuild

> **Warning**: `nuke-local.sh` deletes the namespace and its PV/PVCs. Use only
> on local/dev clusters; never run on a shared cluster. Confirm with the
> namespace owner before invoking.

```bash
# Full teardown
./deploy/scripts/nuke-local.sh

# Nuke then rebuild in one step
NUKE_BEFORE_BOOTSTRAP=true ./deploy/scripts/bootstrap-local.sh
```

`NUKE_BEFORE_BOOTSTRAP=false` by default; must be explicitly set to `true`.

---

## Troubleshooting Quick Reference

| Symptom | Check |
|---|---|
| No metrics on `/metrics` | Verify `METRICS_PORT=9095`; check pod logs for exit code 78 (metrics endpoint failed to start) |
| No traces in Tempo | Verify `OTEL_EXPORTER_OTLP_ENDPOINT`; check Collector pod logs; test with `curl 'http://localhost:30318/api/search?...'` |
| No lineage events at Marquez | Check `OPENLINEAGE_URL` in `openlineage-config` ConfigMap; verify `LINEAGE_SHADOW_EMIT=true` for plan-time failure coverage |
| Stale telemetry alert firing | Run `update_freshness`; verify pipeline completes successfully |
| Coverage CLI exit 2 | Backend unreachable — check `kubectl get pods -n ai-monitor-system`; check NodePort 30090 / 30300 / 30555 |
| `chart_version` shows fallback | RBAC ServiceAccount may lack `secrets:get,list` — check Role |
| `helm upgrade` fails on `Job ... is invalid: spec.template ... immutable` | Existing pipeline Job blocks template changes — `kubectl delete job pyspark-pipeline -n ai-monitor-system --ignore-not-found` then retry |
| Marquez returns 400 on shadow emit | OpenLineage payload missing `producer` / `schemaURL`, or runId is not a UUID, or terminal event sent without preceding `START` |

For deeper agent context (probe usage patterns, common pitfalls, common
operations), see [`CLAUDE.md`](../CLAUDE.md).
