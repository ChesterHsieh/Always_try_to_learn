# AI Monitor System

**A monitoring-first reference framework for DataOps teams to validate observability tooling against a real PySpark pipeline.**

This repo is a runnable best-practice template that answers one question:
**"Can my observability stack actually detect, classify, correlate, and alert on real pipeline failures?"**

It ships with a deliberately simple PySpark batch pipeline plus 10 reproducible
failure scenarios and a probe harness that asserts the monitoring stack reacts
correctly to each one. Use it as-is to evaluate a stack, or fork it as the
starting point for your own pipeline's monitoring baseline.

---

## What this gives you

| Capability | How |
|---|---|
| **End-to-end observability stack on local Kubernetes** | Upstream Helm charts (Prometheus / Grafana / OpenTelemetry Collector / Marquez / Tempo) wired into one chart with sensible defaults |
| **Run-level correlation across metrics / traces / lineage** | Shared `run_id` propagated through Prometheus exemplars, OTel span attributes, and OpenLineage events |
| **10 reproducible failure scenarios** | `scenarios/*.yaml` — each declares expected category, alerts, and probes; runnable individually or as a batch |
| **Probe-driven verification** | `scripts/probe.py` queries Prometheus / Tempo / Marquez and emits single-line PASS/FAIL verdicts with hints |
| **Layered test suite** | Contract (≤ 5 s, no cluster) → Integration (stubbed cluster) → Smoke (live cluster) |
| **Coverage CLI** | `pipeline.coverage` produces a JSON report tying chart versions, alert rules, dashboards, and lineage state into one release-gate artifact |
| **Operator runbook** | Per-category symptoms, reproduction commands, expected alerts, and triage paths |

---

## Required Stack

| Component | Role | Default version |
|---|---|---|
| **OpenLineage** | Pipeline-side lineage emission (Spark listener + shadow emitter) | bundled in pipeline image |
| **Marquez** | OpenLineage backend; queryable run state and dataset versioning | upstream chart `6.7.0` |
| **Prometheus** | Metric collection + alert evaluation | upstream chart `25.27.0` |
| **OpenTelemetry Collector** | Trace ingestion → Tempo | upstream chart `0.78.0` |
| **Grafana Tempo** | Trace storage and query | upstream chart |
| **Grafana** | Dashboards (`pipeline-health`, `lineage-view`) + alert visualization | upstream chart `8.5.1` |
| **Helm 3** | Deployment orchestrator | host-provided |
| **Local Kubernetes** | kind / minikube / Rancher Desktop / Docker Desktop K8s | host-provided |

> The chart is configuration-first. Project-owned templates are minimal —
> almost everything is upstream chart values. See
> [docs/chart-version-matrix.md](docs/chart-version-matrix.md).

---

## Project Layout

| Path | Purpose |
|---|---|
| [`pipeline/`](pipeline/) | PySpark job + telemetry / tracing / lineage helpers + failure classifier + scenario schema |
| [`scenarios/`](scenarios/) | Declarative failure-scenario YAML files consumed by the runner |
| [`scripts/`](scripts/) | `run-scenario.sh`, `run-all-failure-scenarios.sh`, `probe.py` |
| [`deploy/helm/`](deploy/helm/) | Helm chart, upstream chart deps, values overlays, project glue templates |
| [`deploy/scripts/`](deploy/scripts/) | `bootstrap-local.sh`, `run-pipeline.sh`, `run-smoke-test.sh`, `check-monitoring-coverage.sh`, `nuke-local.sh` |
| [`monitoring/`](monitoring/) | Grafana dashboards JSON + Prometheus alert-rule YAML (mounted via chart) |
| [`tests/contract/`](tests/contract/) | Pure-Python invariants (no cluster) — run on every change |
| [`tests/integration/`](tests/integration/) | Wiring tests with stubbed cluster components |
| [`tests/smoke/`](tests/smoke/) | End-to-end smoke against a live local cluster |
| [`docs/`](docs/) | Onboarding, runbook (per-category triage), validation report |

---

## Quick Start

```bash
# 1. Bootstrap stack (creates namespace, builds pipeline image, helm install)
./deploy/scripts/bootstrap-local.sh

# 2. Trigger one happy-path run
./deploy/scripts/run-pipeline.sh

# 3. Verify the stack actually observed it
./scripts/run-scenario.sh success-baseline

# 4. Run all failure scenarios end-to-end and refresh the validation ledger
./scripts/run-all-failure-scenarios.sh --update-report
```

Namespace defaults to `ai-monitor-system`; release name defaults to `monitor`.

### Local UIs (NodePort — no port-forward needed)

| Service | URL |
|---|---|
| Grafana | http://localhost:30300 |
| Prometheus | http://localhost:30090 |
| Marquez Web | http://localhost:30444 |
| Marquez API | http://localhost:30555 |
| Grafana Tempo | http://localhost:30318 |

---

## Failure Scenario Catalog

Each scenario is a single YAML file declaring (a) what to inject, (b) the
expected lifecycle outcome, (c) the alerts that must fire, and (d) the probe
queries that prove the stack saw it. Run individually with `./scripts/run-scenario.sh <name>`.

| Scenario | Category | Tests detection of |
|---|---|---|
| [`success-baseline.yaml`](scenarios/success-baseline.yaml) | _(success)_ | Healthy run baseline; no alerts firing |
| [`input-not-found.yaml`](scenarios/input-not-found.yaml) | `input_not_found` | `FileNotFoundError` from missing input |
| [`invalid-path.yaml`](scenarios/invalid-path.yaml) | `invalid_path` | `IsADirectoryError` from path mis-config |
| [`permission-denied.yaml`](scenarios/permission-denied.yaml) | `permission_denied` | `PermissionError` from RBAC / mount |
| [`spark-task-failed.yaml`](scenarios/spark-task-failed.yaml) | `spark_task_failed` | Py4JJavaError task failure |
| [`spark-driver-error.yaml`](scenarios/spark-driver-error.yaml) | `spark_driver_error` | SparkException at driver |
| [`schema-drift.yaml`](scenarios/schema-drift.yaml) | `spark_driver_error` | **Two-run schema drift** — baseline writes `value: STRING`, drift run reads with mismatched schema → `AnalysisException` (`UNRESOLVED_COLUMN`); proves Marquez records `state=FAILED` with `errorMessage` facet despite OpenLineage Spark listener's plan-time blind spot |
| [`lineage-emission-failed.yaml`](scenarios/lineage-emission-failed.yaml) | `lineage_emission_failed` | Marquez unreachable / OpenLineage error |
| [`telemetry-unavailable.yaml`](scenarios/telemetry-unavailable.yaml) | `telemetry_unavailable` | OTel collector / Prometheus unreachable |
| [`timeout.yaml`](scenarios/timeout.yaml) | `timeout` | `TimeoutError` / `socket.timeout` |
| [`runtime-error.yaml`](scenarios/runtime-error.yaml) | `runtime_error` | Catch-all for unclassified exceptions |

Each scenario's probes assert across **three signal paths simultaneously**:
1. Prometheus metric label (`pipeline_failures_total{failure_category="..."}`)
2. Alert state (`ALERTS{alertname="...",alertstate="firing"}`)
3. Lineage / trace correlation (where applicable)

> **Why the schema-drift scenario is special**: PySpark plan-analyzer
> failures (e.g. `UNRESOLVED_COLUMN`) raise `AnalysisException` _before_
> Spark launches a job, so the OpenLineage Spark listener never observes
> the run. The pipeline therefore emits a shadow `START`+`FAIL` OpenLineage
> event from [`pipeline/lineage_emitter.py`](pipeline/lineage_emitter.py)
> using its own `run_id` — preserving the three-way correlation. This is a
> general pattern for any "fail before Spark engine starts" class of bug.

### Anatomy of a scenario file

```yaml
name: input-not-found
description: Input file missing; pipeline raises FileNotFoundError
pipeline:
  input_records: 0
  inject_failure: input_not_found     # one of pipeline.failure_injection.SUPPORTED_INJECTIONS
  schema_version: v1                  # optional — drives schema-drift mode
  pre_runs:                           # optional — multi-run scenarios (baseline → drift)
    - schema_version: v1
      inject_failure: none
expected_run_status: failed           # succeeded | failed
expected_failure_category: input_not_found  # one of KNOWN_CATEGORIES, or null
expected_alerts:
  - PipelineRunFailed
probes:                               # validated against the live monitoring stack
  - id: failure_category_metric
    cmd: prom-query
    args:
      query: 'pipeline_failures_total{failure_category="input_not_found"}'
      gte: 1
      within: 60
  - id: alert_firing
    cmd: prom-query
    args:
      query: 'ALERTS{alertname="PipelineRunFailed",alertstate="firing"}'
      gte: 1
      within: 60
```

### Probes

Defined in [`scripts/probe.py`](scripts/probe.py). Use them inline in
scenarios or one-off from the CLI.

| `cmd` | Backend | Asserts |
|---|---|---|
| `prom-query` | Prometheus | a PromQL expression evaluates to ≥ N / ≤ N / == N within a window |
| `otel-trace` | Tempo | recent trace from a service exists and contains specified attributes (suffix-matched, so `--has-attr run_id` matches `pipeline.run_id`) |
| `lineage-run-state` | Marquez (OpenLineage) | a given `run_id` reaches a target state (e.g. `FAILED`) within a window |

Each probe emits a single-line JSON verdict with `verdict`, `actual`, `latency_ms`, and a `hint` on failure.

---

## Test Layers

| Layer | When to run | Time | Cluster needed? | Asserts |
|---|---|---|---|---|
| **Contract** ([`tests/contract/`](tests/contract/)) | Every code change | < 5 s | No | Type-level invariants: KNOWN_CATEGORIES stable, scenario YAML schema valid, classifier deterministic, metric labels frozen, three-way alignment between scenarios / runbook / alerts |
| **Integration** ([`tests/integration/`](tests/integration/)) | Before declaring code change "done" | ~10 s | No (stubbed) | Wiring: failure injection → metric increment, OTel span attributes set, alert YAML structure, lifecycle payload shape, lineage-run-state probe behavior |
| **Smoke** ([`tests/smoke/`](tests/smoke/)) | Before committing pipeline / Helm / script changes | minutes | **Yes** | End-to-end: bootstrap → pipeline → probes → coverage; nuke + rebuild idempotency |
| **Live scenario harness** (`./scripts/run-all-failure-scenarios.sh`) | Release gate; recurring monitoring health check | minutes | **Yes** | Every failure category produces matching metric label, alert, and lineage state in real Marquez/Prometheus/Tempo |

```bash
# Inner loop (fastest)
uv run ruff format . && uv run ruff check . --fix
uv run pytest -q tests/contract

# Pre-commit
uv run pytest -q tests/contract tests/integration

# Pre-release
./deploy/scripts/run-smoke-test.sh
./scripts/run-all-failure-scenarios.sh --update-report
./deploy/scripts/check-monitoring-coverage.sh
```

---

## Coverage CLI — release-gate artifact

```bash
python -m pipeline.coverage \
  --namespace ai-monitor-system \
  --marquez-url http://ai-monitor-system-upstream-marquez:9555 \
  --prometheus-url http://ai-monitor-system-upstream-prometheus-server:80 \
  --grafana-url http://ai-monitor-system-upstream-grafana:80 \
  --output .local-data/coverage/release.json
```

| Exit code | Meaning |
|---|---|
| `0` | All checks pass |
| `1` | Warning (stale lineage, datasource latency) |
| `2` | Critical (Prometheus unreachable, rules not loaded, Grafana / Marquez down) |

The JSON report includes pinned chart versions for all four upstream charts,
a list of validation checks (each `pass`/`warn`/`fail` with detail), and
`last_verified_at` — meant to be archived per release.

---

## Adapting this to your own pipeline

1. **Replace [`pipeline/job.py`](pipeline/job.py)** with your Spark job. Keep
   the lifecycle envelope — `record_run_started`, `record_run_succeeded` /
   `record_run_failed`, `start_run_span`, `maybe_shadow_emit` — so all three
   signal paths share the same `run_id`.
2. **Reuse [`pipeline/failure_classifier.py`](pipeline/failure_classifier.py)**
   as-is. The 9 categories cover most generic batch failure modes; extend
   `KNOWN_CATEGORIES` only when a category needs a separate alert routing.
3. **Author scenarios** for whatever failure modes matter to your team — the
   YAML schema is in [`pipeline/scenario_schema.py`](pipeline/scenario_schema.py).
4. **Run the harness in CI** against an ephemeral local cluster
   (`./scripts/run-all-failure-scenarios.sh --update-report`) and treat
   anything less than full pass as a release block.
5. **Read [`docs/runbook.md`](docs/runbook.md)** before customizing alerts —
   the alert/dashboard/runbook three-way alignment is enforced by
   [`tests/contract/test_coverage_alignment_contract.py`](tests/contract/test_coverage_alignment_contract.py),
   so additions must update all three.

---

## Design principles (why it looks like this)

- **Configuration-first integration.** Upstream Helm charts are the source
  of truth for stack components; project templates are glue only.
- **One `run_id` across signals.** Pipeline generates a single UUID; metrics
  exemplars, OTel span attrs, OpenLineage events all carry it. This is what
  makes alert → trace → lineage triage work in three hops.
- **Cardinality discipline.** `run_id` and `failure_message` live in
  Prometheus _exemplars_, never in metric labels — keeps `pipeline_failures_total`
  cardinality bounded as scenarios grow.
- **Probes assert what humans care about.** Probes query the live backends
  rather than internal mocks, so a green run is real evidence the stack would
  catch that failure in production.
- **Three-way alignment is a contract.** A failure category exists only if
  `scenarios/<x>.yaml` + `monitoring/alerts/...` + `docs/runbook.md` all
  reference it. Drift between them is a contract-test failure.
- **Plan-time failures need shadow emission.** The OpenLineage Spark
  listener cannot observe failures before a Spark job is launched (e.g.
  schema drift). The pipeline backstops this by emitting OpenLineage events
  itself on the failure path — preserving lineage detection coverage.

---

## Validation ledger

[`docs/validation-report.md`](docs/validation-report.md) is updated by
`./scripts/run-all-failure-scenarios.sh --update-report` and tracks
per-scenario `last_run_at` + result. Use it as the release-gate witness.

---

## Operational notes

- Pipeline image is built locally as `local/ai-monitor-pyspark:latest` by
  `bootstrap-local.sh` (uses `imagePullPolicy: IfNotPresent` to avoid registry pulls).
- Helm `--create-namespace` is used; reuses existing namespace if present.
- `nuke-local.sh` tears down the namespace + PV/PVC; treat as destructive
  and confirm before invoking.
- For deeper agent context (CLI shortcuts, common failures, probe usage),
  see [`CLAUDE.md`](CLAUDE.md).
