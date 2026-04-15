# Monitoring Contract

## Purpose

Define the minimum observable outputs required for each PySpark pipeline run so dashboards, alerting, and lineage tracing remain consistent.

## Contract Scope

- Applies to local-file-to-local-file pipeline runs in this feature.
- Covers run status, metrics, traces, and lineage metadata.
- Defines required fields and minimum expected timing behavior.

## Required Run Lifecycle Events

### 1) Run Started

- **When emitted**: Immediately after run execution begins.
- **Required fields**:

  - `run_id`
  - `pipeline_name`
  - `status=running`
  - `start_time`
  - `input_path`
  - `output_path`

### 2) Run Completed Successfully

- **When emitted**: On successful completion.
- **Required fields**:

  - `run_id`
  - `status=succeeded`
  - `end_time`
  - `duration_ms`
  - `records_processed` (if available)

### 3) Run Failed

- **When emitted**: On terminal failure.
- **Required fields**:

  - `run_id`
  - `status=failed`
  - `end_time`
  - `failure_category`
  - `failure_message`

## Required Metric Families

- `pipeline_run_total{status=...}`
- `pipeline_run_duration_seconds`
- `pipeline_records_processed_total`
- `pipeline_failures_total{failure_category=...}`
- `pipeline_telemetry_freshness_seconds`

## Required Trace Attributes

- Every pipeline trace span MUST include:

  - `run_id`
  - `pipeline_name`
  - `k8s_namespace`
  - `status` (for terminal span)

## Required Lineage Attributes

- Every lineage event MUST include:

  - `run_id`
  - `job_name`
  - `job_namespace`
  - `source_dataset`
  - `target_dataset`
  - `event_time`

## Alert Contract

- **Critical condition**: run failure detected.
- **Warning condition**: telemetry freshness exceeds configured threshold.
- **Alert payload minimum**:

  - `severity`
  - `summary`
  - `trigger_time`
  - `run_id` (if run-scoped)
  - `dashboard_link`

## Timing Contract

- Run state transitions (`running`, `succeeded`, `failed`) MUST be observable in monitoring views within 2 minutes under normal local cluster conditions.

## Validation Contract

- Smoke test MUST execute one success and one failure run and verify:

  - lifecycle events emitted
  - required metrics visible
  - trace + lineage records present with shared `run_id`
  - alert generated for failed run
