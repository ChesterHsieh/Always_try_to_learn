# Data Model: PySpark Monitoring Framework (Option B)

## Entity: PipelineRun

- **Description**: One execution instance of the reference PySpark job (local input file to local output file) running on Kubernetes.
- **Fields**:
  - `run_id` (string, required, unique)
  - `pipeline_name` (string, required)
  - `status` (enum: queued, running, succeeded, failed, recovering)
  - `start_time` (datetime, required)
  - `end_time` (datetime, optional)
  - `duration_ms` (integer, optional)
  - `input_path` (string, required)
  - `output_path` (string, required)
  - `failure_category` (string, optional)
  - `failure_message` (string, optional)
  - `k8s_namespace` (string, required)
  - `k8s_job_name` (string, required)
- **Validation Rules**:
  - `status=failed` requires `failure_category` and `failure_message`.
  - Terminal statuses (`succeeded`, `failed`) require `end_time`.
  - `duration_ms` MUST be non-negative when present.
- **State Transitions**:
  - queued -> running
  - running -> succeeded | failed
  - failed -> recovering (optional operator state)

## Entity: MonitoringSignal

- **Description**: Normalized envelope for emitted telemetry records across metric/trace/lineage channels.
- **Fields**:
  - `signal_id` (string, required, unique)
  - `signal_type` (enum: metric, trace, lineage, alert)
  - `run_id` (string, required for run-scoped events)
  - `source_component` (string, required; e.g., pipeline-job, otel-collector, prometheus-rule)
  - `timestamp` (datetime, required)
  - `attributes` (map<string,string>, optional)
- **Validation Rules**:
  - Run-scoped signals MUST include `run_id`.
  - `signal_type` determines minimum required attributes defined in contract docs.

## Entity: LineageRecord

- **Description**: OpenLineage-compatible event view linked to a `PipelineRun`.
- **Fields**:
  - `lineage_event_id` (string, required, unique)
  - `run_id` (string, required, foreign key -> PipelineRun.run_id)
  - `job_name` (string, required)
  - `job_namespace` (string, required)
  - `source_dataset` (string, required)
  - `target_dataset` (string, required)
  - `event_time` (datetime, required)
- **Validation Rules**:
  - `source_dataset` and `target_dataset` MUST be non-empty.
  - `job_namespace` SHOULD match deployment namespace unless explicitly overridden.

## Entity: AlertEvent

- **Description**: Alert instance generated from monitoring rules and consumed by operator workflows.
- **Fields**:
  - `alert_id` (string, required, unique)
  - `run_id` (string, optional)
  - `severity` (enum: warning, critical)
  - `summary` (string, required)
  - `trigger_time` (datetime, required)
  - `dashboard_link` (string, optional)
  - `state` (enum: firing, resolved)
- **Validation Rules**:
  - Run-failure alerts SHOULD include `run_id`.
  - Critical alerts MUST include actionable `summary`.

## Entity: CoverageProfile

- **Description**: Versioned readiness profile defining required observability components and checks for onboarding/release.
- **Fields**:
  - `profile_name` (string, required; default `local-minimal`)
  - `profile_version` (string, required)
  - `components` (string list, required; OpenLineage, Prometheus, OTel Collector, Grafana)
  - `resource_budget` (map, required; cpu/memory defaults)
  - `validation_checks` (string list, required)
  - `last_verified_at` (datetime, optional)
- **Validation Rules**:
  - All required components MUST be present before production readiness sign-off.
  - Validation checks MUST map to automated tests/scripts.

## Entity: ChartReleaseBinding

- **Description**: Mapping of observability components to upstream chart release configuration and pinned versions.
- **Fields**:
  - `component_name` (string, required)
  - `chart_name` (string, required)
  - `chart_repo` (string, required)
  - `chart_version` (string, required)
  - `namespace` (string, required)
  - `values_overlay_path` (string, required)
  - `enabled` (boolean, required)
- **Validation Rules**:
  - `chart_version` MUST be pinned (no floating latest tags).
  - `values_overlay_path` MUST exist in repository.

## Relationship Summary

- `PipelineRun` is the primary operational entity.
- `MonitoringSignal`, `LineageRecord`, and run-scoped `AlertEvent` associate to `PipelineRun` via `run_id`.
- `CoverageProfile` defines expected component and validation coverage.
- `ChartReleaseBinding` describes Option B deployment ownership for observability stack components.
