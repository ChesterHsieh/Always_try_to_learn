# Data Model: PySpark Monitoring Framework

## Entity: PipelineRun

- **Description**: One execution of the reference PySpark job processing local input file(s) into local output file(s).
- **Fields**:

  - `run_id` (string, required, unique)
  - `pipeline_name` (string, required)
  - `status` (enum: queued, running, succeeded, failed, recovering)
  - `start_time` (datetime, required)
  - `end_time` (datetime, optional until completion)
  - `input_path` (string, required)
  - `output_path` (string, required)
  - `failure_category` (string, optional)
  - `k8s_namespace` (string, required)
  - `k8s_pod_name` (string, optional)
- **Validation Rules**:

  - `run_id` MUST be present for all telemetry events.
  - `status=failed` MUST include `failure_category`.
  - `end_time` MUST be >= `start_time` when present.
- **State Transitions**:

  - queued -> running
  - running -> succeeded | failed
  - failed -> recovering (optional operational state)

## Entity: LineageRecord

- **Description**: Data movement metadata tied to a `PipelineRun`.
- **Fields**:

  - `lineage_event_id` (string, required, unique)
  - `run_id` (string, required, foreign key -> PipelineRun.run_id)
  - `source_dataset` (string, required)
  - `target_dataset` (string, required)
  - `event_time` (datetime, required)
  - `job_name` (string, required)
  - `job_namespace` (string, required)
- **Validation Rules**:

  - `run_id` MUST match an existing pipeline run.
  - Source and target dataset identifiers MUST not be empty.

## Entity: MetricSample

- **Description**: Operational metric point generated for run health and system behavior.
- **Fields**:

  - `metric_name` (string, required)
  - `run_id` (string, optional for infrastructure-wide metrics)
  - `value` (number, required)
  - `labels` (key-value map, optional)
  - `timestamp` (datetime, required)
- **Validation Rules**:

  - Run-scoped metrics SHOULD include `run_id`.
  - `metric_name` MUST align with agreed naming conventions.

## Entity: TraceSpan

- **Description**: Distributed tracing unit representing a logical work segment in pipeline execution.
- **Fields**:

  - `trace_id` (string, required)
  - `span_id` (string, required)
  - `parent_span_id` (string, optional)
  - `run_id` (string, required)
  - `operation_name` (string, required)
  - `start_time` (datetime, required)
  - `end_time` (datetime, required)
  - `status_code` (enum: unset, ok, error)
- **Validation Rules**:

  - `run_id` MUST be attached for pipeline-relevant spans.
  - `end_time` MUST be >= `start_time`.

## Entity: AlertEvent

- **Description**: Operator-facing event emitted when monitoring conditions are met.
- **Fields**:

  - `alert_id` (string, required, unique)
  - `run_id` (string, optional)
  - `severity` (enum: warning, critical)
  - `trigger_time` (datetime, required)
  - `summary` (string, required)
  - `state` (enum: firing, resolved)
- **Validation Rules**:

  - Critical alerts MUST include actionable summary text.
  - Alert state transitions MUST follow firing -> resolved.

## Relationship Summary

- `PipelineRun` is the primary entity.
- `LineageRecord`, `TraceSpan`, and run-scoped `MetricSample` associate to `PipelineRun` via `run_id`.
- `AlertEvent` may associate to `PipelineRun` for run-level incidents or exist cluster-wide for stack health conditions.
