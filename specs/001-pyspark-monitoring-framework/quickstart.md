# Quickstart: Local Cluster Monitoring Framework (Option B)

## Goal

Run a simple local-file-to-local-file PySpark pipeline on local Kubernetes, with observability components deployed primarily through upstream Helm charts and project overlays.

## Prerequisites

- Local Kubernetes cluster running.
- Helm installed and configured for the active cluster context.
- Docker image build capability for local cluster.
- Recommended local budget: at least 4 CPU and 8 GiB memory available to cluster.

## 1) Prepare sample data

1. Create local input/output directories under `ai-monitor-system/.local-data/`.
2. Place a small input file (for example `sample.txt`) in the input directory.
3. Ensure output directory is writable.

## 2) Bootstrap namespace and core chart overlays

1. Change directory to `ai-monitor-system`.
2. Build local pipeline image.
3. Apply project-owned Helm release for pipeline-specific templates (`pipeline-job`, OpenLineage env/config bridge, Spark defaults).
4. Confirm namespace exists and project release is healthy.

Expected result:

- Pipeline runtime config maps are present.
- Spark OpenLineage listener defaults are injected.

## 3) Deploy upstream observability charts

1. Add/update upstream Helm repositories required for:
   - Prometheus (`prometheus-community/prometheus`)
   - Grafana (`grafana/grafana`)
   - OpenTelemetry Collector (`open-telemetry/opentelemetry-collector`)
   - OpenLineage backend (`marquez/marquez`)
2. Install/upgrade each chart with local-minimal overrides.
3. Wait for all pods/services to become Ready in the same namespace.

Expected result:

- Prometheus is scraping configured targets.
- Grafana is reachable and datasource is connected.
- OTel Collector OTLP receivers are healthy.
- OpenLineage backend is reachable by Spark listener settings.

## 4) Execute pipeline runs

1. Run one successful pipeline execution.
2. Run one failing pipeline execution (for example, invalid input path).
3. Capture run IDs for both cases.

Expected result:

- Success run writes output file and emits terminal status.
- Failure run emits failure status with category and message.

## 5) Validate observability and correlation

1. Confirm dashboard panels reflect run states (`running`, `succeeded`, `failed`).
2. Confirm metrics families include run counters/duration/failure category.
3. Confirm trace spans and lineage events include the same `run_id`.
4. Confirm failure alert includes severity, summary, and run context.
5. Confirm state visibility SLA: transitions visible within 2 minutes.

## 6) Run automated checks

1. Execute contract tests.
2. Execute integration tests.
3. Execute smoke tests.
4. Execute coverage script for required stack components.

Pass criteria:

- No placeholder test remains for required behavior validation.
- All mandatory observability checks pass for success and failure scenarios.
- Correlation and timing contracts are satisfied.

## Troubleshooting Notes

- If metrics are missing, verify scrape target/service discovery and pipeline endpoint exposure strategy.
- If lineage is missing, verify Spark listener config and OpenLineage backend endpoint/namespace.
- If traces are missing, verify OTLP exporter endpoint and collector pipeline configuration.
- If local resources are tight, lower non-critical dashboard refresh and keep single-replica profile while preserving required alerts.
