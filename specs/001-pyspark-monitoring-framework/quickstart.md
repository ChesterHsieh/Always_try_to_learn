# Quickstart: Local Cluster Monitoring Framework

## Goal

Run a simple local-file-to-local-file PySpark pipeline on a local Kubernetes cluster with monitoring enabled via OpenLineage, Prometheus, OpenTelemetry, and Grafana.

## Prerequisites

- Local Kubernetes cluster running.
- Helm installed.
- Docker image build capability for local cluster.
- Enough local resources for minimal profile (recommended baseline: 4 CPU, 8 GB RAM available to cluster).

## 1) Prepare local input/output paths

1. Create input and output directories for the sample pipeline.
2. Place a small test input file in the input directory.
3. Ensure output directory is writable by the pipeline runtime.

## 2) Deploy monitoring stack with minimal profile

1. Navigate to `ai-monitor-system/deploy/helm`.
2. Install chart using local/minimal values profile.
3. Wait until monitoring components are in Ready state.

Expected result:

- Grafana reachable.
- Metrics endpoint scraped by Prometheus.
- OTel collector accepting telemetry.
- OpenLineage backend reachable for lineage events.

## 3) Run the sample pipeline

1. Trigger pipeline with `input_path` and `output_path` configured for local files.
2. Execute one successful run.
3. Execute one intentionally failing run (e.g., invalid input path).

Expected result:

- Output file generated for success run.
- Failure run emits explicit failure category and message.

## 4) Verify observability outputs

1. In Grafana, verify run status panels reflect success/failure runs.
2. In metrics view, verify run counters and duration metrics.
3. In lineage view, verify source-to-target dataset mapping for run IDs.
4. In tracing view, verify spans include `run_id` and pipeline metadata.
5. Verify alert fires for failed run and can be resolved.

## 5) Run smoke validation

1. Execute smoke validation script from `ai-monitor-system/deploy/scripts`.
2. Confirm checks for:

   - lifecycle event emission
   - metrics visibility
   - trace/lineage correlation
   - alert creation for failure case

## Troubleshooting Notes

- If telemetry is missing, validate run IDs are consistently attached across metrics, traces, and lineage.
- If local cluster is resource-constrained, reduce non-critical dashboard refresh frequency and keep single replicas.
- If alerting is noisy, tune thresholds in local profile while preserving failure-detection guarantees.
