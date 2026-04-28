# AI Monitor System

Monitoring-first reference project for a simple local-file-to-local-file PySpark pipeline on Kubernetes.

## Required Stack

- OpenLineage
- Marquez
- Prometheus
- OpenTelemetry Collector
- Grafana
- Helm (for deployment)

## Project Layout

- `pipeline/`: PySpark job and telemetry helpers
- `deploy/helm/`: Helm chart, values, and templates
- `deploy/scripts/`: bootstrap, smoke test, and coverage scripts
- `monitoring/`: Prometheus, OTel, Grafana, and alert configs
- `tests/`: contract, integration, and smoke tests
- `docs/`: onboarding, runbook, and validation notes

## Local Quick Start

1. Ensure local Kubernetes is running.
2. Setup infra (single command from project root):
   - `./deploy/scripts/bootstrap-local.sh`
   - This command also builds `local/ai-monitor-pyspark:latest` locally before Helm deploy.
   - Verify infra pods/services: `kubectl get pods,svc -n ai-monitor-system`
3. Run pipeline with OpenLineage Spark listener settings:
   - `./deploy/scripts/run-pipeline.sh`
4. Run smoke tests:
   - `./deploy/scripts/run-smoke-test.sh`

## Access UIs Locally

- Marquez UI (lineage): `kubectl port-forward -n ai-monitor-system svc/marquez-web 3001:9444` then open `http://localhost:3001`
- Marquez API (optional check): `kubectl port-forward -n ai-monitor-system svc/marquez 9555:9555`
- Grafana: `kubectl port-forward -n ai-monitor-system svc/grafana 3000:80` then open `http://localhost:3000`
- Prometheus: `kubectl port-forward -n ai-monitor-system svc/prometheus-server 9090:80` then open `http://localhost:9090`

## Test

From project root:

- `pytest -q tests/contract tests/integration tests/smoke`
- `./deploy/scripts/check-monitoring-coverage.sh` (requires local cluster resources)

### Pass Criteria

- Contract, integration, and smoke suites pass with no failures.
- Coverage checker confirms Prometheus, Grafana, and OTel collector services are discoverable.
- Pipeline success and failure paths generate expected lifecycle and alert artifacts.

## Deployment Namespace

- `./deploy/scripts/bootstrap-local.sh` always deploys to `ai-monitor-system`.
- Helm release name defaults to `monitor` (override with `RELEASE_NAME=...`).
- The script uses Helm `--create-namespace`, so it creates the namespace if missing and reuses it if it already exists.
- The namespace is forced in both Helm release scope (`--namespace`) and chart values (`--set global.namespace=ai-monitor-system`).
- The pipeline job uses `imagePullPolicy: IfNotPresent` to reuse the local image instead of forcing registry pulls.

## Notes

- Monitoring tool integration is configuration-first (Helm/templates/rules), not custom alerting service code.
- Infra setup now deploys upstream chart-based observability components using local-minimal overrides.
- Spark listener keys follow the OpenLineage Spark guide (`spark.jars.packages`, `spark.extraListeners`, and `spark.openlineage.*`).
- If image pull issues continue, run: `kubectl describe pod -n ai-monitor-system -l job-name=pyspark-pipeline`.
