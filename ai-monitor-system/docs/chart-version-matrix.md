# Chart Version Matrix

Pinned chart versions for the Option B deployment baseline.

| Component | Chart | Repository | Version | Notes |
| --- | --- | --- | --- | --- |
| Prometheus | `prometheus` (alias `upstream-prometheus`) | `https://prometheus-community.github.io/helm-charts` | `25.27.0` | Primary metrics collection backend |
| Grafana | `grafana` (alias `upstream-grafana`) | `https://grafana.github.io/helm-charts` | `8.5.1` | Dashboard and visualization surface |
| OpenTelemetry Collector | `opentelemetry-collector` (alias `upstream-otel-collector`) | `https://open-telemetry.github.io/opentelemetry-helm-charts` | `0.78.0` | Metrics and trace ingestion pipeline |
| OpenLineage backend | `marquez` (alias `upstream-marquez`) | `https://marquezproject.github.io/marquez/` | `0.43.0` | OpenLineage-compatible backend for lineage event querying |

## Update Policy

- Keep chart versions pinned in `deploy/helm/Chart.yaml` and synchronized with this file.
- Update one component at a time and rerun contract/integration/smoke checks.
- Record compatibility notes when changing major versions.
