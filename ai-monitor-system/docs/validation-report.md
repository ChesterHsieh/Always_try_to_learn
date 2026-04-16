# Validation Report

- Helm chart composition updated for Option B baseline.
- Upstream chart version pins documented in `docs/chart-version-matrix.md`.
- Mandatory stack configs present: OpenLineage, Prometheus, OpenTelemetry Collector, Grafana.
- Contract, integration, and smoke tests cover US1/US2/US3 checks.
- Local profile values file present: `deploy/helm/values.local-minimal.yaml`.
