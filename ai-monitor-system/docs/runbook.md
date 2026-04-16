# Runbook

- Bootstrap stack: `deploy/scripts/bootstrap-local.sh`
- Run smoke tests: `deploy/scripts/run-smoke-test.sh`
- Check stack config presence: `deploy/scripts/check-monitoring-coverage.sh`
- Validate Option B chart versions: `docs/chart-version-matrix.md`
- Verify alerts:
  - `PipelineRunFailed` for run failures
  - `TelemetryFreshnessHigh` for stale telemetry

## Rollback Checklist

1. Pin back to previously known-good chart versions in `deploy/helm/Chart.yaml`.
2. Re-apply previous local values profile in `deploy/helm/values.local-minimal.yaml`.
3. Redeploy via `deploy/scripts/bootstrap-local.sh`.
4. Run `deploy/scripts/check-monitoring-coverage.sh`.
5. Re-run smoke suite with `deploy/scripts/run-smoke-test.sh`.

## Troubleshooting Matrix

| Symptom | Likely Cause | First Check |
| --- | --- | --- |
| No lineage events | Marquez endpoint mismatch | `OPENLINEAGE_URL` in `openlineage-config` ConfigMap |
| Missing traces | OTel endpoint mismatch | OTel collector service and exporter settings |
| No failure alerts | Rule/config not loaded | `monitoring/alerts/pipeline-failure-rules.yaml` and Prometheus rules |
| Dashboards empty | Datasource or scrape issue | Grafana datasource config and Prometheus targets |
