# Onboarding Monitoring

1. Bootstrap local stack with `deploy/scripts/bootstrap-local.sh`.
2. Confirm Option B components are healthy:
   - Prometheus service in namespace `ai-monitor-system`
   - Grafana service in namespace `ai-monitor-system`
   - OTel Collector service in namespace `ai-monitor-system`
3. Run `deploy/scripts/check-monitoring-coverage.sh` to verify baseline wiring.
4. Run `deploy/scripts/run-pipeline.sh` for one success case and one failure case.
5. Verify in observability tooling:
   - failed-run alert `PipelineRunFailed` contains run context
   - freshness warning `TelemetryFreshnessHigh` appears when threshold exceeded
   - dashboards show run state and lineage correlation views
