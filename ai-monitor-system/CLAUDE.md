# ai-monitor-system — Agent Operating Notes

Local PySpark + Kubernetes monitoring reference. This file gives an LLM agent
the minimum context to run a self-driven dev cycle without re-deriving layout
or commands. For human-facing setup, see [README.md](README.md).

## Stack at a glance

- Pipeline: PySpark job in [pipeline/job.py](pipeline/job.py) with telemetry helpers
  ([metrics.py](pipeline/metrics.py), [tracing.py](pipeline/tracing.py),
  [otel_setup.py](pipeline/otel_setup.py), [lineage_emitter.py](pipeline/lineage_emitter.py),
  [failure_classifier.py](pipeline/failure_classifier.py)).
- Deploy: Helm chart in [deploy/helm/](deploy/helm/) — upstream observability
  charts pulled in via `Chart.yaml`; pipeline job template in
  [pipeline-job.yaml](deploy/helm/templates/pipeline-job.yaml).
- Values: [values.yaml](deploy/helm/values.yaml) (full) and
  [values.local-minimal.yaml](deploy/helm/values.local-minimal.yaml) (local).
- Monitoring config: [monitoring/](monitoring/) — Grafana dashboards + Prom alert rules.
- Tests: contract → integration → smoke (see "Test layers" below).

## Common operations (do these via Bash)

| Goal | Command |
|---|---|
| Lint | `cd ai-monitor-system && uv run ruff check .` |
| Format | `cd ai-monitor-system && uv run ruff format .` |
| Quick check (no cluster) | `cd ai-monitor-system && uv run pytest -q tests/contract` |
| Full unit/integration | `cd ai-monitor-system && uv run pytest -q tests/contract tests/integration` |
| Smoke (requires cluster) | `./deploy/scripts/run-smoke-test.sh` |
| Helm lint | `helm lint deploy/helm -f deploy/helm/values.local-minimal.yaml` |
| Helm template render | `helm template monitor deploy/helm -f deploy/helm/values.local-minimal.yaml` |
| Bring infra up | `./deploy/scripts/bootstrap-local.sh` |
| Run pipeline once | `./deploy/scripts/run-pipeline.sh` |
| Coverage check | `./deploy/scripts/check-monitoring-coverage.sh` |
| Cluster sanity | `timeout 6 kubectl get nodes --request-timeout=2s` |

Namespace is always `ai-monitor-system`; Helm release defaults to `monitor`.

## Test layers (what to run when)

- **contract/** — pure Python, no cluster, no I/O. Run on every code change as
  the fast inner loop. < 5s.
- **integration/** — exercises telemetry/lineage wiring with stubs. No live
  cluster needed for most cases. Run before declaring a code change "done".
- **smoke/** — end-to-end against a real local cluster (kind/minikube/Docker
  Desktop K8s). Requires `bootstrap-local.sh` to have succeeded. Slow; run
  before committing meaningful pipeline or Helm changes, not in tight loops.

## Self-verification cycle (preferred)

For Python-only changes:
1. `uv run ruff format . && uv run ruff check . --fix`
2. `uv run pytest -q tests/contract` (must be green)
3. `uv run pytest -q tests/integration` (must be green)
4. Only run smoke if you touched pipeline I/O, Helm, or scripts.

For Helm/values changes:
1. `helm lint deploy/helm -f deploy/helm/values.local-minimal.yaml`
2. `helm template monitor deploy/helm -f deploy/helm/values.local-minimal.yaml | head -200`
   (sanity-check rendered manifests)
3. If the cluster is up: `./deploy/scripts/run-smoke-test.sh`

## Destructive actions — always confirm first

These are denied by harness policy and require explicit user approval:
- `./deploy/scripts/nuke-local.sh` (tears down cluster state)
- `kubectl delete *`, `helm uninstall *`, `helm delete *`
- `docker rm`, `docker rmi`, `docker system prune`

If you believe one is needed, ask the user before invoking.

## Useful artifacts

- Local data fixtures: `.local-data/` (gitignored; safe to inspect, not to commit)
- Helm rendered outputs from prior runs: `deploy/helm/tmpcharts-*/` (transient)
- Runbook: [docs/runbook.md](docs/runbook.md)
- Onboarding: [docs/onboarding-monitoring.md](docs/onboarding-monitoring.md)
- Validation report: [docs/validation-report.md](docs/validation-report.md)

## Failure triage shortcuts

- Pipeline pod not starting → `kubectl describe pod -n ai-monitor-system -l job-name=pyspark-pipeline`
- Image not found → rebuild via `bootstrap-local.sh` (it builds
  `local/ai-monitor-pyspark:latest` before deploy); `imagePullPolicy: IfNotPresent`.
- Missing telemetry → check OTel collector pod logs and
  [pipeline/otel_setup.py](pipeline/otel_setup.py) endpoint config.
- Failure-classifier behavior → contract spec lives in
  [tests/contract/test_failure_classifier_contract.py](tests/contract/test_failure_classifier_contract.py).
