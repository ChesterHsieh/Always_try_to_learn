# Technical Steering

## Stack Baseline

- Runtime language: Python 3.11 for pipeline and helpers.
- Data engine: PySpark with OpenLineage Spark listener integration.
- Deployment target: local Kubernetes cluster.
- Packaging/orchestration: Helm chart with upstream chart dependencies.

## Observability Architecture Principles

- Treat observability as first-class system behavior, not an afterthought.
- Use the approved stack consistently: OpenLineage backend, Prometheus, OpenTelemetry Collector, and Grafana.
- Prefer upstream Helm charts for core observability components and keep project-owned templates minimal and integration-focused.
- Keep run correlation explicit (for example, shared run identity across lifecycle metrics, traces, and lineage).

## Configuration Conventions

- Use `deploy/helm/values.yaml` as the canonical baseline and environment-specific override files for local-minimal profiles.
- Keep chart versions pinned and track upgrades intentionally.
- Default to namespace-scoped deployment with deterministic release naming and repeatable bootstrap scripts.

## Quality and Validation Standards

- Tests are layered: contract, integration, and smoke.
- Coverage checks must verify required observability components are deployed and reachable.
- Documentation and runbooks must stay aligned with deployment scripts and chart behavior.

## Operational Constraints

- Monitoring signal freshness should support near-real-time triage.
- Local profile should remain resource-conscious while preserving full-stack behavior.
- Avoid introducing custom platform services when configuration-driven integration is sufficient.
