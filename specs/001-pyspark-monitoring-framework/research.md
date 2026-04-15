# Research: PySpark Monitoring Framework

## Decision 1: Keep pipeline scope to local-file input/output only

- **Decision**: Implement the reference pipeline as local-file-to-local-file processing only.
- **Rationale**: User explicitly requested a simple pipeline and highlighted monitoring as the key design concern.
- **Alternatives considered**:

  - Object storage integration (rejected for v1 due to added complexity and operational overhead)
  - Message queue/stream ingestion (rejected for v1 because it shifts focus away from observability baseline)

## Decision 2: Use Helm as the default deployment interface

- **Decision**: Package all deployable components with Helm and provide a minimal local-cluster values profile.
- **Rationale**: Helm gives repeatable install/upgrade/uninstall workflows and configuration centralization, which supports local experimentation.
- **Alternatives considered**:

  - Raw manifests only (rejected due to lower repeatability and weaker parameter management)
  - Kustomize-only layering (rejected for v1 to keep one primary deployment workflow)

## Decision 3: Establish a low-resource local cluster profile

- **Decision**: Define conservative default resource requests/limits and reduced component replicas for monitoring services.
- **Rationale**: The framework must run on developer-grade local clusters with constrained CPU/memory.
- **Alternatives considered**:

  - Production-like default sizing (rejected because it fails local resource constraints)
  - Per-team manual tuning only (rejected because onboarding would be inconsistent and error-prone)

## Decision 4: Correlate OpenLineage, metrics, and traces around run identity

- **Decision**: Use a shared run identifier and consistent labels/tags across lineage, metrics, and traces.
- **Rationale**: Correlation is required for fast triage and aligns with monitoring-first outcomes in the feature spec.
- **Alternatives considered**:

  - Independent telemetry channels with no common IDs (rejected due to investigation friction)
  - Log-only correlation (rejected due to weaker dashboard and alert integration)

## Decision 5: Focus dashboards on operator triage workflow

- **Decision**: Build a minimal dashboard set around run state, failure counts, recent failures, and telemetry freshness.
- **Rationale**: Operators need immediate actionable visibility, not broad exploratory analytics in v1.
- **Alternatives considered**:

  - Comprehensive analytics dashboards (rejected as non-critical for initial monitoring framework)
  - Trace UI only without dashboards (rejected because on-call workflow benefits from summary views)

## Decision 6: Validate with lightweight smoke tests in local Kubernetes

- **Decision**: Include a smoke test path that deploys stack, runs one success and one failure pipeline case, and verifies visibility in observability outputs.
- **Rationale**: This provides practical delivery evidence while keeping validation fast for local development.
- **Alternatives considered**:

  - Unit-test-only validation (rejected because it misses integration behavior)
  - Heavy load/performance test suite in v1 (rejected as out of scope for local-focused first release)
