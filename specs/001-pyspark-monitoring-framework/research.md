# Research: PySpark Monitoring Framework (Option B)

## Decision 1: Rebase observability components to upstream Helm charts

- **Decision**: Use upstream Helm charts as primary deployment mechanism for Prometheus, Grafana, OpenTelemetry Collector, and OpenLineage backend.
- **Rationale**: Standardized charts reduce custom YAML maintenance, improve upgrade path, and align with operational simplicity goals.
- **Alternatives considered**:
  - Keep fully custom manifests in repo (rejected due to long-term maintenance overhead and drift risk)
  - Use only managed SaaS observability (rejected because local-cluster-first scope requires self-contained stack)

## Decision 2: Keep project-owned chart scope to pipeline and integration overlays

- **Decision**: Retain local chart/templates only for pipeline job execution, Spark/OpenLineage runtime wiring, and required project-specific config bridges.
- **Rationale**: Preserves control for pipeline semantics while offloading generic observability infrastructure lifecycle.
- **Alternatives considered**:
  - Migrate everything to upstream charts with no local wrappers (rejected because pipeline runtime wiring still needs project ownership)
  - Keep all existing local templates unchanged (rejected because it defeats Option B maintenance goals)

## Decision 3: Preserve local-file pipeline scope

- **Decision**: Keep v1 data path as local file input to local file output.
- **Rationale**: User requirement explicitly prioritizes monitoring framework behavior over data-pipeline complexity.
- **Alternatives considered**:
  - Add object storage connector now (rejected due to scope expansion)
  - Add streaming source support now (rejected as out of v1 assumptions)

## Decision 4: Adopt strict run identity correlation contract

- **Decision**: Use `run_id` as mandatory correlation key across metrics, traces, lineage records, and run-scoped alerts.
- **Rationale**: End-to-end triage speed depends on deterministic cross-signal joins.
- **Alternatives considered**:
  - Best-effort correlation by timestamp/job name only (rejected due to ambiguity)
  - Separate IDs per telemetry domain (rejected due to investigation friction)

## Decision 5: Local resource profile remains a first-class deployment contract

- **Decision**: Define and enforce low-resource profile values for all upstream components (single replica defaults, bounded CPU/memory requests/limits).
- **Rationale**: Feature must run reliably on developer-grade clusters for onboarding and smoke validation.
- **Alternatives considered**:
  - Production-sized defaults (rejected because it breaks local usability)
  - Unbounded resources with user tuning (rejected due to inconsistent onboarding outcomes)

## Decision 6: Test strategy upgraded from file-existence checks to behavior checks

- **Decision**: Convert placeholder tests and shallow checks into behavior-driven contract/integration/smoke validations against deployed stack.
- **Rationale**: Existing scaffolding is insufficient evidence for FR/NFR success criteria.
- **Alternatives considered**:
  - Keep placeholder smoke tests for speed (rejected as non-compliant with testing constitution)
  - Manual verification only (rejected because repeatable release gating is required)

## Decision 7: Migration path is phased, not big-bang

- **Decision**: Migrate by component groups: Prometheus/Grafana first, then OTel Collector, then OpenLineage backend integration stabilization.
- **Rationale**: Limits blast radius and keeps rollback simple.
- **Alternatives considered**:
  - One-shot chart migration (rejected due to higher outage/debug risk)
  - No migration sequencing (rejected because ownership boundaries become unclear)
