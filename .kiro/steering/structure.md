# Structure Steering

## Repository Organization Pattern

- Product implementation lives under `ai-monitor-system/`.
- Feature intent, requirements, and execution planning live under `specs/`.
- Steering memory lives under `.kiro/steering/` and should capture durable decision patterns.

## Implementation Boundaries

- `pipeline/`: pipeline runtime behavior and telemetry/lineage/tracing helpers.
- `deploy/helm/`: chart composition, upstream dependencies, values, and project-owned glue templates.
- `deploy/scripts/`: operational entrypoints (bootstrap, run, smoke, coverage checks).
- `monitoring/`: dashboards, alert rules, and collector/prometheus provisioning artifacts.
- `tests/`: contract, integration, and smoke suites mapped to monitoring outcomes.
- `docs/`: onboarding, runbook, compatibility, and validation guidance.

## Design and Change Conventions

- Keep pipeline logic intentionally simple; complexity belongs in observability and operability.
- Prefer configuration-first changes in Helm values/templates over adding custom services.
- Preserve clear ownership boundaries between runtime code, deployment wiring, and monitoring assets.
- Add tests in the same layer as the behavior being changed (contract for schema guarantees, integration for cross-component behavior, smoke for end-to-end readiness).

## Steering Granularity Rule

- Document patterns and decision rules, not exhaustive file inventories.
- If new work follows established patterns, steering should not require updates.
- Update steering only when architecture, standards, or organization conventions change.
