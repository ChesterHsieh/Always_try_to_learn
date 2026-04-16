# Gap Analysis: 001-pyspark-monitoring-framework

## 1) Analysis Summary

- The current `ai-monitor-system` implementation delivers a usable foundation (Helm chart, pipeline skeleton, observability config files, docs, and test layout) aligned with the feature direction.
- The largest gap is **runtime observability completeness**: many monitoring artifacts exist as static configs/tests, but end-to-end signal production and correlation are mostly not validated with real backend integrations.
- OpenLineage integration is partially configured (Spark listener + env/config), but the referenced backend endpoint is not provisioned in-cluster by current manifests.
- Alerting/dashboard coverage is present but minimal and not yet tied to robust run lifecycle semantics (running/succeeded/failed + failure context contract).
- Multiple viable paths exist: incrementally harden current scaffold, replace with upstream chart composition, or use a hybrid approach to preserve speed while reducing custom ops burden.

## 2) Document Status

- Analysis method: requirements-to-code traceability review across `spec.md`, `plan.md`, `tasks.md`, contract docs, pipeline code, Helm/templates, scripts, and tests.
- Steering context note: `.kiro/steering/` was not found in this workspace, so project-memory enrichment from steering docs could not be applied.
- Rules file note: `.kiro/settings/rules/gap-analysis.md` was not found; analysis follows the command’s stated framework and success criteria directly.
- Output language: English fallback used because `spec.json` was not found.

## 3) Requirement Coverage Matrix (Current State)

### FR-001 Run-level monitoring visibility
- **Status**: Partial
- **Evidence**: `pipeline/job.py` returns a lifecycle payload; Prometheus/Grafana configs exist.
- **Gap**: No demonstrated emission path from job runtime into metrics backend; lifecycle state transitions are not fully modeled (mostly terminal success payload).

### FR-002 Actionable failure signals with context
- **Status**: Partial
- **Evidence**: `failure_classifier.py`, alert rule scaffold in `monitoring/alerts/pipeline-failure-rules.yaml`.
- **Gap**: Failure category/message context is not propagated through full telemetry/alert payload chain.

### FR-003 Lineage linked to runs
- **Status**: Partial
- **Evidence**: OpenLineage Spark listener config and `pipeline/lineage.py` event builder.
- **Gap**: In-cluster lineage backend/service wiring appears incomplete (config points to `marquez-api:5000`, but no clear deployment in current Helm templates).

### FR-004 Operational metrics for health/resource/outcomes
- **Status**: Partial
- **Evidence**: Prometheus config and dashboard JSON placeholders.
- **Gap**: Metric families in contract are not comprehensively produced/validated end-to-end.

### FR-005 Correlated telemetry (metrics, traces, lineage)
- **Status**: Partial
- **Evidence**: `telemetry.py`, `tracing.py`, and lineage helper exist.
- **Gap**: Correlation seems mostly helper-level; no strong runtime proof that shared identifiers flow through all backends.

### FR-006 Dashboard-based visualization
- **Status**: Implemented (basic), not production-hardened
- **Evidence**: `monitoring/dashboards/pipeline-health.json`, `lineage-view.json`.
- **Gap**: Minimal panels; no richer triage context, trend/history, or explicit operator workflow coverage.

### FR-007 Alerting workflows with on-call context
- **Status**: Partial
- **Evidence**: alert rule files exist.
- **Gap**: Alert payload contract fields (e.g., dashboard link, run-scoped context) are not fully represented/validated.

### FR-008 Standardized stack coverage (OL/Prom/OTel/Grafana)
- **Status**: Mostly implemented as configuration baseline
- **Evidence**: Helm values/templates and monitoring config files for all required components.
- **Gap**: Operational connectivity and health checks are still shallow in automation.

### FR-009 Onboarding guidance
- **Status**: Implemented
- **Evidence**: `README.md`, `docs/onboarding-monitoring.md`, scripts for bootstrap/run/smoke.
- **Gap**: Guidance may overstate readiness relative to current runtime validation depth.

### FR-010 Monitoring acceptance checks
- **Status**: Partial
- **Evidence**: Coverage script and test suites exist.
- **Gap**: Several smoke tests are placeholders, reducing acceptance-check confidence.

## 4) Non-Functional Gap View

- **NFR-001 Code Quality**: Good structure and documentation boundaries; implementation depth still uneven.
- **NFR-002 Testing**: Largest gap. Multiple tests are placeholders (`assert True`) and do not exercise required behavior.
- **NFR-003 UX Consistency**: Semantic consistency model exists in spec/contract but is not deeply reflected in dashboard/alert payload design.
- **NFR-004 Performance (<=2 min visibility)**: Config intent exists (`for: 1m/2m`), but no strong evidence of measured end-to-end SLA attainment.

## 5) Key Integration Gaps and Risks

1. **Backend Completeness Risk (High)**  
   OpenLineage target endpoint is configured but backend deployment/dependency chain is unclear in current manifests.

2. **Telemetry Reality Gap (High)**  
   Helper functions return dictionaries, but production telemetry export path (OTLP metrics/traces, scrape targets) is not fully proven.

3. **Validation Confidence Gap (High)**  
   Placeholder smoke tests undercut acceptance criteria and may mask runtime regressions.

4. **Prometheus Scrape Target Fidelity (Medium)**  
   Static target assumptions (e.g., `pyspark-pipeline:8000`) may not hold for ephemeral Job pods without explicit metrics endpoint/service strategy.

5. **Operational Runbook Fidelity (Medium)**  
   Docs describe workflow, but some troubleshooting and success checks may not map to actual deployed signals yet.

## 6) Viable Implementation Approaches

### Option A: Extend Current Scaffold (Incremental Hardening)
- **Approach**: Keep existing custom Helm/templates and Python helpers; add missing runtime telemetry emitters, backend deployment wiring, and realistic smoke tests.
- **Pros**: Lowest disruption, preserves existing structure and docs, fastest short-term closure.
- **Cons**: Higher long-term maintenance burden for custom manifests and observability glue.
- **Best when**: Team prioritizes immediate progress with minimal re-architecture.

### Option B: Rebase to Upstream Observability Charts (Standardization-first)
- **Approach**: Replace most custom monitoring templates with upstream Helm dependencies/charts for Prometheus/Grafana/OTel/OpenLineage backend; keep project-specific overlays.
- **Pros**: Better operational maturity, easier upgrades/security patching, less bespoke YAML drift.
- **Cons**: Migration cost, potential local-cluster resource tuning complexity.
- **Best when**: Medium-term maintainability/governance is prioritized.
- **Current direction applied**: Prometheus, Grafana, OTel Collector, and Marquez(OpenLineage backend) are pinned through chart dependencies with local-minimal overrides.

### Option C: Hybrid (Recommended for Design Phase Evaluation)
- **Approach**: Keep current pipeline/job chart and script ergonomics; selectively adopt upstream components for the riskiest monitoring backends.
- **Pros**: Balances delivery speed and operational robustness.
- **Cons**: Requires careful ownership boundaries and compatibility testing.
- **Best when**: Need pragmatic path from prototype to durable reference framework.

## 7) Research Needed in Design Phase

1. **OpenLineage backend topology for local K8s**  
   Determine minimal deployable stack and connectivity model that still satisfies lineage visibility goals.

2. **PySpark metrics export strategy in Kubernetes Jobs**  
   Decide between direct endpoint exposition, PushGateway-like pattern, or OTel-native metrics export from job code.

3. **Telemetry correlation contract implementation**  
   Define concrete propagation model for `run_id` across metrics, traces, and lineage with verifiable assertions.

4. **Alert payload enrichment model**  
   Specify how run context/dashboard links are generated and attached for on-call triage.

5. **SLA verification harness**  
   Design measurable test flow to prove "visible within 2 minutes" under local-cluster constraints.

## 8) Recommended Next Steps

1. Review and confirm preferred implementation path (A, B, or C).
2. Run `/kiro/spec-design 001-pyspark-monitoring-framework` (or with `-y` if intentionally fast-tracking).
3. In design, prioritize closing High-risk gaps first: backend completeness, telemetry reality, and smoke-test fidelity.
4. Convert accepted gap closures into explicit design decisions and task deltas before further implementation.
