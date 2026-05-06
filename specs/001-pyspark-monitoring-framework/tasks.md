# Tasks: PySpark Monitoring Framework (Option B)

**Input**: Design documents from `/specs/001-pyspark-monitoring-framework/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/monitoring-contract.md`, `quickstart.md`

**Tests**: Test tasks are required for behavior changes and operational verification (contract, integration, smoke, and regression).

**Organization**: Tasks are grouped by user story and aligned to Option B migration path (Prometheus/Grafana -> OTel Collector -> OpenLineage backend).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (`[US1]`, `[US2]`, `[US3]`)
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare chart composition baseline and migration scaffolding.

- [x] T001 Validate project scaffolding and Python toolchain in `ai-monitor-system/pyproject.toml`, `ai-monitor-system/ruff.toml`, and `ai-monitor-system/pytest.ini`
- [x] T002 Define Option B chart-composition layout in `ai-monitor-system/deploy/helm/Chart.yaml` and `ai-monitor-system/deploy/helm/values.yaml`
- [x] T003 [P] Add local-minimal override file for upstream releases in `ai-monitor-system/deploy/helm/values.local-minimal.yaml`
- [x] T004 [P] Add pinned chart version registry document in `ai-monitor-system/docs/chart-version-matrix.md`
- [x] T005 Update bootstrap entrypoint for staged upstream installs in `ai-monitor-system/deploy/scripts/bootstrap-local.sh`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared telemetry contracts, run identity, and migration-safe deployment wiring.

**⚠️ CRITICAL**: No user story work starts until this phase completes.

- [x] T006 Implement strict run context validation and lifecycle schema in `ai-monitor-system/pipeline/run_context.py`
- [x] T007 [P] Harden local file I/O checks for deterministic failure categories in `ai-monitor-system/utils/io_adapter.py` _(post-restructure path; originally `pipeline/io_adapter.py`)_
- [x] T008 [P] Implement correlation-first telemetry envelope helpers in `ai-monitor-system/telemetry/telemetry.py` _(post-restructure path; originally `pipeline/telemetry.py`)_
- [x] T009 [P] Add contract fixtures for lifecycle/alert/lineage payloads in `ai-monitor-system/tests/contract/fixtures/monitoring_payloads.json`
- [x] T010 Create project-owned integration bridge templates in `ai-monitor-system/deploy/helm/templates/openlineage-configmap.yaml` and `ai-monitor-system/deploy/helm/templates/spark-defaults-configmap.yaml`
- [x] T011 Add migration-safe namespace and pipeline job templates in `ai-monitor-system/deploy/helm/templates/namespace.yaml` and `ai-monitor-system/deploy/helm/templates/pipeline-job.yaml`
- [x] T012 Add coverage and smoke script skeletons in `ai-monitor-system/deploy/scripts/check-monitoring-coverage.sh` and `ai-monitor-system/deploy/scripts/run-smoke-test.sh`

**Checkpoint**: Foundation ready for user story delivery and upstream chart rollout.

---

## Phase 3: User Story 1 - Detect Pipeline Failures Fast (Priority: P1) 🎯 MVP

**Goal**: Operators detect failed runs quickly with actionable dashboard and alert context.

**Independent Test**: Deploy with Option B stack and confirm one success + one failure run produce status transitions and failure alerts within 2 minutes.

### Tests for User Story 1 (REQUIRED)

- [x] T013 [P] [US1] Create contract test for run lifecycle payload fields in `ai-monitor-system/tests/contract/test_run_lifecycle_contract.py`
- [x] T014 [P] [US1] Create integration test for run status transition timing in `ai-monitor-system/tests/integration/test_run_status_flow.py`
- [x] T015 [P] [US1] Create integration test for failure alert payload semantics in `ai-monitor-system/tests/integration/test_failure_alerts.py`
- [x] T016 [P] [US1] Create smoke test for failed-run dashboard/alert visibility SLA in `ai-monitor-system/tests/smoke/test_us1_failure_detection.py`

### Implementation for User Story 1

- [x] T017 [US1] Implement runtime lifecycle emission path in `ai-monitor-system/pipeline/job.py` and `ai-monitor-system/telemetry/telemetry.py` _(post-restructure path; originally `pipeline/telemetry.py`)_
- [x] T018 [US1] Implement deterministic failure categorization and message mapping in `ai-monitor-system/pipeline/failure_classifier.py`
- [x] T019 [US1] Migrate Prometheus to upstream chart values/overrides in `ai-monitor-system/deploy/helm/values.yaml` and `ai-monitor-system/deploy/helm/values.local-minimal.yaml`
- [x] T020 [US1] Migrate Grafana to upstream chart values and datasource/dashboard provisioning in `ai-monitor-system/deploy/helm/values.yaml` and `ai-monitor-system/monitoring/grafana/datasources.yaml`
- [x] T021 [US1] Expand operator health dashboard for run triage workflow in `ai-monitor-system/monitoring/dashboards/pipeline-health.json`

**Checkpoint**: US1 works independently as MVP on Option B baseline.

---

## Phase 4: User Story 2 - Trace Data Lineage and Run Context (Priority: P2)

**Goal**: Engineers correlate lineage and traces to each run through shared identifiers.

**Independent Test**: Run success/failure cases and verify lineage + trace + metrics share same `run_id` and can be cross-queried.

### Tests for User Story 2 (REQUIRED)

- [x] T022 [P] [US2] Create contract test for lineage required attributes in `ai-monitor-system/tests/contract/test_lineage_contract.py`
- [x] T023 [P] [US2] Create integration test for run-to-lineage correlation in `ai-monitor-system/tests/integration/test_lineage_correlation.py`
- [x] T024 [P] [US2] Create integration test for trace attribute completeness in `ai-monitor-system/tests/integration/test_trace_attributes.py`
- [x] T025 [P] [US2] Create smoke test for lineage/trace correlation path in `ai-monitor-system/tests/smoke/test_us2_lineage_trace.py`

### Implementation for User Story 2

- [x] T026 [US2] Integrate OpenLineage Spark listener runtime wiring in `ai-monitor-system/pipeline/job.py` and `ai-monitor-system/deploy/scripts/run-pipeline.sh`
- [x] T027 [US2] Implement trace span attributes and run metadata propagation in `ai-monitor-system/telemetry/tracing.py` _(post-restructure path; originally `pipeline/tracing.py`)_
- [x] T028 [US2] Add cross-signal correlation fields in `ai-monitor-system/telemetry/telemetry.py` and `ai-monitor-system/telemetry/lineage.py` _(post-restructure paths; originally under `pipeline/`)_
- [x] T029 [US2] Migrate OTel Collector to upstream chart values/overrides in `ai-monitor-system/deploy/helm/values.yaml` and `ai-monitor-system/deploy/helm/values.local-minimal.yaml`
- [x] T030 [US2] Expand lineage-focused dashboard panels for root-cause flow in `ai-monitor-system/monitoring/dashboards/lineage-view.json`

**Checkpoint**: US1 and US2 independently functional with verified correlation.

---

## Phase 5: User Story 3 - Confirm Monitoring Coverage Standards (Priority: P3)

**Goal**: Engineering leads verify standardized stack coverage and onboarding readiness for Option B.

**Independent Test**: Fresh local deployment passes coverage checks for OpenLineage, Prometheus, OTel Collector, and Grafana with no manual patching.

### Tests for User Story 3 (REQUIRED)

- [x] T031 [P] [US3] Create contract test for required stack coverage checks in `ai-monitor-system/tests/contract/test_monitoring_coverage_contract.py`
- [x] T032 [P] [US3] Create integration test for local profile readiness and chart health in `ai-monitor-system/tests/integration/test_local_profile_readiness.py`
- [x] T033 [P] [US3] Create integration test for telemetry freshness warning rule behavior in `ai-monitor-system/tests/integration/test_telemetry_freshness_warning.py`
- [x] T034 [P] [US3] Create smoke test for full-stack coverage and onboarding acceptance in `ai-monitor-system/tests/smoke/test_us3_monitoring_coverage.py`

### Implementation for User Story 3

- [x] T035 [US3] Implement OpenLineage backend upstream chart integration and pinned versions in `ai-monitor-system/deploy/helm/Chart.yaml`, `ai-monitor-system/deploy/helm/values.yaml`, and `ai-monitor-system/docs/chart-version-matrix.md`
- [x] T036 [US3] Implement stack coverage checker against deployed services in `ai-monitor-system/deploy/scripts/check-monitoring-coverage.sh`
- [x] T037 [US3] Standardize stack health and failure rules in `ai-monitor-system/monitoring/alerts/stack-health-rules.yaml` and `ai-monitor-system/monitoring/alerts/pipeline-failure-rules.yaml`
- [x] T038 [US3] Update onboarding runbook for Option B bootstrap and validation in `ai-monitor-system/docs/onboarding-monitoring.md` and `ai-monitor-system/docs/runbook.md`

**Checkpoint**: All user stories are independently testable and meet standardized coverage goals.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening, migration validation, and release-ready documentation.

- [x] T039 [P] Replace placeholder smoke/e2e tests with executable assertions in `ai-monitor-system/tests/smoke/test_end_to_end_local_cluster.py`
- [x] T040 Validate quickstart flow and capture expected outputs in `ai-monitor-system/docs/validation-report.md`
- [x] T041 [P] Align feature docs with final Option B architecture in `specs/001-pyspark-monitoring-framework/quickstart.md` and `specs/001-pyspark-monitoring-framework/gap-analysis.md`
- [x] T042 Add OpenLineage Spark and backend compatibility guidance in `ai-monitor-system/docs/openlineage-spark-config.md`
- [x] T043 Add migration rollback checklist and troubleshooting matrix in `ai-monitor-system/docs/runbook.md`
- [x] T044 Execute full verification suite and record pass criteria in `ai-monitor-system/README.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user story phases.
- **Phase 3 (US1)**: Depends on Phase 2; delivers MVP and Option B first migration slice (Prometheus/Grafana).
- **Phase 4 (US2)**: Depends on Phase 2 and US1 migration baseline; adds OTel Collector + correlation completion.
- **Phase 5 (US3)**: Depends on Phase 2 and benefits from US1/US2 outputs; finalizes OpenLineage backend and coverage standardization.
- **Phase 6 (Polish)**: Depends on completion of targeted user stories.

### User Story Dependencies

- **US1 (P1)**: No dependency on other stories after foundational phase.
- **US2 (P2)**: Builds on US1 runtime lifecycle and dashboard semantics for correlation.
- **US3 (P3)**: Uses US1/US2 outputs to verify full-stack standards and onboarding readiness.

### Within Each User Story

- Write tests first and confirm failure before implementation.
- Implement core instrumentation before dashboard/alert wiring.
- Complete smoke checks before story sign-off.

### Parallel Opportunities

- Setup tasks marked `[P]`: T003, T004.
- Foundational tasks marked `[P]`: T007, T008, T009.
- US1 tests marked `[P]`: T013, T014, T015, T016.
- US2 tests marked `[P]`: T022, T023, T024, T025.
- US3 tests marked `[P]`: T031, T032, T033, T034.
- Polish tasks marked `[P]`: T039, T041.

---

## Parallel Example: User Story 1

```bash
# Run US1 validation tasks in parallel:
Task: "T013 [US1] Contract test in ai-monitor-system/tests/contract/test_run_lifecycle_contract.py"
Task: "T014 [US1] Integration test in ai-monitor-system/tests/integration/test_run_status_flow.py"
Task: "T015 [US1] Integration test in ai-monitor-system/tests/integration/test_failure_alerts.py"
Task: "T016 [US1] Smoke test in ai-monitor-system/tests/smoke/test_us1_failure_detection.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Deliver Phase 3 (US1) with Prometheus/Grafana migration.
3. Validate failed-run detection and alert workflow in local cluster within SLA.
4. Demo and baseline operational feedback.

### Incremental Delivery

1. Add US2 for lineage + trace correlation and OTel Collector migration.
2. Add US3 for OpenLineage backend standardization and onboarding readiness.
3. Finish with Polish phase for docs and end-to-end validation artifacts.

### Parallel Team Strategy

1. Team aligns on Setup + Foundational tasks.
2. After foundation:
   - Engineer A: US1 (failure detection + Prom/Grafana)
   - Engineer B: US2 (correlation + OTel)
   - Engineer C: US3 (coverage + OpenLineage backend)
3. Merge at story checkpoints with smoke validation gates.

---

## Notes

- All tasks use unchecked checkboxes for fresh execution tracking.
- `[P]` tasks are selected to avoid same-file conflicts.
- US1 remains recommended MVP scope.
- Option B migration order is intentional: Prometheus/Grafana -> OTel Collector -> OpenLineage backend.
- Required stack remains OpenLineage, Prometheus, OpenTelemetry, and Grafana with pinned chart versions.
