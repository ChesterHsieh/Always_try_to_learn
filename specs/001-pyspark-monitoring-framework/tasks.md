# Tasks: PySpark Monitoring Framework

**Input**: Design documents from `/specs/001-pyspark-monitoring-framework/`  
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/monitoring-contract.md`, `quickstart.md`

**Tests**: Test tasks are required for behavior changes and operational verification (unit, integration, contract, and smoke).

**Organization**: Tasks are grouped by user story so each story can be implemented and tested independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (`[US1]`, `[US2]`, `[US3]`)
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize repository structure and baseline tooling for local-cluster-first development.

- [x] T001 Create project directory scaffold in `ai-monitor-system/` per `plan.md`
- [x] T002 Initialize Python dependencies and package metadata in `ai-monitor-system/pyproject.toml`
- [x] T003 [P] Add lint/format/test configuration in `ai-monitor-system/ruff.toml`
- [x] T004 [P] Add pytest root configuration in `ai-monitor-system/pytest.ini`
- [x] T005 Create Helm chart skeleton in `ai-monitor-system/deploy/helm/` (`Chart.yaml`, `values.yaml`, `templates/`)
- [x] T006 [P] Add local resource profile defaults in `ai-monitor-system/deploy/k8s/local-profile.yaml`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared contracts, run identity model, and deployment/test plumbing required by all stories.

**⚠️ CRITICAL**: No user story work starts until this phase completes.

- [x] T007 Implement shared run context schema and validators in `ai-monitor-system/pipeline/run_context.py`
- [x] T008 [P] Implement local file I/O adapter with safety checks in `ai-monitor-system/pipeline/io_adapter.py`
- [x] T009 [P] Configure OpenTelemetry instrumentation helpers (`run_id`, labels/tags) in `ai-monitor-system/pipeline/telemetry.py`
- [x] T010 [P] Add monitoring contract fixtures for required payload fields in `ai-monitor-system/tests/contract/fixtures/monitoring_payloads.json`
- [x] T011 Configure Helm values schema with mandatory OpenLineage, Prometheus, OpenTelemetry Collector, and Grafana blocks in `ai-monitor-system/deploy/helm/values.yaml`
- [x] T012 Add mandatory observability config templates for OpenTelemetry Collector, Prometheus, and Grafana in `ai-monitor-system/monitoring/otel/collector-config.yaml`, `ai-monitor-system/monitoring/prometheus/prometheus.yml`, and `ai-monitor-system/monitoring/grafana/datasources.yaml`
- [x] T043 Configure OpenLineage transport/backend settings via Helm values and environment templates in `ai-monitor-system/deploy/helm/templates/openlineage-configmap.yaml`
- [x] T044 Add bootstrap and smoke script templates in `ai-monitor-system/deploy/scripts/bootstrap-local.sh` and `ai-monitor-system/deploy/scripts/run-smoke-test.sh`
- [x] T045 Configure Spark OpenLineage listener defaults (reuse official keys) in `ai-monitor-system/deploy/helm/templates/spark-defaults-configmap.yaml` with `spark.jars.packages`, `spark.extraListeners`, `spark.openlineage.transport.url`, `spark.openlineage.transport.type`, and `spark.openlineage.namespace`

**Checkpoint**: Foundation ready for independent user story delivery.

---

## Phase 3: User Story 1 - Detect Pipeline Failures Fast (Priority: P1) 🎯 MVP

**Goal**: Operators can quickly detect failed runs with actionable context in dashboards and alerts.

**Independent Test**: Run one successful and one failed local-file pipeline execution in local Kubernetes and verify status/alert visibility within 2 minutes.

### Tests for User Story 1 (REQUIRED)

- [x] T013 [P] [US1] Add contract test for run lifecycle event payloads in `ai-monitor-system/tests/contract/test_run_lifecycle_contract.py`
- [x] T014 [P] [US1] Add integration test for success/failure run transitions in `ai-monitor-system/tests/integration/test_run_status_flow.py`
- [x] T015 [P] [US1] Add integration test for failure alert emission in `ai-monitor-system/tests/integration/test_failure_alerts.py`

### Implementation for User Story 1

- [x] T016 [US1] Implement simple local-file-to-local-file PySpark job flow in `ai-monitor-system/pipeline/job.py`
- [x] T017 [US1] Implement failure categorization mapping in `ai-monitor-system/pipeline/failure_classifier.py`
- [x] T018 [US1] Emit run lifecycle metrics (`running/succeeded/failed`) in `ai-monitor-system/pipeline/telemetry.py`
- [x] T019 [US1] Configure Prometheus scrape and rule files (tool config only, no custom alerting service code) in `ai-monitor-system/monitoring/prometheus/prometheus.yml` and `ai-monitor-system/monitoring/alerts/pipeline-failure-rules.yaml`
- [x] T020 [US1] Configure Grafana operator run-health dashboard JSON in `ai-monitor-system/monitoring/dashboards/pipeline-health.json`
- [x] T021 [US1] Wire Helm templates for pipeline + monitoring essentials in `ai-monitor-system/deploy/helm/templates/`
- [x] T022 [US1] Implement US1 smoke assertions in `ai-monitor-system/tests/smoke/test_us1_failure_detection.py`

**Checkpoint**: User Story 1 is independently functional and demoable as MVP.

---

## Phase 4: User Story 2 - Trace Data Lineage and Run Context (Priority: P2)

**Goal**: Data engineers can inspect lineage and correlate run context across telemetry signals.

**Independent Test**: Execute one success and one failure run; verify lineage records and trace/metric correlation by shared `run_id`.

### Tests for User Story 2 (REQUIRED)

- [x] T023 [P] [US2] Add contract test for lineage required attributes in `ai-monitor-system/tests/contract/test_lineage_contract.py`
- [x] T024 [P] [US2] Add integration test for run-to-lineage correlation in `ai-monitor-system/tests/integration/test_lineage_correlation.py`
- [x] T025 [P] [US2] Add integration test for trace attribute completeness in `ai-monitor-system/tests/integration/test_trace_attributes.py`

### Implementation for User Story 2

- [x] T026 [US2] Configure PySpark job submission to reuse OpenLineage Spark listener integration in `ai-monitor-system/pipeline/job.py` and `ai-monitor-system/deploy/scripts/run-pipeline.sh` (no custom lineage protocol implementation)
- [x] T027 [US2] Configure OpenTelemetry trace span instrumentation with run metadata in `ai-monitor-system/pipeline/tracing.py`
- [x] T028 [US2] Extend telemetry correlation to include lineage/trace IDs in `ai-monitor-system/pipeline/telemetry.py`
- [x] T029 [US2] Add lineage-focused dashboard panel definitions in `ai-monitor-system/monitoring/dashboards/lineage-view.json`
- [x] T030 [US2] Add US2 smoke assertions for lineage + trace linkage in `ai-monitor-system/tests/smoke/test_us2_lineage_trace.py`

**Checkpoint**: User Stories 1 and 2 both work independently with correlation validated.

---

## Phase 5: User Story 3 - Confirm Monitoring Coverage Standards (Priority: P3)

**Goal**: Engineering leads can verify standardized observability coverage and onboarding readiness.

**Independent Test**: Run readiness checks on local-cluster deployment and confirm OpenLineage, Prometheus, OTel, and Grafana integration coverage.

### Tests for User Story 3 (REQUIRED)

- [x] T031 [P] [US3] Add contract test for required monitoring stack component checks in `ai-monitor-system/tests/contract/test_monitoring_coverage_contract.py`
- [x] T032 [P] [US3] Add integration test for Helm local-profile deployment readiness in `ai-monitor-system/tests/integration/test_local_profile_readiness.py`
- [x] T033 [P] [US3] Add integration test for telemetry freshness warning behavior in `ai-monitor-system/tests/integration/test_telemetry_freshness_warning.py`

### Implementation for User Story 3

- [x] T034 [US3] Implement monitoring coverage checklist runner in `ai-monitor-system/deploy/scripts/check-monitoring-coverage.sh`
- [x] T035 [US3] Configure Helm templates/values for local minimal replicas/resources for OpenLineage, Prometheus, OpenTelemetry Collector, and Grafana in `ai-monitor-system/deploy/helm/templates/` and `ai-monitor-system/deploy/helm/values.yaml`
- [x] T036 [US3] Configure standardized Prometheus rule files for stack health (tool configuration only) in `ai-monitor-system/monitoring/alerts/stack-health-rules.yaml`
- [x] T037 [US3] Add onboarding and readiness guidance in `ai-monitor-system/docs/onboarding-monitoring.md`
- [x] T038 [US3] Implement US3 smoke assertions for full-stack coverage in `ai-monitor-system/tests/smoke/test_us3_monitoring_coverage.py`

**Checkpoint**: All user stories are independently testable and satisfy coverage standards.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening, documentation alignment, and full workflow validation.

- [x] T039 [P] Consolidate runbook and troubleshooting notes in `ai-monitor-system/docs/runbook.md`
- [x] T040 Align `quickstart.md` commands with implemented scripts in `specs/001-pyspark-monitoring-framework/quickstart.md`
- [x] T041 [P] Add end-to-end smoke orchestrator entrypoint in `ai-monitor-system/tests/smoke/test_end_to_end_local_cluster.py`
- [x] T042 Validate full quickstart flow and record expected outputs in `ai-monitor-system/docs/validation-report.md`
- [x] T046 Add OpenLineage Spark configuration reference and version-pin guidance in `ai-monitor-system/docs/openlineage-spark-config.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user story phases.
- **Phase 3 (US1)**: Depends on Phase 2; delivers MVP.
- **Phase 4 (US2)**: Depends on Phase 2; can run in parallel with US1 after foundation, but recommended after MVP.
- **Phase 5 (US3)**: Depends on Phase 2 and benefits from US1/US2 outputs.
- **Phase 6 (Polish)**: Depends on completion of targeted user stories.

### User Story Dependencies

- **US1 (P1)**: No dependency on other stories after foundational phase.
- **US2 (P2)**: No strict dependency on US1, but reuses pipeline run semantics from foundational tasks.
- **US3 (P3)**: Uses monitoring assets introduced in US1 and telemetry completeness from US2 for full readiness checks.

### Within Each User Story

- Write tests first and confirm failure before implementation.
- Implement core instrumentation before dashboard/alert wiring.
- Complete smoke checks before story sign-off.

### Parallel Opportunities

- Setup tasks marked `[P]`: T003, T004, T006.
- Foundational tasks marked `[P]`: T008, T009, T010.
- Mandatory observability stack setup tasks: T011, T012, T043, T035.
- OpenLineage Spark listener reuse tasks: T045, T026, T046.
- US1 tests marked `[P]`: T013, T014, T015.
- US2 tests marked `[P]`: T023, T024, T025.
- US3 tests marked `[P]`: T031, T032, T033.
- Polish tasks marked `[P]`: T039, T041.

---

## Parallel Example: User Story 2

```bash
# Run US2 contract/integration tests in parallel:
Task: "T023 [US2] Add contract test in ai-monitor-system/tests/contract/test_lineage_contract.py"
Task: "T024 [US2] Add integration test in ai-monitor-system/tests/integration/test_lineage_correlation.py"
Task: "T025 [US2] Add integration test in ai-monitor-system/tests/integration/test_trace_attributes.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Deliver Phase 3 (US1).
3. Validate failed-run detection and alert workflow in local cluster.
4. Demo and baseline operational feedback.

### Incremental Delivery

1. Add US2 for lineage + trace correlation.
2. Add US3 for standards coverage and onboarding readiness.
3. Finish with Polish phase for docs and end-to-end validation artifacts.

### Parallel Team Strategy

1. Team aligns on Setup + Foundational tasks.
2. After foundation:
   - Engineer A: US1
   - Engineer B: US2
   - Engineer C: US3 readiness automation
3. Merge at story checkpoints with smoke validation gates.

---

## Notes

- All tasks follow required checklist format with task IDs and file paths.
- `[P]` tasks are selected to avoid same-file conflicts.
- US1 is the recommended MVP scope for first delivery.
- OpenLineage, Prometheus, OpenTelemetry, and Grafana are mandatory and must be integrated via tool configuration and Helm values/templates, not custom alerting code/services.
- Reuse OpenLineage Spark guide configuration keys and listener approach as default integration path; avoid bespoke lineage transport code unless a blocker is documented.
