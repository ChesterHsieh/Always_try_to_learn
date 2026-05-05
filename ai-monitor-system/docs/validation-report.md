# Validation Report

**Owner**: Platform team  
**Last verified**: 2026-04-18T11:14:11.917197+00:00  
**Coverage report**: `.local-data/coverage/<ts>.json`

---

## Execution Steps

```bash
# 1. Contract tests
pytest tests/contract -v

# 2. Integration tests
pytest tests/integration -v

# 3. Smoke tests
bash deploy/scripts/run-smoke-test.sh

# 4. Coverage CLI
python -m pipeline.coverage --output .local-data/coverage/release.json
cat .local-data/coverage/release.json | python -m json.tool
```

---

## Results Summary

| Layer | Tests | Status |
|-------|-------|--------|
| Contract | `tests/contract/` (27 tests) | pass |
| Integration | `tests/integration/` (23 tests) | pass |
| Smoke (mock-based) | `tests/smoke/` (6 tests, cluster tests skipped) | pass |
| Coverage CLI | `pipeline.coverage` | warn (Prometheus unreachable, chart versions OK) |

---

## Coverage Report Fields

From `.local-data/coverage/<ts>.json`:

| Field | Expected | Actual |
|-------|----------|--------|
| `last_verified_at` | ISO 8601 UTC | `2026-04-18T11:14:11.917197+00:00` |
| `components.upstream-prometheus` | `25.27.0` | `25.27.0` |
| `components.upstream-grafana` | `8.5.1` | `8.5.1` |
| `components.upstream-otel-collector` | `0.78.0` | `0.78.0` |
| `components.upstream-marquez` | `6.7.0` | `6.7.0` |
| `validation_checks` pass count | ≥5 | 0 (cluster not running; backends unreachable) |

---

## Release Gate Criteria

All of the following must be green to release:

- [x] `pytest tests/contract` — exit 0 (27/27 passed)
- [x] `pytest tests/integration` — exit 0 (23/23 passed)
- [x] `bash deploy/scripts/run-smoke-test.sh` — mock-based tests pass; cluster tests require live stack
- [ ] `python -m pipeline.coverage` — exit 2 (Prometheus unreachable; requires deployed stack)
- [x] `coverage-report.json` has `chart_version` semver strings for all 4 components
- [x] `validation-report.md` has `last_verified_at` filled in

---

---

## Failure Scenario Validation Ledger {#failure-scenario-ledger}

Updated automatically by `scripts/run-all-failure-scenarios.sh --update-report`.

| scenario | last_run_at | result | run_id |
|----------|-------------|--------|--------|
| input-not-found | 2026-05-05T04:45:56Z | pass | — |
| invalid-path | 2026-05-05T04:45:56Z | pass | — |
| lineage-emission-failed | 2026-05-05T04:45:56Z | pass | — |
| permission-denied | 2026-05-05T04:45:56Z | pass | — |
| runtime-error | 2026-05-05T04:45:56Z | pass | — |
| schema-drift | 2026-05-05T04:45:56Z | pass | — |
| spark-driver-error | 2026-05-05T04:45:56Z | pass | — |
| spark-task-failed | 2026-05-05T04:45:56Z | pass | — |
| telemetry-unavailable | 2026-05-05T04:45:56Z | pass | — |
| timeout | 2026-05-05T04:45:56Z | pass | — |

---

## Notes

<!-- Add release-specific notes here -->
