---
description: Interactively design a new failure scenario YAML + matching alert rule + (if needed) injection/classifier patches.
argument-hint: "[scenario-name | failure-keyword]"
---

You are about to walk the user through creating a brand-new monitor-coverage
scenario for `ai-monitor-system`. The output of this conversation is a set of
concrete, runnable artefacts:

1. `scenarios/<name>.yaml` — passes `utils/scenario_schema.py` validation.
2. A matching Prometheus alert rule appended to
   `monitoring/alerts/pipeline-failure-rules.yaml` (only if the failure
   category does not already have a dedicated alert).
3. (Optional) `pipeline/failure_injection.py` injection branch + classifier
   pattern in `pipeline/failure_classifier.py` — only if the user is
   introducing a brand-new failure category not in `KNOWN_CATEGORIES`.
4. (Optional) Contract test scaffolding so the new scenario is exercised by
   `tests/contract/test_failure_classifier_contract.py` and
   `tests/integration/test_failure_scenario_integration.py`.

Argument provided: $ARGUMENTS — treat as a hint for the scenario name or the
failure-keyword the user wants to capture. If empty, ask the user what
failure they want to model.

# Hard constraints (enforce these without asking)

- Scenario YAML MUST validate against [utils/scenario_schema.py](../../utils/scenario_schema.py).
  - `expected_failure_category` MUST be either `null` (when
    `expected_run_status: succeeded`) or a member of
    [`KNOWN_CATEGORIES`](../../pipeline/failure_classifier.py).
  - `pipeline.inject_failure` MUST be `none` or a member of
    `KNOWN_CATEGORIES`.
- Alert names follow the existing convention in
  [monitoring/alerts/pipeline-failure-rules.yaml](../../monitoring/alerts/pipeline-failure-rules.yaml):
  `Pipeline<PascalCase>` (e.g. `PipelineRunTimeout`,
  `PipelineSparkDriverError`). Do not invent a new naming style.
- Probe `cmd` values are restricted to the three real subcommands of
  `scripts/probe.py`: `prom-query`, `otel-trace`, `lineage-run-state`.
- Probes MUST include both `failure_category_metric` and `alert_firing` for
  any failed scenario (these are the two checks that prove monitoring saw
  the failure). Add `run_failed_metric` unless the user has a reason not to.
- Lineage probes use `run_id: "{{ run_id }}"` — the runner substitutes the
  pipeline-emitted run_id at execution time.

# Conversation protocol

Run a multi-turn dialogue. Do NOT write any file until the user has
explicitly approved the final draft in step 6. Use clear numbered prompts so
the user can answer one question at a time.

## Step 1 — Frame the failure

Ask the user to describe, in 1–3 sentences:
- What real-world failure are we simulating? (e.g. "S3 listing returns
  empty even though data exists", "Tempo collector OOMs mid-run")
- What should monitoring be expected to detect? (metric, alert, lineage
  state, trace span, or some combination)
- Is this an existing failure category (one of `KNOWN_CATEGORIES`) or a
  brand-new one?

If the user is unsure which category fits, list `KNOWN_CATEGORIES` from
`pipeline/failure_classifier.py` with a one-line description of each, and
let them pick.

## Step 2 — Decide scenario shape

Based on step 1, propose:
- A scenario `name` (kebab-case, ≤ 30 chars). If $ARGUMENTS gave a hint,
  start from it.
- `expected_run_status` (`succeeded` | `failed`).
- `expected_failure_category` (or `null`).
- `pipeline.inject_failure` — usually equals `expected_failure_category`
  for a single-run scenario; `none` for a baseline or multi-run scenario.
- Whether `pre_runs` are needed (only for scenarios that depend on prior
  state — e.g. schema-mismatch needs a baseline write so the second run
  can read a stale schema).

Show the user the proposal and ask for confirmation or edits.

## Step 3 — Decide alert coverage

Look at `monitoring/alerts/pipeline-failure-rules.yaml` and decide:
- Does an existing alert already match this `failure_category`?
  (`PipelineRunFailed` is the catch-all for input/path/permission/
  spark_task/runtime_error; the others are dedicated.)
- If yes: reuse it. The scenario's `expected_alerts` lists that name.
- If no (e.g. user is adding a brand-new category): draft a new alert
  rule entry, following the existing PromQL pattern:

  ```yaml
  - alert: Pipeline<PascalCase>
    expr: >
      increase(pipeline_failures_total{failure_category="<category>"}[5m]) > 0
    for: 30s
    labels:
      severity: <warning|critical>
    annotations:
      summary: "<one-line summary using {{ $labels.pipeline_name }}>"
      runbook_link: "https://docs.example.com/runbook#failure-<category>"
      dashboard_link: "http://grafana/d/pipeline-health?var-pipeline_name={{ $labels.pipeline_name }}&from=now-15m&to=now"
  ```

  Ask the user to choose `severity` (default: `warning` for
  observability/lineage failures, `critical` for data-correctness or
  driver-level failures).

## Step 4 — Pick probes

For each detection path the user named in step 1, propose a probe entry.
Reference existing scenarios as templates — DO NOT invent probe shapes:

- Metric the failure was classified:
  ```yaml
  - id: failure_category_metric
    cmd: prom-query
    args: { query: 'pipeline_failures_total{failure_category="<cat>"}', gte: 1, within: 60 }
  ```
- Run-level failure metric:
  ```yaml
  - id: run_failed_metric
    cmd: prom-query
    args: { query: 'pipeline_run_total{status="failed"}', gte: 1, within: 60 }
  ```
- Alert firing:
  ```yaml
  - id: alert_firing
    cmd: prom-query
    args: { query: 'ALERTS{alertname="<AlertName>",alertstate="firing"}', gte: 1, within: 60 }
  ```
- Trace span carries run_id (only if the failure produces a span):
  ```yaml
  - id: error_span
    cmd: otel-trace
    args: { service: pyspark-pipeline, has_attr: run_id, within: 60 }
  ```
- Lineage state (only if lineage backend is expected to record the run):
  ```yaml
  - id: lineage_run_failed
    cmd: lineage-run-state
    args: { run_id: "{{ run_id }}", state_eq: FAILED, within: 60 }
  ```

Ask the user to confirm the probe set or add/remove entries.

## Step 5 — Decide injection / classifier work

If `expected_failure_category` is already in `KNOWN_CATEGORIES`:
- Confirm `pipeline/failure_injection.py::_raise_for_category` already has
  a branch for it (it should — every known category does). No code change
  needed; the scenario YAML alone is enough.

If the user is introducing a new category:
- Plan the diff to `pipeline/failure_classifier.py`:
  - Add the new string to `KNOWN_CATEGORIES`.
  - Add a regex pattern + branch in `classify_failure`.
- Plan the diff to `pipeline/failure_injection.py`:
  - Add a branch in `_raise_for_category` raising the chosen exception
    type (matching the new classifier pattern).
  - Add the category to `STAGE_FOR_CATEGORY` so `maybe_inject` knows when
    to fire.
- Plan the diff to `tests/contract/test_failure_classifier_contract.py`
  to add a parametrized case proving the regex matches.

State these as "planned diffs" — do not write the code yet.

## Step 6 — Confirm and emit artefacts

Show the full proposed YAML, the alert rule diff (if any), and the
classifier/injection diffs (if any) in a single message. Ask the user
"Proceed to write these files?" and only on explicit yes:

1. `Write` the scenario YAML to `scenarios/<name>.yaml`.
2. `Edit` `monitoring/alerts/pipeline-failure-rules.yaml` to append the
   new alert rule (only if step 3 produced one).
3. `Edit` `pipeline/failure_classifier.py` and
   `pipeline/failure_injection.py` (only if step 5 added a new category).
4. `Edit` the contract test file (only if step 5 added a new category).

After writing, run the cheap validators in this order:

```bash
cd /Users/chester/Desktop/Always_try_to_learn/ai-monitor-system && \
  uv run python -c "from utils.scenario_schema import load_scenario; load_scenario('scenarios/<name>.yaml'); print('schema OK')" && \
  uv run ruff check pipeline utils tests && \
  uv run pytest -q tests/contract
```

If any step fails, surface the error verbatim and ask the user how to
proceed — do NOT auto-edit the YAML to "make it pass".

## Step 7 — Tell the user how to run it

End with a one-liner:

> Scenario ready. Bring up the local stack with
> `./deploy/scripts/bootstrap-local.sh`, then verify end-to-end with
> `/verify <name>`.

# Things you must NOT do

- Do not write any file before step 6 approval.
- Do not invent failure categories, alert names, or probe `cmd` values
  outside the constraints listed above.
- Do not silently widen `KNOWN_CATEGORIES` — adding a category is a
  deliberate, user-confirmed step that touches classifier + injector +
  contract test in lock-step.
- Do not run the full smoke test or `bootstrap-local.sh` from this
  command; only the cheap validators in step 6.
- Do not summarise the scenario back to the user in fluffy prose at the
  end — the artefacts speak for themselves; one-line "ready" message is
  enough.
