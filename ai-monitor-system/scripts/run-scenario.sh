#!/usr/bin/env bash
# Scenario runner — executes a YAML scenario end-to-end:
#   1. (optional) trigger pipeline with scenario-defined inputs
#   2. compare expected_run_status / expected_failure_category to actual payload
#   3. run each probe sequentially against the live monitoring stack
#   4. emit a per-probe verdict line + final summary; exit non-zero if any FAIL
#
# Usage: ./scripts/run-scenario.sh <scenario-name> [--no-pipeline]
#
# Exit codes: 0 = all pass, 1 = at least one FAIL, 2 = script/parse error.
#
# Implementation note: probe arguments live in YAML and may contain quotes,
# braces, commas, etc. (e.g. PromQL labels like `{status="succeeded"}`). We
# never inline JSON into shell heredocs — those break on any quote. Instead,
# a single Python pass reads the scenario, builds the full argv per probe,
# and emits NUL-delimited records that bash safely consumes.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
  echo "usage: $0 <scenario-name> [--no-pipeline]" >&2
  exit 2
fi

SCENARIO_NAME="$1"; shift
RUN_PIPELINE=1
for arg in "$@"; do
  case "$arg" in
    --no-pipeline) RUN_PIPELINE=0 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

SCENARIO_FILE="scenarios/${SCENARIO_NAME}.yaml"
if [ ! -f "$SCENARIO_FILE" ]; then
  echo "scenario not found: $SCENARIO_FILE" >&2
  exit 2
fi

echo "=== Scenario: $SCENARIO_NAME ==="

# ---------------------------------------------------------------------------
# Extract expected values from scenario YAML (single Python pass)
# ---------------------------------------------------------------------------
EXPECTED_JSON="$(SCENARIO_FILE="$SCENARIO_FILE" uv run python <<'PY'
import json, os, yaml

with open(os.environ["SCENARIO_FILE"]) as f:
    spec = yaml.safe_load(f)

pipeline = spec.get("pipeline") or {}
pre_runs = pipeline.get("pre_runs") or []

print(json.dumps({
    "expected_run_status": spec.get("expected_run_status"),
    "expected_failure_category": spec.get("expected_failure_category"),
    "inject_failure": pipeline.get("inject_failure", "none"),
    "schema_version": pipeline.get("schema_version", "v1"),
    "pre_runs": [
        {
            "schema_version": pr.get("schema_version", "v1"),
            "inject_failure": pr.get("inject_failure", "none"),
        }
        for pr in pre_runs
    ],
    "expected_alerts": spec.get("expected_alerts") or [],
}))
PY
)"
if [ $? -ne 0 ] || [ -z "$EXPECTED_JSON" ]; then
  echo "failed to parse scenario YAML: $SCENARIO_FILE" >&2
  exit 2
fi

EXPECTED_RUN_STATUS="$(echo "$EXPECTED_JSON" | uv run python -c "import json,sys; d=json.load(sys.stdin); print(d['expected_run_status'])")"
EXPECTED_FAILURE_CATEGORY="$(echo "$EXPECTED_JSON" | uv run python -c "import json,sys; d=json.load(sys.stdin); print(d.get('expected_failure_category') or 'null')")"
INJECT_FAILURE="$(echo "$EXPECTED_JSON" | uv run python -c "import json,sys; d=json.load(sys.stdin); print(d['inject_failure'])")"
SCHEMA_VERSION_MAIN="$(echo "$EXPECTED_JSON" | uv run python -c "import json,sys; d=json.load(sys.stdin); print(d.get('schema_version','v1'))")"
PRE_RUNS_TSV="$(echo "$EXPECTED_JSON" | uv run python -c "import json,sys; d=json.load(sys.stdin); [print(f\"{p['schema_version']}\t{p['inject_failure']}\") for p in d.get('pre_runs',[])]")"

# ---------------------------------------------------------------------------
# Pipeline trigger
# ---------------------------------------------------------------------------

LIFECYCLE_STATUS=""
LIFECYCLE_FAILURE_CATEGORY=""
LIFECYCLE_RUN_ID=""

if [ "$RUN_PIPELINE" -eq 1 ]; then
  if [ -x "./deploy/scripts/run-pipeline.sh" ]; then
    # Optional pre-runs (e.g. baseline schema_v1 run for schema-drift).
    # Output is logged for transparency but lifecycle/probes are not
    # captured — only the main run's payload feeds expected vs actual.
    if [ -n "$PRE_RUNS_TSV" ]; then
      while IFS=$'\t' read -r pre_sv pre_inj; do
        [ -z "$pre_sv" ] && continue
        echo "--- pre-run (schema_version=${pre_sv} inject_failure=${pre_inj}) ---"
        INJECT_FAILURE="$pre_inj" SCHEMA_VERSION="$pre_sv" \
          ./deploy/scripts/run-pipeline.sh >/dev/null 2>&1 || true
      done <<< "$PRE_RUNS_TSV"
    fi

    echo "--- triggering pipeline (inject_failure=${INJECT_FAILURE} schema_version=${SCHEMA_VERSION_MAIN}) ---"
    PIPELINE_OUTPUT="$(INJECT_FAILURE="$INJECT_FAILURE" SCHEMA_VERSION="$SCHEMA_VERSION_MAIN" \
      ./deploy/scripts/run-pipeline.sh 2>&1)" || true
    # Extract lifecycle JSON from last non-empty line of output
    LIFECYCLE_JSON="$(echo "$PIPELINE_OUTPUT" | grep -E '^\{.*"status"' | tail -1 || true)"
    if [ -n "$LIFECYCLE_JSON" ]; then
      LIFECYCLE_STATUS="$(echo "$LIFECYCLE_JSON" | uv run python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('status',''))" 2>/dev/null || true)"
      LIFECYCLE_FAILURE_CATEGORY="$(echo "$LIFECYCLE_JSON" | uv run python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('failure_category','null'))" 2>/dev/null || true)"
      LIFECYCLE_RUN_ID="$(echo "$LIFECYCLE_JSON" | uv run python -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('run_id',''))" 2>/dev/null || true)"
    fi
  else
    echo "deploy/scripts/run-pipeline.sh not executable; skipping pipeline trigger" >&2
  fi
fi

# ---------------------------------------------------------------------------
# Expected vs actual comparison
# ---------------------------------------------------------------------------
echo "--- expected vs actual ---"

LIFECYCLE_FAIL=0

if [ -n "$LIFECYCLE_STATUS" ]; then
  if [ "$LIFECYCLE_STATUS" = "$EXPECTED_RUN_STATUS" ]; then
    printf "  %-36s PASS  actual=%s\n" "lifecycle.run_status" "$LIFECYCLE_STATUS"
  else
    printf "  %-36s FAIL  expected=%s actual=%s\n" "lifecycle.run_status" "$EXPECTED_RUN_STATUS" "$LIFECYCLE_STATUS"
    LIFECYCLE_FAIL=1
  fi

  if [ "$EXPECTED_RUN_STATUS" = "failed" ]; then
    if [ "$LIFECYCLE_FAILURE_CATEGORY" = "$EXPECTED_FAILURE_CATEGORY" ]; then
      printf "  %-36s PASS  actual=%s\n" "lifecycle.failure_category" "$LIFECYCLE_FAILURE_CATEGORY"
    else
      printf "  %-36s FAIL  expected=%s actual=%s\n" "lifecycle.failure_category" "$EXPECTED_FAILURE_CATEGORY" "$LIFECYCLE_FAILURE_CATEGORY"
      LIFECYCLE_FAIL=1
    fi
  fi
else
  echo "  (pipeline not triggered or lifecycle payload not captured; skipping expected vs actual)"
fi

# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------
echo "--- running probes ---"

PROBE_PLAN="$(SCENARIO_FILE="$SCENARIO_FILE" LIFECYCLE_RUN_ID="$LIFECYCLE_RUN_ID" uv run python <<'PY'
import os, re, yaml

SCENARIO_FILE = os.environ["SCENARIO_FILE"]
RUN_ID = os.environ.get("LIFECYCLE_RUN_ID", "")
US = "\x1f"  # unit separator between argv tokens
TEMPLATE_RE = re.compile(r"\{\{\s*run_id\s*\}\}")

def render(value):
    if isinstance(value, str):
        return TEMPLATE_RE.sub(RUN_ID, value)
    return value

with open(SCENARIO_FILE) as f:
    spec = yaml.safe_load(f)

for p in spec.get("probes", []):
    pid = p["id"]
    cmd = p["cmd"]
    args = p.get("args", {}) or {}

    argv = ["--terse"]
    # prom-query takes the query as a positional first
    if cmd == "prom-query" and "query" in args:
        argv.append(render(str(args["query"])))

    for k, v in args.items():
        if cmd == "prom-query" and k == "query":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, list):
            for item in v:
                argv += [flag, render(str(item))]
        elif isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, render(str(v))]

    print(f"{pid}\t{cmd}\t{US.join(argv)}")
PY
)"
PROBE_RC=$?

if [ $PROBE_RC -ne 0 ] || [ -z "$PROBE_PLAN" ]; then
  echo "failed to build probe plan from $SCENARIO_FILE" >&2
  exit 2
fi

PASS=0; FAIL=0; ERROR=0
SUMMARY=""

while IFS=$'\t' read -r pid pcmd argv_packed; do
  [ -z "$pid" ] && continue

  # Split packed argv on Unit Separator into a real bash array.
  IFS=$'\x1f' read -r -a ARGV <<< "$argv_packed"

  printf "  %-32s " "$pid"
  RESULT=$(uv run python scripts/probe.py "$pcmd" "${ARGV[@]}" 2>&1)
  EXIT=$?

  case $EXIT in
    0) PASS=$((PASS+1)); STATUS="PASS" ;;
    1) FAIL=$((FAIL+1)); STATUS="FAIL" ;;
    *) ERROR=$((ERROR+1)); STATUS="ERROR" ;;
  esac
  echo "$STATUS  $RESULT"
  SUMMARY="${SUMMARY}${pid}\t${STATUS}\n"
done <<< "$PROBE_PLAN"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "--- summary ---"
printf "$SUMMARY"
TOTAL=$((PASS+FAIL+ERROR))
echo "VERDICT: $PASS/$TOTAL PASS  ($FAIL FAIL, $ERROR ERROR)"

if [ "$LIFECYCLE_FAIL" -eq 1 ]; then
  echo "VERDICT: lifecycle expected vs actual MISMATCH — see above"
  exit 1
fi

if [ $ERROR -gt 0 ]; then exit 2; fi
if [ $FAIL -gt 0 ]; then exit 1; fi
exit 0
