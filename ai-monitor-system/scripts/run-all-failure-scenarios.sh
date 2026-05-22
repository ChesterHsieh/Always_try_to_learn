#!/usr/bin/env bash
# run-all-failure-scenarios.sh — one-shot runner for all failure scenarios.
#
# Executes every scenario with expected_run_status=failed in sequential order,
# collects per-scenario results, and reports a summary table.
#
# Usage:
#   ./scripts/run-all-failure-scenarios.sh [--update-report]
#
# Flags:
#   --update-report   Rewrite the ledger table in docs/validation-report.md
#                     with this run's results and timestamps.
#
# Exit codes:
#   0  All scenarios PASS
#   1  At least one scenario FAIL
#   2  Script/parse error
#
# Compatibility: bash 3.2+ (macOS default)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

UPDATE_REPORT=0
for arg in "$@"; do
  case "$arg" in
    --update-report) UPDATE_REPORT=1 ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Discover failure scenarios (expected_run_status=failed)
# ---------------------------------------------------------------------------
FAILURE_SCENARIOS="$(uv run python <<'PY'
import pathlib, yaml

scenarios = []
for path in sorted(pathlib.Path("scenarios").glob("*.yaml")):
    spec = yaml.safe_load(path.read_text())
    if spec.get("expected_run_status") == "failed":
        scenarios.append(path.stem)

for s in scenarios:
    print(s)
PY
)"

if [ -z "$FAILURE_SCENARIOS" ]; then
  echo "No failure scenarios found in scenarios/" >&2
  exit 2
fi

echo "=== run-all-failure-scenarios ==="
echo "Scenarios to run:"
while IFS= read -r s; do echo "  - $s"; done <<< "$FAILURE_SCENARIOS"
echo ""

# ---------------------------------------------------------------------------
# Run each scenario — collect results into parallel lists (bash 3.2 compat)
# ---------------------------------------------------------------------------
OVERALL_PASS=0
OVERALL_FAIL=0
SCENARIO_LIST=""     # newline-separated scenario names (in order)
RESULT_LIST=""       # newline-separated results (pass/fail, same order)
RUN_ID_LIST=""       # newline-separated run_ids (same order)
TIMESTAMP_NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

while IFS= read -r scenario; do
  echo "--- $scenario ---"
  # Capture output; allow non-zero exit (scenario failure is expected)
  set +e
  OUTPUT="$(./scripts/run-scenario.sh "$scenario" 2>&1)"
  EXIT=$?
  set -e
  echo "$OUTPUT"

  if [ $EXIT -eq 0 ]; then
    RESULT="pass"
    OVERALL_PASS=$((OVERALL_PASS + 1))
  else
    RESULT="fail"
    OVERALL_FAIL=$((OVERALL_FAIL + 1))
  fi

  # Extract run_id from lifecycle JSON line if present
  RUN_ID="$(echo "$OUTPUT" | grep -oE '"run_id":"[^"]*"' | head -1 | sed 's/"run_id":"//;s/"//' || true)"
  RUN_ID="${RUN_ID:-—}"

  SCENARIO_LIST="${SCENARIO_LIST}${scenario}"$'\n'
  RESULT_LIST="${RESULT_LIST}${RESULT}"$'\n'
  RUN_ID_LIST="${RUN_ID_LIST}${RUN_ID}"$'\n'
  echo ""
done <<< "$FAILURE_SCENARIOS"

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo "=== Summary ==="
printf "%-30s  %s\n" "SCENARIO" "RESULT"
printf "%-30s  %s\n" "--------" "------"

# Walk parallel lists together
SC_LINES="$(echo "$SCENARIO_LIST" | grep -v '^$' || true)"
RE_LINES="$(echo "$RESULT_LIST"   | grep -v '^$' || true)"
IDX=1
while IFS= read -r sc; do
  re="$(echo "$RE_LINES" | sed -n "${IDX}p")"
  printf "%-30s  %s\n" "$sc" "$re"
  IDX=$((IDX + 1))
done <<< "$SC_LINES"

TOTAL=$((OVERALL_PASS + OVERALL_FAIL))
echo ""
echo "VERDICT: $OVERALL_PASS/$TOTAL PASS  ($OVERALL_FAIL FAIL)"

# ---------------------------------------------------------------------------
# Update validation report ledger
# ---------------------------------------------------------------------------
if [ "$UPDATE_REPORT" -eq 1 ]; then
  echo ""
  echo "--- updating docs/validation-report.md ledger ---"

  LEDGER_TMPFILE="$(mktemp)"

  {
    echo "| scenario | last_run_at | result | run_id |"
    echo "|----------|-------------|--------|--------|"
    IDX=1
    while IFS= read -r sc; do
      re="$(echo "$RE_LINES" | sed -n "${IDX}p")"
      rid="$(echo "$RUN_ID_LIST" | grep -v '^$' | sed -n "${IDX}p")"
      rid="${rid:-—}"
      echo "| $sc | $TIMESTAMP_NOW | $re | $rid |"
      IDX=$((IDX + 1))
    done <<< "$SC_LINES"
  } > "$LEDGER_TMPFILE"

  uv run python <<PYEOF
import re, pathlib

report_path = pathlib.Path("docs/validation-report.md")
new_table = pathlib.Path("$LEDGER_TMPFILE").read_text()
text = report_path.read_text()

pattern = re.compile(
    r"(\| scenario \| last_run_at \| result \| run_id \|\n\|[-|]+\|\n)"
    r"(?:\|[^\n]+\|\n)*",
    re.MULTILINE,
)
new_text = pattern.sub(new_table, text, count=1)
report_path.write_text(new_text)
print("Ledger updated.")
PYEOF

  rm -f "$LEDGER_TMPFILE"
fi

# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------
if [ "$OVERALL_FAIL" -gt 0 ]; then exit 1; fi
exit 0
