#!/usr/bin/env bash
# Scenario runner — executes a YAML scenario end-to-end:
#   1. (optional) trigger pipeline with scenario-defined inputs
#   2. run each probe sequentially against the live monitoring stack
#   3. emit a per-probe verdict line + final summary; exit non-zero if any FAIL
#
# Usage: ./scripts/run-scenario.sh <scenario-name> [--no-pipeline]
#   <scenario-name>: name (without .yaml) under scenarios/
#   --no-pipeline:   skip pipeline trigger; only run probes (useful for dev /
#                    when pipeline already ran or you only want to verify the
#                    monitoring stack itself)
#
# YAML parsing intentionally minimal — uses python via `uv run` to avoid a
# yq dependency, and to keep the contract with probe.py uniform.
#
# Exit codes:
#   0 — all probes PASS
#   1 — one or more probes FAIL
#   2 — runner / probe internal error (couldn't even execute)

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

if [ "$RUN_PIPELINE" -eq 1 ]; then
  if [ -x "./deploy/scripts/run-pipeline.sh" ]; then
    echo "--- triggering pipeline ---"
    ./deploy/scripts/run-pipeline.sh || {
      echo "pipeline trigger failed (continuing to probes anyway so we capture state)" >&2
    }
  else
    echo "deploy/scripts/run-pipeline.sh not executable; skipping pipeline trigger" >&2
  fi
fi

echo "--- running probes ---"

# Use python to parse YAML and emit one shell-quoted command per probe.
# Output format (per line): <probe_id>\t<probe_cmd>\t<json_args>
PROBE_PLAN="$(uv run python <<PY
import json, sys, yaml
with open("${SCENARIO_FILE}") as f:
    spec = yaml.safe_load(f)
for p in spec.get("probes", []):
    print(f"{p['id']}\t{p['cmd']}\t{json.dumps(p.get('args', {}))}")
PY
)"

if [ -z "$PROBE_PLAN" ]; then
  echo "no probes defined in $SCENARIO_FILE" >&2
  exit 2
fi

PASS=0; FAIL=0; ERROR=0
SUMMARY=""

while IFS=$'\t' read -r pid pcmd pargs; do
  [ -z "$pid" ] && continue
  # Translate JSON args dict → CLI flags. Convention: each key becomes
  # --key (with underscores → hyphens); list values become repeated --key.
  CLI_ARGS=$(uv run python <<PY
import json, shlex, sys
args = json.loads('''$pargs''')
out = []
for k, v in args.items():
    flag = "--" + k.replace("_", "-")
    if isinstance(v, list):
        for item in v:
            out += [flag, str(item)]
    elif isinstance(v, bool):
        if v: out += [flag]
    else:
        # query is a positional for prom-query; emit as-is via flag-less below
        out += [flag, str(v)]
print(" ".join(shlex.quote(x) for x in out))
PY
)

  # prom-query takes the query as a positional. Re-route if present.
  POSITIONAL=""
  if [ "$pcmd" = "prom-query" ]; then
    QUERY=$(uv run python <<PY
import json
print(json.loads('''$pargs''').get("query",""))
PY
)
    POSITIONAL=$(printf '%q' "$QUERY")
    # strip --query <q> from CLI_ARGS
    CLI_ARGS=$(echo "$CLI_ARGS" | sed -E "s/--query [^ ]+( |$)//")
  fi

  printf "  %-28s " "$pid"
  RESULT=$(eval uv run python scripts/probe.py "$pcmd" --terse $POSITIONAL $CLI_ARGS 2>&1)
  EXIT=$?

  case $EXIT in
    0) PASS=$((PASS+1)); STATUS="PASS" ;;
    1) FAIL=$((FAIL+1)); STATUS="FAIL" ;;
    *) ERROR=$((ERROR+1)); STATUS="ERROR" ;;
  esac
  echo "$STATUS  $RESULT"
  SUMMARY="${SUMMARY}${pid}\t${STATUS}\n"
done <<< "$PROBE_PLAN"

echo "--- summary ---"
printf "$SUMMARY"
TOTAL=$((PASS+FAIL+ERROR))
echo "VERDICT: $PASS/$TOTAL PASS  ($FAIL FAIL, $ERROR ERROR)"

if [ $ERROR -gt 0 ]; then exit 2; fi
if [ $FAIL -gt 0 ]; then exit 1; fi
exit 0
