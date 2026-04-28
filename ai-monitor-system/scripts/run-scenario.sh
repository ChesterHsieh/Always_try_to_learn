#!/usr/bin/env bash
# Scenario runner — executes a YAML scenario end-to-end:
#   1. (optional) trigger pipeline with scenario-defined inputs
#   2. run each probe sequentially against the live monitoring stack
#   3. emit a per-probe verdict line + final summary; exit non-zero if any FAIL
#
# Usage: ./scripts/run-scenario.sh <scenario-name> [--no-pipeline]
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

if [ "$RUN_PIPELINE" -eq 1 ]; then
  if [ -x "./deploy/scripts/run-pipeline.sh" ]; then
    echo "--- triggering pipeline ---"
    ./deploy/scripts/run-pipeline.sh || {
      echo "pipeline trigger failed (continuing to probes anyway)" >&2
    }
  else
    echo "deploy/scripts/run-pipeline.sh not executable; skipping pipeline trigger" >&2
  fi
fi

echo "--- running probes ---"

# Build per-probe argv lists in a single Python pass. Output format:
#   <id>\t<cmd>\t<argv0><argv1>...<argvN>\n
# Using  (Unit Separator) between argv tokens — never appears in PromQL
# or service names, no escaping needed. \t separates id/cmd/argv-list.
PROBE_PLAN="$(uv run python <<'PY'
import json, os, sys, yaml

SCENARIO_FILE = os.environ["SCENARIO_FILE"]
US = "\x1f"  # unit separator between argv tokens

with open(SCENARIO_FILE) as f:
    spec = yaml.safe_load(f)

for p in spec.get("probes", []):
    pid = p["id"]
    cmd = p["cmd"]
    args = p.get("args", {}) or {}

    argv = ["--terse"]
    # prom-query takes the query as a positional first
    if cmd == "prom-query" and "query" in args:
        argv.append(str(args["query"]))

    for k, v in args.items():
        if cmd == "prom-query" and k == "query":
            continue
        flag = "--" + k.replace("_", "-")
        if isinstance(v, list):
            for item in v:
                argv += [flag, str(item)]
        elif isinstance(v, bool):
            if v:
                argv.append(flag)
        else:
            argv += [flag, str(v)]

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

echo "--- summary ---"
printf "$SUMMARY"
TOTAL=$((PASS+FAIL+ERROR))
echo "VERDICT: $PASS/$TOTAL PASS  ($FAIL FAIL, $ERROR ERROR)"

if [ $ERROR -gt 0 ]; then exit 2; fi
if [ $FAIL -gt 0 ]; then exit 1; fi
exit 0
