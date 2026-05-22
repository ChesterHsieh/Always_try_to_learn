#!/usr/bin/env bash
# Thin wrapper: delegates all logic to Python coverage CLI.
set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-monitor-system}"
MARQUEZ_URL="${MARQUEZ_URL:-http://ai-monitor-system-upstream-marquez:9555}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://ai-monitor-system-upstream-prometheus-server:80}"
GRAFANA_URL="${GRAFANA_URL:-http://ai-monitor-system-upstream-grafana:80}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/.local-data/coverage"
mkdir -p "${OUTPUT_DIR}"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

exec python -m utils.coverage \
  --namespace "${NAMESPACE}" \
  --marquez-url "${MARQUEZ_URL}" \
  --prometheus-url "${PROMETHEUS_URL}" \
  --grafana-url "${GRAFANA_URL}" \
  --output "${OUTPUT_DIR}/${TIMESTAMP}.json"
