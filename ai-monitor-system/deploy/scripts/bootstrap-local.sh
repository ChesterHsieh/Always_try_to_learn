#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="${SCRIPT_DIR}/../helm"

helm upgrade --install ai-monitor-system "${HELM_DIR}" -f "${HELM_DIR}/values.yaml"
echo "Local stack bootstrapped"
