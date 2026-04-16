#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="${SCRIPT_DIR}/../helm"
PROJECT_DIR="${SCRIPT_DIR}/../.."
NAMESPACE="ai-monitor-system"
PIPELINE_IMAGE="local/ai-monitor-pyspark:latest"
VALUES_FILE="${HELM_DIR}/values.yaml"
LOCAL_VALUES_FILE="${HELM_DIR}/values.local-minimal.yaml"

echo "Building local pipeline image: ${PIPELINE_IMAGE}"
docker build -t "${PIPELINE_IMAGE}" "${PROJECT_DIR}"

kubectl delete job pyspark-pipeline -n "${NAMESPACE}" --ignore-not-found

echo "Adding upstream Helm repositories (Option B baseline)"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts >/dev/null 2>&1 || true
helm repo update >/dev/null

echo "Building chart dependencies"
helm dependency build "${HELM_DIR}" >/dev/null

if [[ -f "${LOCAL_VALUES_FILE}" ]]; then
  VALUES_FILE="${LOCAL_VALUES_FILE}"
fi

echo "Deploying release with values: ${VALUES_FILE}"
helm upgrade --install ai-monitor-system "${HELM_DIR}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --set global.namespace="${NAMESPACE}" \
  -f "${VALUES_FILE}"

echo "Local stack bootstrapped in namespace: ${NAMESPACE}"
