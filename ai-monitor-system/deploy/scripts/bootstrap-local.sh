#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELM_DIR="$(cd "${SCRIPT_DIR}/../helm" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAMESPACE="ai-monitor-system"
RELEASE_NAME="${RELEASE_NAME:-monitor}"
PIPELINE_IMAGE="local/ai-monitor-pyspark:latest"
VALUES_FILE="${HELM_DIR}/values.yaml"
LOCAL_VALUES_FILE="${HELM_DIR}/values.local-minimal.yaml"
LOCAL_DATA_DIR="${PROJECT_DIR}/.local-data"

NUKE_BEFORE_BOOTSTRAP="${NUKE_BEFORE_BOOTSTRAP:-false}"

BOOTSTRAP_START=$(date +%s)

if [[ "${NUKE_BEFORE_BOOTSTRAP}" == "true" ]]; then
  echo "NUKE_BEFORE_BOOTSTRAP=true → running nuke-local.sh"
  NAMESPACE="${NAMESPACE}" RELEASE_NAME="${RELEASE_NAME}" \
    bash "${SCRIPT_DIR}/nuke-local.sh"
fi

mkdir -p "${LOCAL_DATA_DIR}/input" "${LOCAL_DATA_DIR}/output"

echo "Building local pipeline image: ${PIPELINE_IMAGE}"
docker build -t "${PIPELINE_IMAGE}" "${PROJECT_DIR}"

kubectl delete job pyspark-pipeline -n "${NAMESPACE}" --ignore-not-found

echo "Adding upstream Helm repositories"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts >/dev/null 2>&1 || true
helm repo add ilum https://charts.ilum.cloud >/dev/null 2>&1 || true
helm repo update >/dev/null

echo "Building chart dependencies"
helm dependency build "${HELM_DIR}" >/dev/null

if [[ -f "${LOCAL_VALUES_FILE}" ]]; then
  VALUES_FILE="${LOCAL_VALUES_FILE}"
fi

echo "Deploying release with values: ${VALUES_FILE}"
helm upgrade --install "${RELEASE_NAME}" "${HELM_DIR}" \
  --namespace "${NAMESPACE}" \
  --create-namespace \
  --set global.namespace="${NAMESPACE}" \
  --set-string localData.hostPath="${LOCAL_DATA_DIR}" \
  -f "${VALUES_FILE}"

BOOTSTRAP_DURATION_SECONDS=$(( $(date +%s) - BOOTSTRAP_START ))
echo "Local stack bootstrapped in namespace: ${NAMESPACE}"
echo "BOOTSTRAP_DURATION_SECONDS=${BOOTSTRAP_DURATION_SECONDS}"
