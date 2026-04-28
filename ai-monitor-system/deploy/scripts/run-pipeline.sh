#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HELM_DIR="$(cd "${SCRIPT_DIR}/../helm" && pwd)"
NAMESPACE="${NAMESPACE:-ai-monitor-system}"
RELEASE_NAME="${RELEASE_NAME:-monitor}"
PIPELINE_IMAGE="${PIPELINE_IMAGE:-local/ai-monitor-pyspark:latest}"
LOCAL_DATA_DIR="${PROJECT_ROOT}/.local-data"
INPUT_PATH="${INPUT_PATH:-${LOCAL_DATA_DIR}/input/sample.txt}"
OUTPUT_PATH="${OUTPUT_PATH:-${LOCAL_DATA_DIR}/output/result.txt}"

RUN_START=$(date +%s)

mkdir -p "$(dirname "${INPUT_PATH}")" "$(dirname "${OUTPUT_PATH}")"
if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "hello lineage" > "${INPUT_PATH}"
fi

cd "${PROJECT_ROOT}"
echo "Building pipeline image: ${PIPELINE_IMAGE}"
docker build -t "${PIPELINE_IMAGE}" "${PROJECT_ROOT}" >/dev/null

kubectl delete job pyspark-pipeline -n "${NAMESPACE}" --ignore-not-found >/dev/null

VALUES_FILE="${HELM_DIR}/values.yaml"
if [[ -f "${HELM_DIR}/values.local-minimal.yaml" ]]; then
  VALUES_FILE="${HELM_DIR}/values.local-minimal.yaml"
fi

helm upgrade --install "${RELEASE_NAME}" "${HELM_DIR}" \
  --namespace "${NAMESPACE}" \
  --set global.namespace="${NAMESPACE}" \
  --set-string localData.hostPath="${LOCAL_DATA_DIR}" \
  -f "${VALUES_FILE}" >/dev/null

EXIT_CODE=0
if ! kubectl wait --for=condition=complete "job/pyspark-pipeline" -n "${NAMESPACE}" --timeout=300s; then
  # Check if failed
  if kubectl wait --for=condition=failed "job/pyspark-pipeline" -n "${NAMESPACE}" --timeout=5s 2>/dev/null; then
    echo "ERROR: Pipeline job FAILED"
    EXIT_CODE=1
  else
    echo "ERROR: Pipeline job TIMED OUT after 300s"
    EXIT_CODE=2
  fi
fi

POD_NAME="$(kubectl get pod -n "${NAMESPACE}" -l job-name=pyspark-pipeline -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")"
if [[ -n "${POD_NAME}" ]]; then
  kubectl logs -n "${NAMESPACE}" "${POD_NAME}" || true
fi

RUN_DURATION_SECONDS=$(( $(date +%s) - RUN_START ))
echo "RUN_DURATION_SECONDS=${RUN_DURATION_SECONDS}"

if [[ -f "${OUTPUT_PATH}" ]]; then
  echo "Pipeline output file created at ${OUTPUT_PATH}"
fi

exit "${EXIT_CODE}"
