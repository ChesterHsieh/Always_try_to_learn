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

INJECT_FAILURE_ARGS=""
if [[ -n "${INJECT_FAILURE:-}" && "${INJECT_FAILURE}" != "none" ]]; then
  INJECT_FAILURE_ARGS="--set pyspark.injectFailure=${INJECT_FAILURE}"
fi

SCHEMA_VERSION_ARGS=""
if [[ -n "${SCHEMA_VERSION:-}" && "${SCHEMA_VERSION}" != "v1" ]]; then
  SCHEMA_VERSION_ARGS="--set pyspark.schemaVersion=${SCHEMA_VERSION}"
fi

helm upgrade --install "${RELEASE_NAME}" "${HELM_DIR}" \
  --namespace "${NAMESPACE}" \
  --set global.namespace="${NAMESPACE}" \
  --set-string localData.hostPath="${LOCAL_DATA_DIR}" \
  -f "${VALUES_FILE}" \
  ${INJECT_FAILURE_ARGS} \
  ${SCHEMA_VERSION_ARGS} >/dev/null

EXIT_CODE=0
# Poll for either complete or failed — don't wait the full 300s on failure scenarios.
DEADLINE=$(( $(date +%s) + 300 ))
JOB_DONE=0
while [ $(date +%s) -lt $DEADLINE ]; do
  STATUS="$(kubectl get job pyspark-pipeline -n "${NAMESPACE}" \
    -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null || true)"
  if echo "$STATUS" | grep -qE "Complete|Failed"; then
    JOB_DONE=1
    break
  fi
  sleep 3
done

if [ $JOB_DONE -eq 0 ]; then
  echo "ERROR: Pipeline job TIMED OUT after 300s"
  EXIT_CODE=2
elif echo "$STATUS" | grep -q "Failed"; then
  echo "ERROR: Pipeline job FAILED"
  EXIT_CODE=1
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
