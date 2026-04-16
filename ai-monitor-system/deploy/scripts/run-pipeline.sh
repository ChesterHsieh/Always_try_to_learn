#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_DATA_DIR="${PROJECT_ROOT}/.local-data"
INPUT_PATH="${INPUT_PATH:-${LOCAL_DATA_DIR}/input/sample.txt}"
OUTPUT_PATH="${OUTPUT_PATH:-${LOCAL_DATA_DIR}/output/result.txt}"

mkdir -p "$(dirname "${INPUT_PATH}")" "$(dirname "${OUTPUT_PATH}")"
if [[ ! -f "${INPUT_PATH}" ]]; then
  echo "hello lineage" > "${INPUT_PATH}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export INPUT_PATH
export OUTPUT_PATH

spark-submit \
  --conf spark.jars.packages=io.openlineage:openlineage-spark_2.13:1.45.0 \
  --conf spark.extraListeners=io.openlineage.spark.agent.OpenLineageSparkListener \
  --conf spark.openlineage.transport.url="${OPENLINEAGE_URL:-http://marquez-api:5000}" \
  --conf spark.openlineage.transport.type="${OPENLINEAGE_TRANSPORT:-http}" \
  --conf spark.openlineage.namespace="${OPENLINEAGE_NAMESPACE:-ai_monitor_system}" \
  --conf spark.driverEnv.PYTHONPATH="${PROJECT_ROOT}" \
  --conf spark.executorEnv.PYTHONPATH="${PROJECT_ROOT}" \
  pipeline/job.py
