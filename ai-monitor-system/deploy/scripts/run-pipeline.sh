#!/usr/bin/env bash
set -euo pipefail

spark-submit \
  --conf spark.jars.packages=io.openlineage:openlineage-spark:1.45.0 \
  --conf spark.extraListeners=io.openlineage.spark.agent.OpenLineageSparkListener \
  --conf spark.openlineage.transport.url="${OPENLINEAGE_URL:-http://marquez-api:5000}" \
  --conf spark.openlineage.transport.type="${OPENLINEAGE_TRANSPORT:-http}" \
  --conf spark.openlineage.namespace="${OPENLINEAGE_NAMESPACE:-ai_monitor_system}" \
  pipeline/job.py
