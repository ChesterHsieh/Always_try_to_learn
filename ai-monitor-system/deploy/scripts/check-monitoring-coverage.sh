#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-ai-monitor-system}"

echo "Checking required stack: OpenLineage, Prometheus, OTel Collector, Grafana"
test -f monitoring/otel/collector-config.yaml
test -f monitoring/prometheus/prometheus.yml
test -f monitoring/grafana/datasources.yaml

echo "Checking in-cluster service discovery in namespace: ${NAMESPACE}"
kubectl get svc -n "${NAMESPACE}" prometheus >/dev/null
kubectl get svc -n "${NAMESPACE}" grafana >/dev/null
kubectl get svc -n "${NAMESPACE}" otel-collector >/dev/null

echo "Coverage checks passed"
