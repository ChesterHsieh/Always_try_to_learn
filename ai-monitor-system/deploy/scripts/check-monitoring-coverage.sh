#!/usr/bin/env bash
set -euo pipefail

echo "Checking required stack: OpenLineage, Prometheus, OTel Collector, Grafana"
test -f monitoring/otel/collector-config.yaml
test -f monitoring/prometheus/prometheus.yml
test -f monitoring/grafana/datasources.yaml
echo "Coverage files present"
