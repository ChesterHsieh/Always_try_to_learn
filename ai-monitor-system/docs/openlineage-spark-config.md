# OpenLineage Spark Config

Reuse official Spark listener configuration:

- `spark.jars.packages=io.openlineage:openlineage-spark:1.45.0`
- `spark.extraListeners=io.openlineage.spark.agent.OpenLineageSparkListener`
- `spark.openlineage.transport.url=<backend-url>`
- `spark.openlineage.transport.type=http`
- `spark.openlineage.namespace=ai_monitor_system`

Version pin is aligned with the OpenLineage Spark guide currently referenced by this feature.

## Backend Compatibility Guidance

- Preferred local backend: Marquez deployed via upstream chart (`upstream-marquez`).
- Keep Spark package and backend API versions aligned when upgrading.
- Validate connectivity by checking:
  - pipeline pod environment variable `OPENLINEAGE_URL`
  - `spark-defaults.conf` values in `spark-defaults` ConfigMap
  - backend service reachability from namespace `ai-monitor-system`

## Upgrade Notes

- Update chart version pins in both:
  - `deploy/helm/Chart.yaml`
  - `docs/chart-version-matrix.md`
- Re-run contract, integration, and smoke tests after any OpenLineage version change.
