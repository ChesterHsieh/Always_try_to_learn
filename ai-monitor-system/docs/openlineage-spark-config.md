# OpenLineage Spark Config

Reuse official Spark listener configuration:

- `spark.jars.packages=io.openlineage:openlineage-spark:1.45.0`
- `spark.extraListeners=io.openlineage.spark.agent.OpenLineageSparkListener`
- `spark.openlineage.transport.url=<backend-url>`
- `spark.openlineage.transport.type=http`
- `spark.openlineage.namespace=ai_monitor_system`

Version pin is aligned with the OpenLineage Spark guide currently referenced by this feature.
