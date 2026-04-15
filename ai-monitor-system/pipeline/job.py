import os
import uuid
from datetime import datetime, timezone

from pipeline.io_adapter import read_local_file, write_local_file
from pipeline.telemetry import lifecycle_metric_payload


def create_spark_session():
    from pyspark.sql import SparkSession

    # Reuse OpenLineage Spark listener integration from official guide.
    return (
        SparkSession.builder.master("local[*]")
        .appName("ai_monitor_simple_pipeline")
        .config("spark.jars.packages", "io.openlineage:openlineage-spark:1.45.0")
        .config("spark.extraListeners", "io.openlineage.spark.agent.OpenLineageSparkListener")
        .config("spark.openlineage.transport.url", os.getenv("OPENLINEAGE_URL", "http://marquez-api:5000"))
        .config("spark.openlineage.transport.type", os.getenv("OPENLINEAGE_TRANSPORT", "http"))
        .config("spark.openlineage.namespace", os.getenv("OPENLINEAGE_NAMESPACE", "ai_monitor_system"))
        .getOrCreate()
    )


def run_pipeline(input_path: str, output_path: str) -> dict[str, str]:
    run_id = str(uuid.uuid4())
    _ = create_spark_session()
    text = read_local_file(input_path)
    write_local_file(output_path, text.upper())
    return lifecycle_metric_payload(run_id, "succeeded")


if __name__ == "__main__":
    input_path = os.getenv("INPUT_PATH", "/data/input/sample.txt")
    output_path = os.getenv("OUTPUT_PATH", "/data/output/result.txt")
    payload = run_pipeline(input_path, output_path)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    print(payload)
