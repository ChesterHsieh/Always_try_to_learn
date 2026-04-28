import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timezone

from pyspark.sql import functions as F

from pipeline.failure_classifier import classify_failure
from pipeline.io_adapter import read_local_file, write_local_file
from pipeline.lineage_emitter import maybe_shadow_emit
from pipeline.metrics import get_metrics_port, get_recorder, push_metrics_if_configured
from pipeline.telemetry import lifecycle_metric_payload
from pipeline.tracing import get_current_trace_id, start_run_span

logger = logging.getLogger(__name__)

PIPELINE_NAME = "ai_monitor_simple_pipeline"
K8S_NAMESPACE = os.environ.get("K8S_NAMESPACE", "ai-monitor-system")


def create_spark_session():
    from pyspark.sql import SparkSession

    openlineage_jar = os.getenv(
        "OPENLINEAGE_JAR_PATH",
        "/opt/openlineage/openlineage-spark_2.12-1.45.0.jar",
    )

    return (
        SparkSession.builder.master("local[*]")
        .appName(PIPELINE_NAME)
        .config("spark.jars", openlineage_jar)
        .config("spark.extraListeners", "io.openlineage.spark.agent.OpenLineageSparkListener")
        .config(
            "spark.openlineage.transport.url",
            os.getenv("OPENLINEAGE_URL", "http://ai-monitor-system-upstream-marquez:9555"),
        )
        .config("spark.openlineage.transport.type", os.getenv("OPENLINEAGE_TRANSPORT", "http"))
        .config(
            "spark.openlineage.namespace", os.getenv("OPENLINEAGE_NAMESPACE", "ai_monitor_system")
        )
        .getOrCreate()
    )


def run_pipeline(input_path: str, output_path: str) -> dict[str, str]:
    from pipeline import otel_setup

    run_id = str(uuid.uuid4())
    start_time = time.monotonic()

    # Initialize OTel tracer (fail-fast on error)
    otel_setup.configure_tracer()

    # Start metrics endpoint (fail-fast on error)
    recorder = get_recorder()
    metrics_port = get_metrics_port()
    # Only start the HTTP server outside of test mode (METRICS_SKIP_SERVER env)
    if not os.environ.get("METRICS_SKIP_SERVER"):
        try:
            recorder.start_endpoint(port=metrics_port)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("Metrics endpoint failed to start: %s", exc, extra={"run_id": run_id})
            raise SystemExit(78) from exc

    failure_info: dict | None = None

    try:
        with start_run_span(
            run_id=run_id, pipeline_name=PIPELINE_NAME, k8s_namespace=K8S_NAMESPACE
        ):
            trace_id = get_current_trace_id()
            recorder.record_run_started(run_id=run_id, pipeline_name=PIPELINE_NAME)

            spark = create_spark_session()
            spark_output_dir = f"{output_path}.spark"
            if os.path.exists(spark_output_dir):
                shutil.rmtree(spark_output_dir)

            transformed_df = spark.read.text(input_path).select(
                F.upper(F.col("value")).alias("value")
            )
            transformed_df.write.mode("overwrite").text(spark_output_dir)

            output_part_files = [
                f
                for f in os.listdir(spark_output_dir)
                if f.startswith("part-") and not f.endswith(".crc")
            ]
            if not output_part_files:
                raise RuntimeError(f"Spark output part file not found in {spark_output_dir}")

            output_part_path = os.path.join(spark_output_dir, sorted(output_part_files)[0])
            text = read_local_file(output_part_path)
            write_local_file(output_path, text.strip() + "\n")

            spark.stop()
            duration = time.monotonic() - start_time
            record_count = len(text.strip().splitlines())

            recorder.record_run_succeeded(
                run_id=run_id,
                pipeline_name=PIPELINE_NAME,
                duration_seconds=duration,
                records_processed=record_count,
                trace_id=trace_id,
            )
            recorder.update_freshness(
                run_id=run_id,
                pipeline_name=PIPELINE_NAME,
                seconds_since_last_event=0.0,
            )

            maybe_shadow_emit(
                run_id=run_id,
                job_name=PIPELINE_NAME,
                namespace=K8S_NAMESPACE,
                source_dataset=input_path,
                target_dataset=output_path,
                trace_id=trace_id,
            )

    except Exception as err:
        duration = time.monotonic() - start_time
        failure_category = classify_failure(err)
        failure_message = str(err)
        failure_info = {"failure_category": failure_category, "failure_message": failure_message}

        try:
            spark.stop()  # type: ignore[possibly-undefined]
        except Exception:
            pass

        recorder.record_run_failed(
            run_id=run_id,
            pipeline_name=PIPELINE_NAME,
            duration_seconds=duration,
            failure_category=failure_category,
            failure_message=failure_message,
            trace_id=trace_id,
        )
        recorder.update_freshness(
            run_id=run_id,
            pipeline_name=PIPELINE_NAME,
            seconds_since_last_event=duration,
        )

        maybe_shadow_emit(
            run_id=run_id,
            job_name=PIPELINE_NAME,
            namespace=K8S_NAMESPACE,
            source_dataset=input_path,
            target_dataset=output_path,
            trace_id=trace_id,
        )

    if failure_info:
        return lifecycle_metric_payload(
            run_id,
            "failed",
            pipeline_name=PIPELINE_NAME,
            failure_category=failure_info["failure_category"],
            failure_message=failure_info["failure_message"],
        )
    return lifecycle_metric_payload(run_id, "succeeded", pipeline_name=PIPELINE_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_path = os.getenv("INPUT_PATH", "/data/input/sample.txt")
    output_path = os.getenv("OUTPUT_PATH", "/data/output/result.txt")
    try:
        payload = run_pipeline(input_path, output_path)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        print(payload)
    finally:
        from pipeline import otel_setup

        otel_setup.force_flush(timeout_millis=2000)
        # Batch jobs exit before Prometheus can scrape — push final metrics
        # to the Pushgateway if PUSHGATEWAY_URL is set in the env.
        push_metrics_if_configured(job_name="pyspark-pipeline", pipeline_name=PIPELINE_NAME)
