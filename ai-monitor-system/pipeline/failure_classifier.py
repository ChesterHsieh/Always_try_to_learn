import re
import socket
from typing import Final

KNOWN_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "input_not_found",
        "invalid_path",
        "permission_denied",
        "spark_task_failed",
        "spark_driver_error",
        "lineage_emission_failed",
        "telemetry_unavailable",
        "timeout",
        "runtime_error",
    }
)

_SPARK_TASK_PATTERN = re.compile(
    r"task.*failed|TaskFailed|org\.apache\.spark.*TaskFailed", re.IGNORECASE
)
_SPARK_DRIVER_PATTERN = re.compile(
    r"driver.*error|SparkException|org\.apache\.spark\.SparkException", re.IGNORECASE
)
_LINEAGE_PATTERN = re.compile(r"openlineage|marquez|lineage", re.IGNORECASE)
_TELEMETRY_PATTERN = re.compile(r"otel|opentelemetry|prometheus|collector", re.IGNORECASE)


def classify_failure(error: BaseException) -> str:
    type_name = type(error).__name__

    if isinstance(error, FileNotFoundError):
        return "input_not_found"
    if isinstance(error, IsADirectoryError):
        return "invalid_path"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "timeout"

    # Check for Py4J Java errors (PySpark bridge exceptions)
    if type_name == "Py4JJavaError":
        msg = str(error)
        if _LINEAGE_PATTERN.search(msg):
            return "lineage_emission_failed"
        if _SPARK_TASK_PATTERN.search(msg):
            return "spark_task_failed"
        return "spark_driver_error"

    # PySpark-native plan analyzer exceptions (e.g. UNRESOLVED_COLUMN,
    # type mismatches discovered before execution). These come from
    # `pyspark.errors.exceptions.captured` and are thin wrappers over
    # Spark Catalyst errors — drift between expected and actual schema
    # is the canonical example.
    if type_name in {"AnalysisException", "ParseException", "SparkUpgradeException"}:
        return "spark_driver_error"

    try:
        import requests.exceptions as req_exc

        if isinstance(error, req_exc.ConnectionError):
            msg = str(error)
            if _LINEAGE_PATTERN.search(msg):
                return "lineage_emission_failed"
            if _TELEMETRY_PATTERN.search(msg):
                return "telemetry_unavailable"
            return "lineage_emission_failed"
        if isinstance(error, (req_exc.Timeout, req_exc.ReadTimeout, req_exc.ConnectTimeout)):
            return "timeout"
    except ImportError:
        pass

    msg = str(error)
    if _SPARK_TASK_PATTERN.search(msg):
        return "spark_task_failed"
    if _SPARK_DRIVER_PATTERN.search(msg):
        return "spark_driver_error"
    if _LINEAGE_PATTERN.search(msg):
        return "lineage_emission_failed"
    if _TELEMETRY_PATTERN.search(msg):
        return "telemetry_unavailable"

    return "runtime_error"
