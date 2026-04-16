import uuid


def build_trace_attributes(run_id: str, pipeline_name: str, namespace: str, status: str) -> dict[str, str]:
    return {
        "trace_id": str(uuid.uuid4()),
        "run_id": run_id,
        "pipeline_name": pipeline_name,
        "k8s_namespace": namespace,
        "status": status,
    }
