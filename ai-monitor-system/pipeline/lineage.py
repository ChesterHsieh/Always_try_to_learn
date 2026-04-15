def build_openlineage_event(
    run_id: str,
    job_name: str,
    namespace: str,
    source_dataset: str,
    target_dataset: str,
) -> dict:
    return {
        "eventType": "COMPLETE",
        "run": {"runId": run_id},
        "job": {"namespace": namespace, "name": job_name},
        "inputs": [{"namespace": namespace, "name": source_dataset}],
        "outputs": [{"namespace": namespace, "name": target_dataset}],
    }
