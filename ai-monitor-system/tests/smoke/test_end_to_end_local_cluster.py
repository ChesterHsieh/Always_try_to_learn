from pathlib import Path


def test_end_to_end_local_cluster_assets_ready() -> None:
    required_paths = [
        "deploy/scripts/bootstrap-local.sh",
        "deploy/scripts/run-pipeline.sh",
        "deploy/scripts/check-monitoring-coverage.sh",
        "deploy/scripts/run-smoke-test.sh",
        "deploy/helm/values.local-minimal.yaml",
    ]
    for rel_path in required_paths:
        assert Path(rel_path).exists(), f"missing required path: {rel_path}"
