"""Smoke test conftest: lazy cluster availability check."""

import subprocess

import pytest


def _check_cluster() -> bool:
    try:
        # Use a very short timeout; kubectl can hang waiting for unreachable cluster
        result = subprocess.run(
            ["kubectl", "get", "nodes", "--request-timeout=2s"],
            capture_output=True,
            timeout=4,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


_CLUSTER_AVAILABLE: bool | None = None


def _cluster_available() -> bool:
    global _CLUSTER_AVAILABLE
    if _CLUSTER_AVAILABLE is None:
        _CLUSTER_AVAILABLE = _check_cluster()
    return _CLUSTER_AVAILABLE


@pytest.fixture()
def require_cluster():
    if not _cluster_available():
        pytest.skip("No local Kubernetes cluster available")
