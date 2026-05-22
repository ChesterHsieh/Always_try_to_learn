"""Smoke test: nuke + rebuild cycle (Task 12.4). Requires local Kubernetes cluster."""

import json
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parent.parent.parent / "deploy" / "scripts"
NAMESPACE = "ai-monitor-system"
RELEASE_NAME = "monitor"


@pytest.mark.usefixtures("require_cluster")
def test_nuke_clears_namespace():
    result = subprocess.run(
        ["bash", str(SCRIPTS / "nuke-local.sh")],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "NAMESPACE": NAMESPACE, "RELEASE_NAME": RELEASE_NAME},
    )
    assert result.returncode == 0, f"nuke-local.sh failed: {result.stderr}"

    ns_result = subprocess.run(
        ["kubectl", "get", "ns", NAMESPACE],
        capture_output=True,
        text=True,
    )
    assert "NotFound" in ns_result.stderr or ns_result.returncode != 0

    pv_result = subprocess.run(
        ["kubectl", "get", "pv", "-o", "json"], capture_output=True, text=True
    )
    if pv_result.returncode == 0:
        pvs = json.loads(pv_result.stdout).get("items", [])
        lingering = [
            pv for pv in pvs if pv.get("spec", {}).get("claimRef", {}).get("namespace") == NAMESPACE
        ]
        assert not lingering, f"Lingering PVs: {[p['metadata']['name'] for p in lingering]}"


@pytest.mark.usefixtures("require_cluster")
def test_rebuild_after_nuke():
    result = subprocess.run(
        ["bash", str(SCRIPTS / "bootstrap-local.sh")],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "NUKE_BEFORE_BOOTSTRAP": "true",
            "NAMESPACE": NAMESPACE,
            "RELEASE_NAME": RELEASE_NAME,
        },
    )
    assert result.returncode == 0, f"bootstrap failed: {result.stderr}"

    helm_result = subprocess.run(
        ["helm", "list", "-n", NAMESPACE, "-o", "json"],
        capture_output=True,
        text=True,
    )
    assert helm_result.returncode == 0
    releases = json.loads(helm_result.stdout)
    deployed = {r["name"] for r in releases if r.get("status") == "deployed"}
    assert RELEASE_NAME in deployed, f"Release not deployed: {deployed}"
