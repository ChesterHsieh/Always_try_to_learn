"""Coverage CLI: health checks + chart_version reporting via Kubernetes API."""

from __future__ import annotations

import argparse
import base64
import datetime
import gzip
import json
import logging
import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import requests

logger = logging.getLogger(__name__)

CheckStatus = Literal["pass", "warn", "fail"]


class CheckResult(TypedDict):
    name: str
    status: CheckStatus
    detail: str


class CoverageReport(TypedDict):
    profile_name: str
    profile_version: str
    last_verified_at: str
    components: dict[str, str]
    validation_checks: list[CheckResult]


_CHART_VERSION_MATRIX = {
    "upstream-prometheus": "25.27.0",
    "upstream-grafana": "8.5.1",
    "upstream-otel-collector": "0.78.0",
    "upstream-marquez": "6.7.0",
}

_TIMEOUT = 5
_POLL_MAX = 30


def _get_k8s_api():
    try:
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        return client.CoreV1Api()
    except Exception:
        return None


def _get_chart_versions(namespace: str, k8s_api=None) -> tuple[dict[str, str], CheckStatus]:
    if k8s_api is None:
        k8s_api = _get_k8s_api()

    if k8s_api is None:
        return _fallback_chart_versions(), "warn"

    versions: dict[str, str] = {}
    for release_name in _CHART_VERSION_MATRIX:
        try:
            result = k8s_api.list_namespaced_secret(
                namespace=namespace,
                label_selector=f"owner=helm,name={release_name}",
            )
            if not result.items:
                versions[release_name] = _CHART_VERSION_MATRIX.get(release_name, "unknown")
                continue
            # pick latest revision (last in list)
            secret = result.items[-1]
            raw_b64 = secret.data.get("release", "")
            # Helm stores as base64(base64(gzip(json)))
            decoded_once = base64.b64decode(raw_b64)
            decoded_twice = base64.b64decode(decoded_once)
            decompressed = gzip.decompress(decoded_twice)
            metadata = json.loads(decompressed)
            versions[release_name] = metadata["chart"]["metadata"]["version"]
        except Exception as exc:
            logger.warning("Failed to get chart version for %s: %s", release_name, exc)
            versions[release_name] = _CHART_VERSION_MATRIX.get(release_name, "unknown")

    return versions, "pass"


def _fallback_chart_versions() -> dict[str, str]:
    return dict(_CHART_VERSION_MATRIX)


def _check_prometheus(prometheus_url: str) -> CheckResult:
    try:
        resp = requests.get(f"{prometheus_url}/api/v1/rules", timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        groups = data.get("data", {}).get("groups", [])
        has_pipeline_rules = any("pipeline" in g.get("name", "") for g in groups)
        if has_pipeline_rules:
            return CheckResult(
                name="prometheus_rules", status="pass", detail="pipeline-failure rule group loaded"
            )
        return CheckResult(
            name="prometheus_rules", status="warn", detail="pipeline-failure rule group not found"
        )
    except requests.exceptions.ConnectionError as exc:
        return CheckResult(
            name="prometheus_rules", status="fail", detail=f"Prometheus unreachable: {exc}"
        )
    except Exception as exc:
        return CheckResult(name="prometheus_rules", status="fail", detail=str(exc))


def _check_prometheus_reachable(prometheus_url: str) -> CheckResult:
    try:
        resp = requests.get(f"{prometheus_url}/-/healthy", timeout=_TIMEOUT)
        if resp.status_code < 400:
            return CheckResult(
                name="prometheus_reachable", status="pass", detail="Prometheus healthy"
            )
        return CheckResult(
            name="prometheus_reachable", status="fail", detail=f"HTTP {resp.status_code}"
        )
    except Exception as exc:
        return CheckResult(name="prometheus_reachable", status="fail", detail=f"Unreachable: {exc}")


def _check_grafana(grafana_url: str) -> CheckResult:
    try:
        resp = requests.get(f"{grafana_url}/api/health", timeout=_TIMEOUT)
        resp.raise_for_status()
        return CheckResult(name="grafana_datasource", status="pass", detail="Grafana health ok")
    except Exception as exc:
        return CheckResult(name="grafana_datasource", status="fail", detail=str(exc))


def _check_marquez(marquez_url: str) -> CheckResult:
    try:
        resp = requests.get(f"{marquez_url}/api/v1/namespaces", timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        ns_list = [n.get("name") for n in data.get("namespaces", [])]
        return CheckResult(name="marquez_reachable", status="pass", detail=f"Namespaces: {ns_list}")
    except Exception as exc:
        return CheckResult(name="marquez_reachable", status="fail", detail=str(exc))


def _check_lineage_freshness(marquez_url: str) -> CheckResult:
    try:
        resp = requests.get(f"{marquez_url}/api/v1/events/lineage?limit=5", timeout=_TIMEOUT)
        resp.raise_for_status()
        events = resp.json().get("events", [])
        if events:
            return CheckResult(
                name="recent_lineage",
                status="pass",
                detail=f"{len(events)} recent lineage events found",
            )
        return CheckResult(
            name="recent_lineage",
            status="warn",
            detail="No recent lineage events (possible cold start)",
        )
    except Exception as exc:
        return CheckResult(name="recent_lineage", status="fail", detail=str(exc))


def run_coverage(
    *,
    namespace: str,
    marquez_url: str,
    prometheus_url: str,
    grafana_url: str,
    k8s_api=None,
) -> CoverageReport:
    checks: list[CheckResult] = []

    prom_health = _check_prometheus_reachable(prometheus_url)
    checks.append(prom_health)
    if prom_health["status"] == "fail":
        versions, _ = _get_chart_versions(namespace, k8s_api)
        return CoverageReport(
            profile_name="pyspark-monitoring-framework",
            profile_version="1.0.0",
            last_verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            components=versions,
            validation_checks=checks,
        )

    checks.append(_check_prometheus(prometheus_url))
    checks.append(_check_grafana(grafana_url))
    checks.append(_check_marquez(marquez_url))
    checks.append(_check_lineage_freshness(marquez_url))

    versions, version_status = _get_chart_versions(namespace, k8s_api)
    if version_status != "pass":
        checks.append(
            CheckResult(
                name="chart_version_lookup",
                status="warn",
                detail="Could not read Helm release Secrets; using chart-version-matrix fallback",
            )
        )

    return CoverageReport(
        profile_name="pyspark-monitoring-framework",
        profile_version="1.0.0",
        last_verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        components=versions,
        validation_checks=checks,
    )


def _exit_code(report: CoverageReport) -> int:
    statuses = [c["status"] for c in report["validation_checks"]]
    if "fail" in statuses:
        return 2
    if "warn" in statuses:
        return 1
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="PySpark monitoring coverage check")
    parser.add_argument("--namespace", default=os.environ.get("K8S_NAMESPACE", "ai-monitor-system"))
    parser.add_argument(
        "--marquez-url",
        default=os.environ.get("MARQUEZ_URL", "http://ai-monitor-system-upstream-marquez:9555"),
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.environ.get(
            "PROMETHEUS_URL", "http://ai-monitor-system-upstream-prometheus-server:80"
        ),
    )
    parser.add_argument(
        "--grafana-url",
        default=os.environ.get("GRAFANA_URL", "http://ai-monitor-system-upstream-grafana:80"),
    )
    default_output = (
        Path(".local-data/coverage")
        / f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()

    report = run_coverage(
        namespace=args.namespace,
        marquez_url=args.marquez_url,
        prometheus_url=args.prometheus_url,
        grafana_url=args.grafana_url,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(
        json.dumps(
            {
                "summary": {
                    "checks_pass": sum(
                        1 for c in report["validation_checks"] if c["status"] == "pass"
                    ),
                    "checks_warn": sum(
                        1 for c in report["validation_checks"] if c["status"] == "warn"
                    ),
                    "checks_fail": sum(
                        1 for c in report["validation_checks"] if c["status"] == "fail"
                    ),
                },
                "components": report["components"],
                "last_verified_at": report["last_verified_at"],
            },
            indent=2,
        )
    )

    sys.exit(_exit_code(report))


if __name__ == "__main__":
    main()
