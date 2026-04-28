#!/usr/bin/env python3
"""Observability probes for the ai-monitor-system stack.

Each subcommand performs a real HTTP request against a live monitoring
component (Prometheus, OTel Collector / trace backend, Marquez, Grafana,
Alertmanager) and emits a single JSON object on stdout describing the
verdict. Exit code is 0 on PASS, 1 on FAIL, 2 on probe-internal error.

Design rules (see CLAUDE.md / phase-2 design):
  - One probe = one assertion. No shell pipes, no jq.
  - Output is always a single-line JSON object on stdout.
  - On failure, include a `hint` field pointing the next investigation step.
  - Built-in polling/timeout via `--within` so callers never write retry loops.
  - No external deps beyond `requests` (already in pyproject deps tree).

Usage examples:
  probe prom-query 'pipeline_records_processed_total' --gte 1 --within 60
  probe prom-query 'up{job="otel-collector"}' --eq 1
  probe otel-trace --service pyspark-pipeline --has-attr run_id --within 30

Verdict JSON schema:
  {
    "probe":   "<subcommand>",
    "verdict": "PASS" | "FAIL" | "ERROR",
    "actual":  <value or null>,
    "expected":"<human readable>",
    "latency_ms": <int>,
    "hint":    "<optional, only on FAIL/ERROR>"
  }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    import requests
except ImportError:
    print(
        json.dumps(
            {
                "probe": "init",
                "verdict": "ERROR",
                "hint": "requests not installed; run `uv add requests` in ai-monitor-system",
            }
        )
    )
    sys.exit(2)


DEFAULT_PROM_URL = "http://localhost:9090"
DEFAULT_TRACE_URL = "http://localhost:16686"  # Jaeger query API; OTel collector exporters often land here
DEFAULT_POLL_INTERVAL = 2.0


@dataclass
class Verdict:
    probe: str
    verdict: str
    expected: str = ""
    actual: Any = None
    latency_ms: int = 0
    hint: str = ""
    extra: dict = field(default_factory=dict)
    terse: bool = False

    def emit(self) -> int:
        if self.terse:
            payload = {"probe": self.probe, "verdict": self.verdict}
            if self.verdict != "PASS" and self.hint:
                payload["hint"] = self.hint
            if self.verdict == "PASS" and isinstance(self.actual, (int, float, str, bool)):
                payload["actual"] = self.actual
        else:
            payload = {
                k: v for k, v in asdict(self).items()
                if k not in {"extra", "terse"} and (v != "" or k in {"actual"})
            }
            payload.update(self.extra)
        print(json.dumps(payload, default=str))
        return 0 if self.verdict == "PASS" else 1 if self.verdict == "FAIL" else 2


def poll_until(
    fn,
    *,
    within_s: float,
    interval_s: float = DEFAULT_POLL_INTERVAL,
):
    """Call `fn()` until it returns a truthy (passed, value) tuple, or timeout.

    `fn` must return a tuple `(passed: bool, value: Any)`. We return the last
    `(passed, value, elapsed_ms)`.
    """
    start = time.monotonic()
    deadline = start + within_s
    last_value = None
    while True:
        passed, value = fn()
        last_value = value
        if passed:
            return True, value, int((time.monotonic() - start) * 1000)
        if time.monotonic() >= deadline:
            return False, last_value, int((time.monotonic() - start) * 1000)
        time.sleep(min(interval_s, max(0.1, deadline - time.monotonic())))


# ---------------------------------------------------------------------------
# prom-query
# ---------------------------------------------------------------------------

def _eval_assertion(value: float | None, *, gte, lte, eq, exists: bool) -> tuple[bool, str]:
    """Return (passed, expected_human)."""
    if exists:
        return value is not None, "exists"
    if value is None:
        # numeric assertions all fail when no sample
        if gte is not None:
            return False, f">= {gte}"
        if lte is not None:
            return False, f"<= {lte}"
        if eq is not None:
            return False, f"== {eq}"
        return False, "value present"
    if gte is not None:
        return value >= gte, f">= {gte}"
    if lte is not None:
        return value <= lte, f"<= {lte}"
    if eq is not None:
        return value == eq, f"== {eq}"
    # No assertion provided: existence implies pass
    return True, "exists"


def cmd_prom_query(args) -> int:
    url = args.prom_url.rstrip("/") + "/api/v1/query"

    def attempt():
        try:
            resp = requests.get(url, params={"query": args.query}, timeout=args.timeout)
        except requests.RequestException as e:
            return False, {"error": f"request failed: {e}"}
        if resp.status_code != 200:
            return False, {"error": f"http {resp.status_code}", "body": resp.text[:200]}
        body = resp.json()
        if body.get("status") != "success":
            return False, {"error": "prometheus reported non-success", "body": body}
        results = body.get("data", {}).get("result", [])
        if not results:
            return False, None
        # Take first sample's value [timestamp, "value-as-string"]
        try:
            value = float(results[0]["value"][1])
        except (KeyError, IndexError, ValueError):
            return False, {"error": "unexpected result shape", "body": results[:1]}
        passed, _ = _eval_assertion(
            value, gte=args.gte, lte=args.lte, eq=args.eq, exists=args.exists
        )
        return passed, value

    passed, value, elapsed_ms = poll_until(attempt, within_s=args.within)
    _, expected_human = _eval_assertion(
        value if isinstance(value, (int, float)) else None,
        gte=args.gte, lte=args.lte, eq=args.eq, exists=args.exists,
    )

    v = Verdict(probe="prom-query", verdict="PASS" if passed else "FAIL",
                expected=expected_human, actual=value, latency_ms=elapsed_ms,
                terse=args.terse)
    v.extra["query"] = args.query
    if not passed:
        if value is None:
            v.hint = "no samples — check pipeline emit + Prometheus scrape + metric name"
        elif isinstance(value, dict) and "error" in value:
            v.hint = f"prometheus unreachable at {args.prom_url}; check /-/healthy"
        else:
            v.hint = f"metric present but failed {expected_human}"
    return v.emit()


# ---------------------------------------------------------------------------
# otel-trace (Jaeger query API; works with OTel collector → Jaeger exporter)
# ---------------------------------------------------------------------------

def cmd_otel_trace(args) -> int:
    """Query Jaeger HTTP API for a recent trace from the given service.

    Endpoint: GET /api/traces?service=<name>&limit=20&lookback=<within>m
    """
    url = args.trace_url.rstrip("/") + "/api/traces"

    def attempt():
        params = {"service": args.service, "limit": 20, "lookback": "1h"}
        try:
            resp = requests.get(url, params=params, timeout=args.timeout)
        except requests.RequestException as e:
            return False, {"error": f"request failed: {e}"}
        if resp.status_code != 200:
            return False, {"error": f"http {resp.status_code}", "body": resp.text[:200]}
        traces = resp.json().get("data", []) or []
        if not traces:
            return False, {"trace_count": 0}

        # Flatten span attribute keys across all returned traces
        seen_attrs: set[str] = set()
        for trace in traces:
            for span in trace.get("spans", []) or []:
                for tag in span.get("tags", []) or []:
                    k = tag.get("key")
                    if k:
                        seen_attrs.add(k)
                # OTel-style attributes sometimes nested under "process" tags
            for proc in (trace.get("processes") or {}).values():
                for tag in proc.get("tags", []) or []:
                    k = tag.get("key")
                    if k:
                        seen_attrs.add(k)

        missing = [a for a in args.has_attr if a not in seen_attrs]
        passed = not missing
        return passed, {
            "trace_count": len(traces),
            "attrs_found": sorted(seen_attrs),
            "missing_attrs": missing,
        }

    passed, value, elapsed_ms = poll_until(attempt, within_s=args.within)
    expected = (
        f"trace from service={args.service} with attrs={args.has_attr}"
        if args.has_attr
        else f"any trace from service={args.service}"
    )
    v = Verdict(probe="otel-trace", verdict="PASS" if passed else "FAIL",
                expected=expected, actual=value, latency_ms=elapsed_ms,
                terse=args.terse)
    v.extra["service"] = args.service
    if not passed:
        if isinstance(value, dict) and value.get("error"):
            v.hint = f"trace backend unreachable at {args.trace_url}"
        elif isinstance(value, dict) and value.get("trace_count") == 0:
            v.hint = f"no spans for service={args.service} — check OTEL endpoint + collector pipeline"
        elif isinstance(value, dict) and value.get("missing_attrs"):
            v.hint = f"missing span attrs {value['missing_attrs']} — check tracing.py set_attribute calls"
    return v.emit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--terse", action="store_true",
                   help="emit minimal JSON (probe+verdict+hint only); use for subagents/scenarios")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="probe", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prom-query", help="Query Prometheus and assert on result")
    pp.add_argument("query", help="PromQL query")
    pp.add_argument("--prom-url", default=DEFAULT_PROM_URL)
    g = pp.add_mutually_exclusive_group()
    g.add_argument("--gte", type=float, help="assert sample >= value")
    g.add_argument("--lte", type=float, help="assert sample <= value")
    g.add_argument("--eq", type=float, help="assert sample == value")
    g.add_argument("--exists", action="store_true", help="assert any sample present")
    pp.add_argument("--within", type=float, default=30.0, help="poll budget seconds")
    pp.add_argument("--timeout", type=float, default=5.0, help="per-request timeout")
    _add_common(pp)
    pp.set_defaults(func=cmd_prom_query)

    op = sub.add_parser("otel-trace", help="Verify traces present in Jaeger-compatible backend")
    op.add_argument("--service", required=True, help="OTel service.name to look for")
    op.add_argument("--has-attr", action="append", default=[],
                    help="repeatable; assert this span/process attribute key exists")
    op.add_argument("--trace-url", default=DEFAULT_TRACE_URL,
                    help="Jaeger query API base (e.g. http://localhost:16686)")
    op.add_argument("--within", type=float, default=30.0, help="poll budget seconds")
    op.add_argument("--timeout", type=float, default=5.0)
    _add_common(op)
    op.set_defaults(func=cmd_otel_trace)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # noqa: BLE001 — last-resort to keep JSON contract
        print(json.dumps({
            "probe": getattr(args, "cmd", "unknown"),
            "verdict": "ERROR",
            "hint": f"probe crashed: {type(e).__name__}: {e}",
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
