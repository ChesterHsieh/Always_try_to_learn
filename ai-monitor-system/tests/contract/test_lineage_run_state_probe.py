"""Contract tests: lineage-run-state probe subcommand (Task 3.2).

Verifies the two-state (PASS / FAIL only) assertion contract against a mock
lineage backend. No SKIP state; backend unavailability = FAIL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add scripts/ to sys.path so we can import probe without installation
_SCRIPTS_DIR = str(Path(__file__).parent.parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import probe as probe_module  # noqa: E402


def _run_lineage_probe(argv: list[str]) -> tuple[int, dict]:
    parser = probe_module.build_parser()
    args = parser.parse_args(argv)
    captured: list[str] = []
    with patch("builtins.print", side_effect=captured.append):
        rc = args.func(args)
    return rc, json.loads(captured[0]) if captured else {}


def _mock_response(json_body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = str(json_body)
    return resp


# ---------------------------------------------------------------------------
# PASS cases
# ---------------------------------------------------------------------------


def test_lineage_run_state_pass_when_state_failed() -> None:
    """Returns PASS when lineage backend reports run state = FAILED."""
    resp = _mock_response({"run": {"runFacets": {}, "state": "FAILED"}})
    with patch("requests.get", return_value=resp):
        rc, verdict = _run_lineage_probe(
            [
                "lineage-run-state",
                "--run-id",
                "abc-123",
                "--state-eq",
                "FAILED",
                "--within",
                "1",
            ]
        )
    assert rc == 0, f"Expected PASS, got rc={rc}, verdict={verdict}"
    assert verdict["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# FAIL cases
# ---------------------------------------------------------------------------


def test_lineage_run_state_fail_when_state_running() -> None:
    """Returns FAIL when run state is RUNNING (not yet terminal)."""
    resp = _mock_response({"run": {"state": "RUNNING"}})
    with patch("requests.get", return_value=resp):
        rc, verdict = _run_lineage_probe(
            [
                "lineage-run-state",
                "--run-id",
                "abc-123",
                "--state-eq",
                "FAILED",
                "--within",
                "1",
            ]
        )
    assert rc == 1, f"Expected FAIL (rc=1), got rc={rc}"
    assert verdict["verdict"] == "FAIL"


def test_lineage_run_state_fail_on_404() -> None:
    """Returns FAIL (not ERROR or SKIP) when backend returns 404."""
    resp = _mock_response({}, status_code=404)
    with patch("requests.get", return_value=resp):
        rc, verdict = _run_lineage_probe(
            [
                "lineage-run-state",
                "--run-id",
                "nonexistent-run",
                "--state-eq",
                "FAILED",
                "--within",
                "1",
            ]
        )
    assert rc == 1, f"Expected FAIL (rc=1), got rc={rc}"
    assert verdict["verdict"] == "FAIL"
    assert "hint" in verdict


def test_lineage_run_state_fail_when_backend_unreachable() -> None:
    """Backend connection error produces FAIL, not SKIP."""
    import requests as req_lib

    with patch("requests.get", side_effect=req_lib.ConnectionError("connection refused")):
        rc, verdict = _run_lineage_probe(
            [
                "lineage-run-state",
                "--run-id",
                "abc-123",
                "--state-eq",
                "FAILED",
                "--within",
                "1",
            ]
        )
    assert rc == 1, f"Expected FAIL (rc=1), got rc={rc}"
    assert verdict["verdict"] == "FAIL"
    assert "hint" in verdict


# ---------------------------------------------------------------------------
# No SKIP state
# ---------------------------------------------------------------------------


def test_lineage_run_state_verdict_never_skip() -> None:
    """Probe must never emit SKIP — only PASS or FAIL are allowed verdicts."""
    # Simulate backend returning unexpected body
    resp = _mock_response({"unexpected": "body"})
    with patch("requests.get", return_value=resp):
        rc, verdict = _run_lineage_probe(
            [
                "lineage-run-state",
                "--run-id",
                "abc-123",
                "--state-eq",
                "FAILED",
                "--within",
                "1",
            ]
        )
    assert verdict.get("verdict") in ("PASS", "FAIL"), (
        f"Expected PASS or FAIL, got {verdict.get('verdict')!r}"
    )
