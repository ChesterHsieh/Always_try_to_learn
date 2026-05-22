"""Integration tests: failure injection wired into pipeline/job.py (Task 2.2).

These tests verify that setting INJECT_FAILURE=<category> causes job.run_pipeline
to fail with the correct failure_category in its lifecycle metric payload.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pipeline.job as job_module
from pipeline.failure_classifier import KNOWN_CATEGORIES
from utils.io_adapter import write_local_file


@pytest.fixture(autouse=True)
def skip_metrics_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METRICS_SKIP_SERVER", "1")


def _make_spark_mock(tmp_output_path: str) -> MagicMock:
    def fake_write_text(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "part-00000-fake.txt").write_text("HELLO\n")

    df_mock = MagicMock()
    df_mock.write.mode.return_value.text = fake_write_text
    df_mock.select.return_value = df_mock

    spark_mock = MagicMock()
    spark_mock.read.text.return_value = df_mock
    return spark_mock


@pytest.mark.parametrize("category", sorted(KNOWN_CATEGORIES))
def test_inject_failure_env_produces_correct_category(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    category: str,
) -> None:
    """Setting INJECT_FAILURE=<category> must result in payload.failure_category == category."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", category)

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        payload = job_module.run_pipeline(str(src), str(dst))

    assert payload["status"] == "failed", f"Expected failed for {category!r}, got {payload}"
    assert payload["failure_category"] == category, (
        f"Expected failure_category={category!r}, got {payload.get('failure_category')!r}"
    )


def test_inject_failure_none_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """INJECT_FAILURE=none (or unset) must still result in a successful run."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.setenv("INJECT_FAILURE", "none")

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        payload = job_module.run_pipeline(str(src), str(dst))

    assert payload["status"] == "succeeded", f"Expected succeeded, got {payload}"


def test_inject_failure_unset_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When INJECT_FAILURE is not set, the pipeline must succeed normally."""
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")
    monkeypatch.delenv("INJECT_FAILURE", raising=False)

    spark_mock = _make_spark_mock(str(dst))

    with (
        patch.object(job_module, "create_spark_session", return_value=spark_mock),
        patch("pipeline.job.F") as mock_f,
    ):
        mock_f.upper.return_value = MagicMock()
        mock_f.col.return_value = MagicMock()
        payload = job_module.run_pipeline(str(src), str(dst))

    assert payload["status"] == "succeeded", f"Expected succeeded, got {payload}"
