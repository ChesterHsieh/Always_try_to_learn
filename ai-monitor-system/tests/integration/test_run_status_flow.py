from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import pipeline.job as job_module
from pipeline.io_adapter import write_local_file


@pytest.fixture(autouse=True)
def skip_metrics_server(monkeypatch):
    monkeypatch.setenv("METRICS_SKIP_SERVER", "1")


def test_run_status_flow(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello\n")

    # The mock spark write side-effect creates the expected output dir + part file
    def fake_write_text(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "part-00000-fake.txt").write_text("HELLO\n")

    # Build mock so that .read.text(…).select(…).write.mode(…).text(…) calls fake_write_text
    df_mock = MagicMock()
    df_mock.write.mode.return_value.text = fake_write_text
    df_mock.select.return_value = df_mock

    spark_mock = MagicMock()
    spark_mock.read.text.return_value = df_mock

    with patch.object(job_module, "create_spark_session", return_value=spark_mock):
        # Also patch F.upper and F.col so PySpark internals are not invoked
        with patch("pipeline.job.F") as mock_f:
            mock_f.upper.return_value = MagicMock()
            mock_f.col.return_value = MagicMock()
            payload = job_module.run_pipeline(str(src), str(dst))

    assert payload["status"] == "succeeded", f"Got: {payload}"
    assert dst.exists()


def test_run_status_flow_failed_when_input_missing(tmp_path: Path) -> None:
    src = tmp_path / "missing.txt"
    dst = tmp_path / "out.txt"

    spark_mock = MagicMock()
    spark_mock.read.text.side_effect = FileNotFoundError(f"No such file: {src}")

    with patch.object(job_module, "create_spark_session", return_value=spark_mock):
        payload = job_module.run_pipeline(str(src), str(dst))

    assert payload["status"] == "failed"
    assert payload["failure_category"] == "input_not_found"
