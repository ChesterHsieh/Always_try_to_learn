from pathlib import Path

from pipeline.io_adapter import write_local_file
import pipeline.job as job_module


def test_run_status_flow(tmp_path: Path) -> None:
    job_module.create_spark_session = lambda: None
    src = tmp_path / "in.txt"
    dst = tmp_path / "out.txt"
    write_local_file(str(src), "hello")
    payload = job_module.run_pipeline(str(src), str(dst))
    assert payload["status"] == "succeeded"
    assert dst.exists()


def test_run_status_flow_failed_when_input_missing(tmp_path: Path) -> None:
    job_module.create_spark_session = lambda: None
    src = tmp_path / "missing.txt"
    dst = tmp_path / "out.txt"
    payload = job_module.run_pipeline(str(src), str(dst))
    assert payload["status"] == "failed"
    assert payload["failure_category"] == "input_not_found"
