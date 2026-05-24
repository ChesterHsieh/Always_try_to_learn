"""直連 SSH transport 測試：解析連線、rsync 上傳、查完成標記。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from launcher.transport import SshTarget, SshTransport, TransportError


def _ok(cmd):
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _fail(cmd):
    return subprocess.CompletedProcess(cmd, 1, "", "")


@pytest.mark.unit
def test_parse_ssh_command() -> None:
    t = SshTarget.from_ssh_command("ssh root@1.2.3.4 -p 12345")
    assert (t.user, t.host, t.port) == ("root", "1.2.3.4", 12345)


@pytest.mark.unit
def test_parse_none_raises() -> None:
    with pytest.raises(TransportError, match="未提供直連 SSH"):
        SshTarget.from_ssh_command(None)


@pytest.mark.unit
def test_parse_garbage_raises() -> None:
    with pytest.raises(TransportError, match="無法解析"):
        SshTarget.from_ssh_command("connect via web terminal")


@pytest.mark.unit
def test_upload_success_runs_mkdir_then_rsync(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(cmd):
        calls.append(cmd)
        return _ok(cmd)

    t = SshTransport(SshTarget("root", "1.2.3.4", 22), runner=runner)
    assert t.upload(tmp_path, "/workspace/datasets/x/") is True
    # 第一個是遠端 mkdir，第二個是 rsync
    assert any("mkdir -p" in " ".join(c) for c in calls)
    assert any(c[0] == "rsync" for c in calls)


@pytest.mark.unit
def test_upload_mkdir_failure_returns_false(tmp_path: Path) -> None:
    t = SshTransport(SshTarget("root", "h", 22), runner=_fail)
    assert t.upload(tmp_path, "/dest") is False


@pytest.mark.unit
def test_upload_rsync_failure_returns_false(tmp_path: Path) -> None:
    seq = iter([_ok, _fail])  # mkdir 成功、rsync 失敗

    def runner(cmd):
        return next(seq)(cmd)

    t = SshTransport(SshTarget("root", "h", 22), runner=runner)
    assert t.upload(tmp_path, "/dest") is False


@pytest.mark.unit
def test_marker_status_done() -> None:
    def runner(cmd):
        return _ok(cmd) if "run.done" in " ".join(cmd) else _fail(cmd)

    t = SshTransport(SshTarget("root", "h", 22), runner=runner)
    assert t.marker_status("stcklnd") == "done"


@pytest.mark.unit
def test_marker_status_failed() -> None:
    def runner(cmd):
        return _ok(cmd) if "run.failed" in " ".join(cmd) else _fail(cmd)

    t = SshTransport(SshTarget("root", "h", 22), runner=runner)
    assert t.marker_status("stcklnd") == "failed"


@pytest.mark.unit
def test_marker_status_pending() -> None:
    t = SshTransport(SshTarget("root", "h", 22), runner=_fail)
    assert t.marker_status("stcklnd") is None


@pytest.mark.unit
def test_key_path_added_to_ssh_opts() -> None:
    t = SshTarget("root", "h", 22, key_path=Path("/k.pem"))
    opts = t._ssh_opts()
    assert "-i" in opts and "/k.pem" in opts
