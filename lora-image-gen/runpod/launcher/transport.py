"""從本機透過直接 SSH 操作遠端 pod：上傳資料夾、輪詢完成標記。

RunPod 的 proxy SSH 不支援 scp/rsync，runpodctl 也沒有可遠端觸發的傳檔 / 任意
shell exec，故採直接 SSH（pod 須開 22 port + 註冊本機 public key）。連線資訊
（ip/port）由 RunpodClient.extract_endpoints 解析出的 ssh_command 提供。
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 形如 "ssh root@1.2.3.4 -p 12345"
_SSH_RE = re.compile(r"ssh\s+(?P<user>[^@]+)@(?P<host>\S+)\s+-p\s+(?P<port>\d+)")


class TransportError(Exception):
    """SSH 連線資訊無效，或遠端操作失敗。"""


@dataclass(frozen=True)
class SshTarget:
    user: str
    host: str
    port: int
    key_path: Path | None = None

    @classmethod
    def from_ssh_command(cls, ssh_command: str | None, key_path: Path | None = None) -> "SshTarget":
        if not ssh_command:
            raise TransportError("pod 未提供直連 SSH 資訊（需開 22 port 直連 SSH）")
        m = _SSH_RE.search(ssh_command)
        if not m:
            raise TransportError(f"無法解析 SSH 連線指令：{ssh_command!r}")
        return cls(user=m["user"], host=m["host"], port=int(m["port"]), key_path=key_path)

    def _ssh_opts(self) -> list[str]:
        opts = ["-p", str(self.port), "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if self.key_path:
            opts += ["-i", str(self.key_path)]
        return opts


@dataclass
class SshTransport:
    """本機 SSH 操作 pod 的實作；以注入的 runner 執行子行程，方便測試。"""

    target: SshTarget
    runner: "Runner" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.runner is None:
            self.runner = _subprocess_runner

    def upload(self, source: Path, dest: str) -> bool:
        """rsync 本機資料夾到 pod 的 dest，回傳是否成功。"""
        ssh = "ssh " + " ".join(self.target._ssh_opts())
        # 先確保目的目錄存在
        if self._remote(f"mkdir -p {dest}").returncode != 0:
            return False
        cmd = [
            "rsync", "-az", "-e", ssh,
            f"{str(source).rstrip('/')}/",
            f"{self.target.user}@{self.target.host}:{dest}",
        ]
        return self.runner(cmd).returncode == 0

    def marker_status(self, concept: str) -> str | None:
        """查 pod 上的完成標記：done / failed / None（尚在進行）。"""
        base = f"/workspace/training/{concept}"
        if self._remote(f"test -f {base}.run.done").returncode == 0:
            return "done"
        if self._remote(f"test -f {base}.run.failed").returncode == 0:
            return "failed"
        return None

    def _remote(self, remote_cmd: str) -> subprocess.CompletedProcess:
        cmd = ["ssh", *self.target._ssh_opts(),
               f"{self.target.user}@{self.target.host}", remote_cmd]
        return self.runner(cmd)


# 子行程執行器：吃 argv，回傳 CompletedProcess（可注入假物件做測試）。
from collections.abc import Callable  # noqa: E402

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _subprocess_runner(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)
