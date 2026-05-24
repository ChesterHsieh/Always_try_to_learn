"""RunPod SDK 薄封裝測試（task 8.3），以假 API 驗證 pod 生命週期。"""
from __future__ import annotations

import pytest

from launcher.runpod_client import (
    COMFY_PORT,
    VOLUME_MOUNT_PATH,
    PodError,
    RunpodClient,
)


class FakeApi:
    """記錄呼叫參數的假 RunPod API。"""

    def __init__(self, *, create_result=None, pod_states=None):
        self.create_result = create_result if create_result is not None else {"id": "pod-1"}
        self.pod_states = list(pod_states or [])
        self.create_kwargs = None
        self.terminated: list[str] = []

    def create_pod(self, **kwargs):
        self.create_kwargs = kwargs
        return self.create_result

    def get_pod(self, pod_id):
        if self.pod_states:
            return self.pod_states.pop(0)
        return {"id": pod_id, "runtime": {"ports": []}}

    def terminate_pod(self, pod_id):
        self.terminated.append(pod_id)


@pytest.mark.unit
def test_create_pod_passes_volume_and_ports() -> None:
    api = FakeApi()
    client = RunpodClient(api=api)
    handle = client.create_training_pod(
        name="lora-train-x", image="img", gpu_type="A6000",
        network_volume_id="vol-1", data_center_id="EU-RO-1",
        container_disk_gb=30, env={"K": "V"}, start_command="bash run.sh",
    )
    assert handle.pod_id == "pod-1"
    kw = api.create_kwargs
    assert kw["network_volume_id"] == "vol-1"
    assert kw["volume_mount_path"] == VOLUME_MOUNT_PATH
    assert kw["cloud_type"] == "SECURE"
    assert f"{COMFY_PORT}/http" in kw["ports"]
    assert "22/tcp" in kw["ports"]
    assert kw["docker_args"] == "bash run.sh"


@pytest.mark.unit
def test_create_pod_without_id_raises() -> None:
    client = RunpodClient(api=FakeApi(create_result={"error": "no capacity"}))
    with pytest.raises(PodError, match="未含 pod id"):
        client.create_training_pod(
            name="x", image="i", gpu_type="g", network_volume_id="v",
            data_center_id="d", container_disk_gb=10, env={}, start_command="c",
        )


@pytest.mark.unit
def test_wait_until_ready_polls_then_returns() -> None:
    api = FakeApi(pod_states=[
        {"id": "pod-1", "runtime": None},
        {"id": "pod-1", "runtime": {"ports": []}},
    ])
    slept: list[float] = []
    client = RunpodClient(api=api, poll_interval=5)
    ready = client.wait_until_ready("pod-1", sleep=slept.append)
    assert ready["runtime"] is not None
    assert slept == [5]  # 第一次未就緒睡一輪


@pytest.mark.unit
def test_wait_until_ready_times_out() -> None:
    api = FakeApi(pod_states=[{"id": "pod-1", "runtime": None}] * 5)
    clock = iter([0, 0, 100, 100, 100])
    client = RunpodClient(api=api, poll_interval=1, ready_timeout=10)
    with pytest.raises(PodError, match="未就緒"):
        client.wait_until_ready("pod-1", now=lambda: next(clock), sleep=lambda _: None)


@pytest.mark.unit
def test_get_status_none_raises() -> None:
    class NoneApi(FakeApi):
        def get_pod(self, pod_id):
            return None

    with pytest.raises(PodError, match="查不到 pod"):
        RunpodClient(api=NoneApi()).get_status("pod-1")


@pytest.mark.unit
def test_extract_endpoints_builds_comfy_url_and_ssh() -> None:
    client = RunpodClient(api=FakeApi())
    pod = {
        "id": "pod-xyz",
        "runtime": {"ports": [
            {"privatePort": 22, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 12345},
            {"privatePort": COMFY_PORT, "isIpPublic": True, "ip": "1.2.3.4", "publicPort": 8188},
        ]},
    }
    ep = client.extract_endpoints(pod)
    assert ep.comfy_url == f"https://pod-xyz-{COMFY_PORT}.proxy.runpod.net"
    assert ep.ssh_command == "ssh root@1.2.3.4 -p 12345"


@pytest.mark.unit
def test_extract_endpoints_no_ssh_when_no_public_port() -> None:
    client = RunpodClient(api=FakeApi())
    pod = {"id": "pod-xyz", "runtime": {"ports": []}}
    ep = client.extract_endpoints(pod)
    assert ep.comfy_url is not None
    assert ep.ssh_command is None


@pytest.mark.unit
def test_terminate_calls_api() -> None:
    api = FakeApi()
    RunpodClient(api=api).terminate("pod-1")
    assert api.terminated == ["pod-1"]
