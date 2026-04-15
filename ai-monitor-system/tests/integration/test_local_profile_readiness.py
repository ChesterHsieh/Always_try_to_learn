from pathlib import Path


def test_local_profile_exists() -> None:
    assert Path("deploy/k8s/local-profile.yaml").exists()
