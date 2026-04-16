from pathlib import Path


def test_local_profile_exists() -> None:
    assert Path("deploy/k8s/local-profile.yaml").exists()


def test_local_minimal_values_exists_for_option_b() -> None:
    assert Path("deploy/helm/values.local-minimal.yaml").exists()
