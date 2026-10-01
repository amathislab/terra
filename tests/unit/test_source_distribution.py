"""Source-distribution boundary tests."""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_sdist_and_wheel_include_runtime_files(tmp_path: Path, release_source: Path) -> None:
    """Build from a source copy and check packaged runtime files."""

    source = release_source

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_sdist; build_sdist('dist')",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
    )
    archives = list((source / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        members = {Path(name).as_posix() for name in archive.getnames()}

    assert any(name.endswith("/src/terra/datasets/prism/adapter.py") for name in members)
    assert any(name.endswith("/src/terra/benchmarking/reconstruction/methods/terra.py") for name in members)
    assert any(name.endswith("/src/terra/rl/configs/ppo_multi_motion.yaml") for name in members)
    assert any(name.endswith("/scripts/terra/train_smoke.py") for name in members)
    assert any(name.endswith("/scripts/terra/ppo_smoke_overrides.json") for name in members)
    assert any(name.endswith("/uv.lock") for name in members)

    unpacked = tmp_path / "unpacked"
    with tarfile.open(archives[0], "r:gz") as archive:
        archive.extractall(unpacked, filter="data")
    (distribution_source,) = unpacked.iterdir()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; build_wheel('wheel')",
        ],
        cwd=distribution_source,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = list((distribution_source / "wheel").glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_members = set(wheel.namelist())
    assert "terra/rl/configs/ppo_multi_motion.yaml" in wheel_members
