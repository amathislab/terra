"""Source-distribution boundary tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_sdist_and_wheel_include_runtime_files(tmp_path: Path) -> None:
    """Build from a source copy and check packaged runtime files."""

    source = tmp_path / "source"
    source.mkdir()
    for filename in (
        "pyproject.toml",
        "MANIFEST.in",
        "uv.lock",
        "README.md",
        "LICENSE",
        "terra_build_backend.py",
        ".python-version",
        ".git_archival.txt",
        ".gitattributes",
    ):
        shutil.copy2(REPO / filename, source / filename)
    for directory in ("src", "scripts", "docs", "tests"):
        shutil.copytree(REPO / directory, source / directory)

    environment = dict(os.environ)
    environment["TERRA_GIT_COMMIT"] = "c" * 40
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from terra_build_backend import build_sdist; build_sdist('dist')",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    archives = list((source / "dist").glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        members = {Path(name).as_posix() for name in archive.getnames()}

    assert any(name.endswith("/src/terra/datasets/prism/adapter.py") for name in members)
    assert any(name.endswith("/src/terra/benchmarking/reconstruction/methods/terra.py") for name in members)
    assert any(name.endswith("/src/terra/_build_commit.txt") for name in members)
    assert any(name.endswith("/src/terra/rl/configs/ppo_multi_motion.yaml") for name in members)
    assert any(name.endswith("/terra_build_backend.py") for name in members)
    assert any(name.endswith("/scripts/terra/train_smoke.py") for name in members)
    assert any(name.endswith("/scripts/terra/ppo_smoke_overrides.json") for name in members)
    assert any(name.endswith("/uv.lock") for name in members)

    stale_build_file = source / "build/lib/terra/removed_module.py"
    stale_build_file.parent.mkdir(parents=True, exist_ok=True)
    stale_build_file.write_text("raise RuntimeError('stale build file')\n")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from terra_build_backend import build_wheel; build_wheel('wheel')",
        ],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    wheels = list((source / "wheel").glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_members = set(wheel.namelist())
        assert wheel.read("terra/_build_commit.txt") == b"cccccccccccccccccccccccccccccccccccccccc\n"
    assert "terra/removed_module.py" not in wheel_members
    assert "terra/rl/configs/ppo_multi_motion.yaml" in wheel_members
