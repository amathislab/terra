"""Release archives and fitting objectives must not depend on host state."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("enclosing_git", (False, True))
def test_git_archive_can_publish_revision_and_build_a_stamped_wheel(tmp_path: Path, enclosing_git: bool) -> None:
    checkout = tmp_path / "checkout"
    package = checkout / "src/terra"
    package.mkdir(parents=True)
    for filename in ("__init__.py", "_revision.py", "_files.py"):
        shutil.copy2(REPO / "src/terra" / filename, package / filename)
    for filename in (
        ".gitattributes",
        ".git_archival.txt",
        "terra_build_backend.py",
        "pyproject.toml",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
    ):
        shutil.copy2(REPO / filename, checkout / filename)
    # A released source archive already contains an expanded marker. This fresh
    # fixture has its own commit, so give Git a template to expand for that commit.
    (checkout / ".git_archival.txt").write_text("$Format:%H$\n")
    environment = dict(os.environ)
    for key in ("TERRA_GIT_COMMIT", "PYTHONPATH", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        environment.pop(key, None)
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")

    def git(*arguments: str, cwd: Path = checkout) -> str:
        return subprocess.run(
            ["git", *arguments], cwd=cwd, env=environment, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("add", ".")
    git(
        "-c",
        "user.name=Release Test",
        "-c",
        "user.email=release@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "Archive fixture",
    )
    expected_commit = git("rev-parse", "HEAD")
    archive_path = tmp_path / "source.zip"
    git("archive", "--format=zip", f"--output={archive_path}", "HEAD")
    unpacked = tmp_path / "unpacked"
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(unpacked)
    assert not (unpacked / ".git").exists()
    assert not (unpacked / "src/terra/_build_commit.txt").exists()
    # Remove the temporary checkout too: the archive must supply its own revision.
    shutil.rmtree(checkout)
    if enclosing_git:
        git("init", "--quiet", cwd=tmp_path)
        git(
            "-c",
            "user.name=Release Test",
            "-c",
            "user.email=release@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Unrelated parent repository",
            cwd=tmp_path,
        )
        assert git("rev-parse", "HEAD", cwd=tmp_path) != expected_commit
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd() / 'src')); "
            "from terra._revision import write_git_commit; write_git_commit(Path('results'))",
        ],
        cwd=unpacked,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (unpacked / "results/GIT_COMMIT").read_text() == expected_commit + "\n"
    subprocess.run(
        [sys.executable, "-c", "from terra_build_backend import build_wheel; build_wheel('dist')"],
        cwd=unpacked,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel_path,) = (unpacked / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel_path) as wheel:
        assert wheel.read("terra/_build_commit.txt").decode() == expected_commit + "\n"


@pytest.fixture
def isolated_prior_configuration(monkeypatch):
    import loco_mujoco

    monkeypatch.delenv("MUSCLEMIMIC_MOSHPP_POSE_BODY_PRIOR_PATH", raising=False)
    monkeypatch.delenv("MUSCLEMIMIC_MOSHPP_ASSETS_PATH", raising=False)
    configuration = {}
    monkeypatch.setattr(loco_mujoco, "load_path_config", lambda: configuration)
    return configuration


def test_paper_fitting_accepts_unconfigured_l2_prior(isolated_prior_configuration) -> None:
    from terra.datasets.marker_fitting import require_paper_pose_prior

    require_paper_pose_prior()


@pytest.mark.parametrize("source", ("environment_file", "environment_directory", "config_file", "config_directory"))
def test_paper_fitting_rejects_gmm_from_host_configuration(
    source: str, tmp_path: Path, monkeypatch, isolated_prior_configuration
) -> None:
    from terra.datasets.marker_fitting import require_paper_pose_prior

    prior = tmp_path / "pose_body_prior.pkl"
    prior.write_bytes(b"fixture: reject before loading any model")
    if source.endswith("file"):
        key, value = "MUSCLEMIMIC_MOSHPP_POSE_BODY_PRIOR_PATH", str(prior)
    else:
        key, value = "MUSCLEMIMIC_MOSHPP_ASSETS_PATH", str(tmp_path)
    if source.startswith("environment"):
        monkeypatch.setenv(key, value)
    else:
        isolated_prior_configuration[key] = value
    with pytest.raises(ValueError, match="Paper conversion uses the L2 pose prior"):
        require_paper_pose_prior()
