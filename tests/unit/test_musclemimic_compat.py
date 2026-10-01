"""Contract tests for TERRA's single MuscleMimic compatibility boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import musclemimic.retargeting as dependency_api
import terra._musclemimic as compat


def test_compatibility_boundary_reexports_the_fork_integration_api():
    for name in compat._REQUIRED_RETARGETING_EXPORTS:
        assert getattr(compat, name) is getattr(dependency_api, name)
    assert compat.torso_frame.__module__ == "musclemimic.utils.torso_frame"


def test_incompatible_distribution_fails_with_installation_guidance(tmp_path):
    fake_package = tmp_path / "musclemimic"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text('__version__ = "0.1.0"\n')
    source_root = Path(__file__).parents[2] / "src"
    environment = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join((str(tmp_path), str(source_root))),
    )
    completed = subprocess.run(
        # Skip site initialization so the development fork's editable import
        # finder cannot override the deliberately incompatible fixture.
        [sys.executable, "-S", "-c", "import terra._musclemimic"],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )

    assert completed.returncode != 0
    assert "requires the amathislab/musclemimic_terra_release package" in completed.stderr
    assert "a different 'musclemimic' distribution is not compatible" in completed.stderr
