"""Package-boundary tests that must run in a fresh interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.resources import files

import terra


def test_root_all_advertises_only_the_stable_file_oriented_api():
    assert set(terra.__all__) == {
        "__version__",
        "ARTIFACT_ROOT_ENV",
        "DATA_ROOT_ENV",
        "MODEL_ROOT_ENV",
        "SUPPORTED_METHODS",
        "RetargetArtifacts",
        "RetargetPaths",
        "RetargetResult",
        "RetargetingMethod",
        "StorageRoots",
        "TerrainInput",
        "TerrainMetadata",
        "ValidatedRetargetArtifacts",
        "load_retarget_analysis",
        "load_smplh_motion",
        "normalize_motion_name",
        "retarget",
        "retarget_c3d",
        "retarget_cache_paths",
        "retarget_mat",
        "retarget_smplh",
        "retarget_trc",
        "save_retarget_result",
        "validate_retarget_artifacts",
    }
    assert "TerraRetargeter" not in terra.__all__
    assert "fit_terra_motion" not in terra.__all__


def test_package_metadata_import_does_not_initialize_numerical_stacks():
    script = """
import json
import sys
import terra

print(json.dumps({
    "version": terra.__version__,
    "methods": terra.SUPPORTED_METHODS,
    "motion_name": terra.normalize_motion_name("Study/Subject/Trial.npz").as_posix(),
    "loaded": sorted(name for name in sys.modules if name in {
        "jax", "torch", "terra.api", "terra.pipeline", "musclemimic.retargeting"
    }),
}))
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    report = json.loads(completed.stdout)
    assert report["methods"] == ["terra", "omniretarget", "gmr", "smpl"]
    assert report["motion_name"] == "Study/Subject/Trial"
    assert report["loaded"] == []


def test_package_declares_inline_type_information():
    assert files(terra).joinpath("py.typed").is_file()
