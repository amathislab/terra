"""Package-boundary tests that must run in a fresh interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path

import terra


def test_root_all_advertises_only_the_stable_file_oriented_api():
    assert set(terra.__all__) == {
        "__version__",
        "ARTIFACT_ROOT_ENV",
        "DATA_ROOT_ENV",
        "MODEL_ROOT_ENV",
        "RETARGET_ARTIFACT_FORMAT_VERSION",
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
    for removed in (
        "AUGMENTATION_VERSION",
        "TerrainAugmentation",
        "fit_terra_height_augmented_motion",
        "height_augmentation_method",
        "paired_terrain_scenes",
        "TerraRetargeter",
        "fit_terra_motion",
        "omniretarget_profile_is_clean",
    ):
        assert not hasattr(terra, removed)


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
    assert report["version"] == "1.0.0"
    assert report["methods"] == ["terra", "omniretarget", "gmr", "smpl"]
    assert report["motion_name"] == "Study/Subject/Trial"
    assert report["loaded"] == []


def test_package_declares_inline_type_information():
    assert files(terra).joinpath("py.typed").is_file()


def test_runtime_git_dependencies_are_commit_pinned():
    project_root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = metadata["project"]["dependencies"]

    assert (
        "musclemimic @ "
        "git+https://github.com/amathislab/musclemimic.git"
        "@f1c2dbfd0d8e31d306b2c4e2369aecdc7bc21993"
    ) in requirements
    assert (
        "holosoma-retargeting @ "
        "git+https://github.com/merc-s/holosoma.git"
        "@12022b5ca5e12e460156c2d91a908f80f6c63fa1"
        "#subdirectory=src/holosoma_retargeting"
    ) in requirements
    assert "musclemimic" not in metadata["tool"]["uv"]["sources"]
