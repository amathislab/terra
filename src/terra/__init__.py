"""TERRA: terrain-aware retargeting for musculoskeletal control.

Use :func:`retarget` for a filesystem SMPL-H/C3D/TRC/MAT input and
:func:`save_retarget_result` to publish the paired trajectory, analysis, and terrain
artifacts consumed by PPO. Exports are loaded on first access so importing the
package for metadata or configuration discovery does not initialize the numerical,
visualization, and accelerator stacks.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any


def _export_group(module: str, *names: str) -> dict[str, tuple[str, str]]:
    return {name: (module, name) for name in names}


_EXPORTS = {
    **_export_group("terra._methods", "SUPPORTED_METHODS", "RetargetingMethod"),
    **_export_group(
        "terra.paths",
        "ARTIFACT_ROOT_ENV",
        "DATA_ROOT_ENV",
        "MODEL_ROOT_ENV",
        "StorageRoots",
    ),
    **_export_group(
        "terra.artifacts",
        "RETARGET_ARTIFACT_FORMAT_VERSION",
        "RetargetArtifacts",
        "RetargetPaths",
        "ValidatedRetargetArtifacts",
        "load_retarget_analysis",
        "normalize_motion_name",
        "retarget_cache_paths",
        "save_retarget_result",
        "validate_retarget_artifacts",
    ),
    **_export_group(
        "terra.contracts",
        "RetargetResult",
    ),
    **_export_group(
        "terra.api",
        "TerrainInput",
        "retarget",
        "retarget_c3d",
        "retarget_mat",
        "retarget_smplh",
        "retarget_trc",
    ),
    **_export_group("terra.smplh", "load_smplh_motion"),
    **_export_group("terra.terrain.metadata", "TerrainMetadata"),
}

if TYPE_CHECKING:
    from terra._methods import SUPPORTED_METHODS, RetargetingMethod
    from terra.api import TerrainInput, retarget, retarget_c3d, retarget_mat, retarget_smplh, retarget_trc
    from terra.artifacts import (
        RETARGET_ARTIFACT_FORMAT_VERSION,
        RetargetArtifacts,
        RetargetPaths,
        ValidatedRetargetArtifacts,
        load_retarget_analysis,
        normalize_motion_name,
        retarget_cache_paths,
        save_retarget_result,
        validate_retarget_artifacts,
    )
    from terra.contracts import RetargetResult
    from terra.paths import ARTIFACT_ROOT_ENV, DATA_ROOT_ENV, MODEL_ROOT_ENV, StorageRoots
    from terra.smplh import load_smplh_motion
    from terra.terrain.metadata import TerrainMetadata

try:
    __version__ = version("terra-retargeting")
except PackageNotFoundError:  # Source trees without installed package metadata.
    __version__ = "1.0.0"

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
