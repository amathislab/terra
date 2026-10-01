"""Load OmniRetarget and define TERRA's contribution-free baseline adapter."""

from __future__ import annotations

from collections.abc import Mapping

from terra._musclemimic import SMPLH_BONE_ORDER_NAMES
from terra.baselines._spec import BaselineSpec

OMNIRETARGET_BASELINE = BaselineSpec(
    key="omniretarget",
    label="OmniRetarget",
    config={"algorithm": "omniretarget", "method_profile": "omniretarget"},
)

try:
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
    from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
    from holosoma_retargeting.src.utils import (
        extract_foot_sticking_sequence_velocity,
        preprocess_motion_data,
        transform_from_human_to_world,
    )

    OMNIRETARGET_INSTALLED = True
except ImportError:
    OMNIRETARGET_INSTALLED = False
    SMPLH_DEMO_JOINTS = None
    InteractionMeshRetargeter = None
    extract_foot_sticking_sequence_velocity = None
    preprocess_motion_data = None
    transform_from_human_to_world = None


def require_omniretarget() -> None:
    """Raise an actionable error when the external baseline is unavailable."""
    if not OMNIRETARGET_INSTALLED:
        raise ImportError(
            "OmniRetarget (holosoma_retargeting) is required. Run `uv sync --locked` "
            "from the standalone TERRA repository."
        )


def demo_joint_permutation() -> list[int]:
    """Build the permutation from MuscleMimic to OmniRetarget joint order."""
    require_omniretarget()
    return [SMPLH_BONE_ORDER_NAMES.index(name) for name in SMPLH_DEMO_JOINTS]


def fit_motion(env_name, robot_conf, motion_data, logger, config: Mapping[str, object] | None = None):
    """Run the contribution-free OmniRetarget profile through TERRA's adapter."""
    from terra.pipeline import fit_terra_motion
    from terra.profiles import resolve_method_profile

    resolved = resolve_method_profile("omniretarget", dict(config or {}))
    return fit_terra_motion(env_name, robot_conf, motion_data, logger, resolved)


__all__ = [
    "OMNIRETARGET_BASELINE",
    "OMNIRETARGET_INSTALLED",
    "SMPLH_DEMO_JOINTS",
    "InteractionMeshRetargeter",
    "demo_joint_permutation",
    "extract_foot_sticking_sequence_velocity",
    "fit_motion",
    "preprocess_motion_data",
    "require_omniretarget",
    "transform_from_human_to_world",
]
