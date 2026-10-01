"""Shared motion preparation for reconstruction methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def array_metadata(value: np.ndarray, *, units: str = "m") -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.str,
        "units": units,
    }


def fitted_motion_assets(motion: str, base_env_name: str) -> dict[str, Any]:
    """Describe the fitted motion inputs without exposing installation paths."""

    from loco_mujoco.smpl.retargeting import (
        OPTIMIZED_SHAPE_FILE_NAME,
    )

    return {
        "source_motion": {"identifier": motion},
        "fitted_shape": {
            "base_env_name": base_env_name,
            "file_name": OPTIMIZED_SHAPE_FILE_NAME,
        },
    }


@dataclass(frozen=True)
class MotionLandmarks:
    joints: np.ndarray
    fps: float
    joint_names: tuple[str, ...]


def prepare_motion_landmarks(
    motion: str,
    env_name: str,
    *,
    source_path: Path | None,
    smpl_model_path: Path | None,
    cache_root: Path | None,
) -> tuple[MotionLandmarks, dict[str, Any]]:
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.runtime import shape_cache_path
    from terra.smplh import load_smplh_motion
    from terra.source import motion_world_joints

    base_env_name = env_name.replace("Mjx", "")
    if base_env_name != "MyoFullBody":
        raise ValueError(
            "env_name must resolve to MyoFullBody because this method is explicitly a "
            "MyoFullBody fitted and site-calibrated landmarks"
        )
    if source_path is None or smpl_model_path is None or cache_root is None:
        raise ValueError(
            "reconstruction baselines require a dataset_config that resolves the source motion, "
            "SMPL-H model root, and fitted-shape cache"
        )
    motion_data = load_smplh_motion(source_path)
    joints, fps, normalization = motion_world_joints(
        motion,
        env_name=env_name,
        use_fitted_shape=True,
        motion_data=motion_data,
        calibrate_sites=True,
        smpl_model_path=smpl_model_path,
        fitted_shape_path=shape_cache_path(env_name, cache_root),
        return_normalization=True,
    )
    landmarks = MotionLandmarks(np.asarray(joints), float(fps), tuple(SMPLH_DEMO_JOINTS))
    identity = {
        "input_stage": "fitted_myofullbody_landmarks",
        "adapter": "terra.source.motion_world_joints",
        "env_name": env_name,
        "base_env_name": base_env_name,
        "use_fitted_shape": True,
        "calibrate_sites": True,
        "floor_normalization": "motion_wide_minimum_toe_z",
        "normalization": normalization,
        "joint_order": list(landmarks.joint_names),
        "fps": landmarks.fps,
        "joints": array_metadata(landmarks.joints),
        "artifacts": fitted_motion_assets(motion, base_env_name)
        | {"source_motion": {"identifier": motion, "path": str(source_path)}},
    }
    return landmarks, identity
