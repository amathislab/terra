"""General Motion Retargeting (GMR) baseline adapter."""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from terra._musclemimic import fit_gmr_motion as _fit_gmr_motion
from terra.baselines._spec import BaselineSpec
from terra.baselines.temporal import resample_smplh_motion
from terra.smplh import load_smplh_motion

_GMR_RUNTIME_LOCK = threading.RLock()

GMR_BASELINE = BaselineSpec(
    key="gmr",
    label="GMR",
    dependency_extra="baselines",
    config={
        "algorithm": "gmr",
        "offset_to_ground": True,
        "target_fps": 30,
        "exact_target_fps": False,
        "solver": "daqp",
        "damping": 0.5,
        # Match upstream GMR's released safety default instead of the historical
        # MuscleMimic wrapper fallback.
        "use_velocity_limit": True,
        "use_fitted_shape": True,
        "allow_cache_download": False,
    },
)


@contextmanager
def runtime_paths(
    cache_root: str | Path | None,
    model_root: str | Path | None,
) -> Iterator[None]:
    """Scope the two path variables still required by the pinned GMR backend.

    LocoMuJoCo's GMR shape fitter does not yet accept its cache root as an argument.
    Keep that legacy process-global interface behind one serialized adapter and restore
    the caller's environment exactly after the backend returns.
    """

    if cache_root is None and model_root is None:
        yield
        return

    requested = {
        "CONVERTED_AMASS_PATH": cache_root,
        "SMPL_MODEL_PATH": model_root,
    }
    with _GMR_RUNTIME_LOCK:
        previous = {name: os.environ.get(name) for name in requested}
        try:
            for name, value in requested.items():
                if value is not None:
                    os.environ[name] = str(Path(value).expanduser().resolve())
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def prepare_fitted_shape(
    env_name: str,
    cache_root: str | Path,
    model_root: str | Path,
    *,
    iterations: int = 500,
) -> Path:
    """Materialize and validate GMR's shared fitted shape before workers start."""

    if iterations < 1:
        raise ValueError("GMR shape-fitting iterations must be positive")
    with runtime_paths(cache_root, model_root):
        from loco_mujoco.smpl.retargeting import prepare_gmr_fitted_shape

        path = Path(prepare_gmr_fitted_shape(env_name=env_name, iterations=iterations)).resolve()
    metadata = path.with_name(path.name.replace("_shape.pkl", "_shape_metadata.json"))
    if not path.is_file() or not metadata.is_file():
        raise RuntimeError(f"GMR shape preparation did not publish a complete artifact pair: {path}")
    return path


def fit_motion(env_name, robot_conf, motion_data, logger, config: Mapping[str, object] | None = None):
    """Run GMR with TERRA's frozen baseline defaults plus explicit overrides."""
    resolved = GMR_BASELINE.resolved_config(config)
    resolved.pop("algorithm", None)
    resolved.pop("allow_cache_download", None)
    cache_root = resolved.pop("cache_root", None)
    model_root = resolved.get("smpl_model_path")
    exact_target_fps = resolved.pop("exact_target_fps")
    if not isinstance(exact_target_fps, bool):
        raise ValueError("GMR exact_target_fps must be a boolean")
    with runtime_paths(cache_root, model_root):
        if not exact_target_fps:
            return _fit_gmr_motion(env_name, robot_conf, motion_data, logger, resolved)
        target_fps = resolved.get("target_fps")
        canonical, report = resample_smplh_motion(load_smplh_motion(motion_data), target_fps)
        with tempfile.TemporaryDirectory(prefix="terra-gmr-rate-") as directory:
            source = Path(directory) / "motion_poses.npz"
            body_pose = np.asarray(canonical["pose_aa"])
            hand_pose = np.zeros((len(body_pose), 90), dtype=body_pose.dtype)
            poses = np.concatenate((body_pose[:, :66], hand_pose), axis=1)
            np.savez(
                source,
                poses=poses,
                root_orient=poses[:, :3],
                pose_body=poses[:, 3:66],
                left_hand_pose=poses[:, 66:111],
                right_hand_pose=poses[:, 111:156],
                trans=canonical["trans"],
                betas=canonical["betas"],
                gender=np.asarray(canonical["gender"]),
                mocap_framerate=np.asarray(canonical["fps"]),
                mocap_frame_rate=np.asarray(canonical["fps"]),
            )
            trajectory, analysis = _fit_gmr_motion(
                env_name,
                robot_conf,
                str(source),
                logger,
                resolved,
            )
        return trajectory, dict(analysis) | report


__all__ = ["GMR_BASELINE", "fit_motion", "prepare_fitted_shape", "runtime_paths"]
