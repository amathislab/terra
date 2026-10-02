"""Exact temporal resampling for solve-rate control experiments."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def validate_target_fps(value: object) -> float:
    """Return a finite positive target rate without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("target_fps must be positive and finite")
    target_fps = float(value)
    if not np.isfinite(target_fps) or target_fps <= 0.0:
        raise ValueError("target_fps must be positive and finite")
    return target_fps


def resample_smplh_motion(
    motion_data: Mapping[str, object],
    target_fps: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Resample canonical SMPL-H poses and translations on an endpoint-aligned clock.

    Joint rotations use spherical interpolation independently for every axis-angle
    triplet. Root translations use linear interpolation. Samples lie exactly on the
    requested clock starting at source time zero; a source tail shorter than one target
    period is omitted and recorded in the returned report.
    """

    target_fps = validate_target_fps(target_fps)
    try:
        source_fps = validate_target_fps(motion_data["fps"])
        pose_aa = np.asarray(motion_data["pose_aa"])
        trans = np.asarray(motion_data["trans"])
    except KeyError as exc:
        raise ValueError(f"SMPL-H motion is missing {exc.args[0]!r}") from exc
    if pose_aa.ndim != 2 or pose_aa.shape[1] % 3:
        raise ValueError("SMPL-H pose_aa must have shape (frames, 3 * joints)")
    if trans.shape != (len(pose_aa), 3):
        raise ValueError("SMPL-H trans must have shape (frames, 3)")
    if len(pose_aa) < 1:
        raise ValueError("SMPL-H motion must contain at least one frame")
    if not np.isfinite(pose_aa).all() or not np.isfinite(trans).all():
        raise ValueError("SMPL-H temporal resampling requires finite poses and translations")

    duration_s = (len(pose_aa) - 1) / source_fps
    output_frames = max(1, int(np.floor(duration_s * target_fps + 1.0e-9)) + 1)
    source_times = np.arange(len(pose_aa), dtype=np.float64) / source_fps
    target_times = np.arange(output_frames, dtype=np.float64) / target_fps

    if len(pose_aa) == 1:
        resampled_pose = pose_aa.copy()
        resampled_trans = trans.copy()
    else:
        joints = pose_aa.reshape(len(pose_aa), -1, 3)
        resampled_pose = np.empty((output_frames, joints.shape[1], 3), dtype=np.float64)
        for joint in range(joints.shape[1]):
            rotations = Rotation.from_rotvec(joints[:, joint].astype(np.float64, copy=False))
            resampled_pose[:, joint] = Slerp(source_times, rotations)(target_times).as_rotvec()
        resampled_pose = resampled_pose.reshape(output_frames, pose_aa.shape[1]).astype(
            pose_aa.dtype,
            copy=False,
        )
        resampled_trans = np.column_stack(
            [np.interp(target_times, source_times, trans[:, axis]) for axis in range(3)]
        ).astype(trans.dtype, copy=False)

    result = dict(motion_data)
    result.update(pose_aa=resampled_pose, trans=resampled_trans, fps=target_fps)
    report: dict[str, object] = {
        "solve_rate_source_fps": source_fps,
        "solve_rate_target_fps": target_fps,
        "solve_rate_source_frames": len(pose_aa),
        "solve_rate_target_frames": output_frames,
        "solve_rate_duration_s": duration_s,
        "solve_rate_trimmed_tail_s": duration_s - float(target_times[-1]),
        "solve_rate_resampling": "endpoint-aligned linear translation and per-joint rotation SLERP",
    }
    return result, report


__all__ = ["resample_smplh_motion", "validate_target_fps"]
