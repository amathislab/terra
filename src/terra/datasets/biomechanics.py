"""Portable biomechanical ground-truth sidecars for converted motions.

The sidecar keeps both the samples supplied by a dataset and values synchronized to
the converted SMPL-H trajectory.  Consumers should use ``emg`` and ``grf_*`` for a
frame-for-frame comparison with a policy rollout; ``*_native`` is retained so that
filtering or contact processing can be changed without re-reading the source release.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

BIOMECHANICS_SCHEMA_VERSION = 1
COORDINATE_FRAME = "TERRA Z-up right-handed; positions m, forces N, moments Nm"


def biomechanics_path(motion_path: str | Path) -> Path:
    """Return the biomechanical sidecar next to an AMASS-like motion."""

    path = Path(motion_path)
    return path.with_name(f"{path.stem}_biomechanics.npz")


def motion_clock(motion_path: str | Path) -> tuple[np.ndarray, float]:
    """Read the exact clock of a converted AMASS/SMPL-H motion."""

    with np.load(motion_path, allow_pickle=False) as data:
        poses = np.asarray(data["poses"])
        fps = float(np.asarray(data["mocap_framerate"]).reshape(()))
    if poses.ndim != 2 or not len(poses) or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion clock in {motion_path}")
    return np.arange(len(poses), dtype=np.float64) / fps, fps


def resample_linear(
    native_time_s: np.ndarray,
    values: np.ndarray,
    motion_time_s: np.ndarray,
    *,
    boundary_tolerance_s: float | None = None,
) -> np.ndarray:
    """Linearly sample arbitrary trailing dimensions on the motion clock.

    A source is permitted to end less than one native sample before the last motion
    frame.  This occurs in force files whose half-open interval ends immediately before
    the final marker sample.  Larger extrapolations are rejected.
    """

    times = np.asarray(native_time_s, dtype=np.float64).reshape(-1)
    array = np.asarray(values)
    target = np.asarray(motion_time_s, dtype=np.float64).reshape(-1)
    if array.shape[:1] != times.shape or not len(times):
        raise ValueError(f"Native time/value shapes disagree: {times.shape} and {array.shape}")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Native timestamps must be finite and strictly increasing")
    if boundary_tolerance_s is None:
        boundary_tolerance_s = float(np.max(np.diff(times))) * 1.01 if len(times) > 1 else 0.0
    if target[0] < times[0] - boundary_tolerance_s or target[-1] > times[-1] + boundary_tolerance_s:
        raise ValueError(
            f"Source clock [{times[0]:.6f}, {times[-1]:.6f}] does not cover motion "
            f"clock [{target[0]:.6f}, {target[-1]:.6f}]"
        )
    flat = np.asarray(array, dtype=np.float64).reshape(len(times), -1)
    aligned = np.empty((len(target), flat.shape[1]), dtype=np.float64)
    clipped = np.clip(target, times[0], times[-1])
    for column in range(flat.shape[1]):
        aligned[:, column] = np.interp(clipped, times, flat[:, column])
    return aligned.reshape((len(target), *array.shape[1:])).astype(np.float32)


def empty_emg(motion_frames: int) -> dict[str, Any]:
    return {
        "emg_available": np.array(False),
        "emg": np.empty((motion_frames, 0), dtype=np.float32),
        "emg_native": np.empty((0, 0), dtype=np.float32),
        "emg_native_time_s": np.empty(0, dtype=np.float64),
        "emg_channels": np.empty(0, dtype="U1"),
        "emg_muscles": np.empty(0, dtype="U1"),
        "emg_channel_types": np.empty(0, dtype="U1"),
        "emg_channel_sides": np.empty(0, dtype="U1"),
        "emg_units": np.array(""),
        "emg_processing": np.array("unavailable"),
        "emg_source_paths": np.empty(0, dtype="U1"),
    }


def empty_grf(motion_frames: int) -> dict[str, Any]:
    shape = (motion_frames, 0, 3)
    native_shape = (0, 0, 3)
    return {
        "grf_available": np.array(False),
        "grf_cop_available": np.array(False),
        "grf_force": np.empty(shape, dtype=np.float32),
        "grf_moment": np.empty(shape, dtype=np.float32),
        "grf_cop": np.empty(shape, dtype=np.float32),
        "grf_valid": np.empty((motion_frames, 0), dtype=bool),
        "grf_force_native": np.empty(native_shape, dtype=np.float32),
        "grf_moment_native": np.empty(native_shape, dtype=np.float32),
        "grf_cop_native": np.empty(native_shape, dtype=np.float32),
        "grf_valid_native": np.empty((0, 0), dtype=bool),
        "grf_native_time_s": np.empty(0, dtype=np.float64),
        "grf_channels": np.empty(0, dtype="U1"),
        "grf_source_paths": np.empty(0, dtype="U1"),
    }


def write_biomechanics(
    path: str | Path,
    *,
    dataset: str,
    motion: str,
    motion_time_s: np.ndarray,
    motion_fps: float,
    synchronization: str,
    emg: dict[str, Any] | None = None,
    grf: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write and validate one sidecar atomically."""

    target = Path(path)
    clock = np.asarray(motion_time_s, dtype=np.float64).reshape(-1)
    payload: dict[str, Any] = {
        "schema_version": np.array(BIOMECHANICS_SCHEMA_VERSION, dtype=np.int64),
        "dataset": np.array(dataset),
        "motion": np.array(motion),
        "motion_frames": np.array(len(clock), dtype=np.int64),
        "motion_fps": np.array(motion_fps, dtype=np.float64),
        "motion_time_s": clock,
        "coordinate_frame": np.array(COORDINATE_FRAME),
        "synchronization": np.array(synchronization),
    }
    payload.update(empty_emg(len(clock)) if emg is None else emg)
    payload.update(empty_grf(len(clock)) if grf is None else grf)
    payload.update(metadata or {})
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(target)
    validate_biomechanics(target)


def validate_biomechanics(
    path: str | Path,
    *,
    motion_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate schema, units, and frame-level synchronization."""

    required = {
        "schema_version", "dataset", "motion", "motion_frames", "motion_fps",
        "motion_time_s", "coordinate_frame", "synchronization",
        "emg_available", "emg", "emg_native", "emg_native_time_s",
        "emg_channels", "emg_muscles", "emg_channel_types", "emg_channel_sides",
        "emg_units", "emg_processing",
        "emg_source_paths", "grf_available", "grf_cop_available", "grf_force", "grf_moment",
        "grf_cop", "grf_valid", "grf_force_native", "grf_moment_native",
        "grf_cop_native", "grf_valid_native", "grf_native_time_s",
        "grf_channels", "grf_source_paths",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        version = int(np.asarray(data["schema_version"]).reshape(()))
        frames = int(np.asarray(data["motion_frames"]).reshape(()))
        fps = float(np.asarray(data["motion_fps"]).reshape(()))
        clock = np.asarray(data["motion_time_s"], dtype=np.float64)
        emg_available = bool(np.asarray(data["emg_available"]).reshape(()))
        emg = np.asarray(data["emg"])
        emg_native = np.asarray(data["emg_native"])
        emg_time = np.asarray(data["emg_native_time_s"], dtype=np.float64)
        channels = np.asarray(data["emg_channels"]).reshape(-1)
        muscles = np.asarray(data["emg_muscles"]).reshape(-1)
        channel_types = np.asarray(data["emg_channel_types"]).reshape(-1)
        channel_sides = np.asarray(data["emg_channel_sides"]).reshape(-1)
        grf_available = bool(np.asarray(data["grf_available"]).reshape(()))
        grf_cop_available = bool(np.asarray(data["grf_cop_available"]).reshape(()))
        force = np.asarray(data["grf_force"])
        moment = np.asarray(data["grf_moment"])
        cop = np.asarray(data["grf_cop"])
        valid = np.asarray(data["grf_valid"], dtype=bool)
        force_native = np.asarray(data["grf_force_native"])
        moment_native = np.asarray(data["grf_moment_native"])
        cop_native = np.asarray(data["grf_cop_native"])
        valid_native = np.asarray(data["grf_valid_native"], dtype=bool)
        grf_time = np.asarray(data["grf_native_time_s"], dtype=np.float64)
        grf_channels = np.asarray(data["grf_channels"]).reshape(-1)
    if version != BIOMECHANICS_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {version}")
    expected_clock = np.arange(frames, dtype=np.float64) / fps
    if frames < 1 or not np.isfinite(fps) or fps <= 0 or not np.allclose(clock, expected_clock, atol=1e-10):
        raise ValueError("motion_time_s is not the declared uniform motion clock")
    if motion_path is not None:
        actual_clock, actual_fps = motion_clock(motion_path)
        if not np.isclose(fps, actual_fps) or not np.array_equal(clock, actual_clock):
            raise ValueError("biomechanics clock does not match the converted motion")
    if (
        emg.shape != (frames, len(channels))
        or len(muscles) != len(channels)
        or len(channel_types) != len(channels)
        or len(channel_sides) != len(channels)
    ):
        raise ValueError(f"invalid aligned EMG shape {emg.shape}")
    if emg_native.shape != (len(emg_time), len(channels)):
        raise ValueError(f"invalid native EMG shape {emg_native.shape}")
    if emg_available:
        if (
            not len(channels)
            or len(set(map(str, channels))) != len(channels)
            or not np.isfinite(emg).all()
            or not np.isfinite(emg_native).all()
            or not np.isfinite(emg_time).all()
            or np.any(np.diff(emg_time) <= 0)
        ):
            raise ValueError("available EMG is empty or non-finite")
    elif len(channels) or emg_native.size:
        raise ValueError("unavailable EMG must use empty arrays")
    n_grf = len(grf_channels)
    if force.shape != (frames, n_grf, 3) or moment.shape != force.shape or cop.shape != force.shape:
        raise ValueError(f"invalid aligned GRF shapes {force.shape}, {moment.shape}, {cop.shape}")
    if valid.shape != (frames, n_grf):
        raise ValueError(f"invalid aligned GRF-valid shape {valid.shape}")
    native_shape = (len(grf_time), n_grf, 3)
    if force_native.shape != native_shape or moment_native.shape != native_shape or cop_native.shape != native_shape:
        raise ValueError("invalid native GRF shapes")
    if valid_native.shape != (len(grf_time), n_grf):
        raise ValueError("invalid native GRF-valid shape")
    if grf_available:
        if (
            not n_grf
            or len(set(map(str, grf_channels))) != n_grf
            or not np.isfinite(force).all()
            or not np.isfinite(moment).all()
            or not np.isfinite(grf_time).all()
            or np.any(np.diff(grf_time) <= 0)
        ):
            raise ValueError("available force/moment data is empty or non-finite")
        if grf_cop_available and np.any(valid & ~np.isfinite(cop).all(axis=-1)):
            raise ValueError("active GRF samples have a non-finite CoP")
    elif n_grf or force_native.size:
        raise ValueError("unavailable GRF must use empty arrays")
    return {
        "frames": frames,
        "fps": fps,
        "emg_available": emg_available,
        "emg_channels": len(channels),
        "grf_available": grf_available,
        "grf_channels": n_grf,
    }
