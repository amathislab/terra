"""Safe loading and validation for individual SMPL-H motion archives."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_smplh_motion(path: str | Path) -> dict[str, object]:
    """Load and validate one AMASS-compatible SMPL-H motion archive.

    Required fields are ``trans`` (frames x 3 metres), ``betas`` (at least 10
    coefficients), scalar ``gender``, and either AMASS ``poses`` (at least 66
    columns) or canonical ``pose_aa`` (72 columns). The frame rate comes from
    ``fps``, ``mocap_framerate``, or ``mocap_frame_rate``. At least three
    frames, a positive rate, and finite numeric arrays are required.

    AMASS hand rotations are omitted because MyoFullBody tracks the 22-body
    subset; six zero hand-root values preserve the 72-value layout. NumPy
    pickle deserialization is disabled. The returned mapping contains
    ``pose_aa``, ``trans``, ``betas``, ``fps``, and ``gender``.

    Raises:
        FileNotFoundError: If the archive is absent.
        ValueError: If its extension, fields, dimensions, or values are invalid.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SMPL-H motion not found: {source}")
    if source.suffix.casefold() != ".npz":
        raise ValueError(f"SMPL-H motion must be an .npz archive, got {source.name!r}")

    with np.load(source, allow_pickle=False) as archive:
        _require_archive_fields(archive, "trans", "betas", "gender")
        pose_aa = _load_pose_aa(archive)
        trans = np.asarray(archive["trans"], dtype=float)
        betas = _load_betas(archive)
        fps = _load_frame_rate(archive)
        gender = _load_gender(archive)

    _validate_smplh_motion(pose_aa, trans, betas, fps)
    return {"pose_aa": pose_aa, "trans": trans, "betas": betas, "fps": fps, "gender": gender}


def _require_archive_fields(archive: np.lib.npyio.NpzFile, *fields: str) -> None:
    missing = sorted(set(fields) - set(archive.files))
    if missing:
        raise ValueError(f"SMPL-H archive is missing required field(s): {', '.join(missing)}")


def _load_pose_aa(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    if "pose_aa" in archive.files:
        return np.asarray(archive["pose_aa"], dtype=float)
    if "poses" not in archive.files:
        raise ValueError("SMPL-H archive must contain 'poses' or 'pose_aa'")

    poses = np.asarray(archive["poses"], dtype=float)
    if poses.ndim != 2 or poses.shape[1] < 66:
        raise ValueError(f"poses must have shape (frames, >=66), got {poses.shape}")
    hand_roots = np.zeros((len(poses), 6), dtype=poses.dtype)
    return np.concatenate((poses[:, :66], hand_roots), axis=1)


def _load_betas(archive: np.lib.npyio.NpzFile) -> np.ndarray:
    betas = np.asarray(archive["betas"], dtype=float)
    if betas.ndim == 2 and betas.shape[0] == 1:
        return betas[0]
    return betas


def _load_frame_rate(archive: np.lib.npyio.NpzFile) -> float:
    rate_fields = ("fps", "mocap_framerate", "mocap_frame_rate")
    rate_field = next((name for name in rate_fields if name in archive.files), None)
    if rate_field is None:
        raise ValueError("SMPL-H archive is missing fps/mocap_framerate")
    value = _load_scalar(archive, rate_field, "motion frame rate")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"motion frame rate must be numeric, got {value!r}") from exc


def _load_gender(archive: np.lib.npyio.NpzFile) -> str:
    value = _load_scalar(archive, "gender", "gender")
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("SMPL-H gender bytes must be UTF-8 encoded") from exc
    gender = str(value).casefold()
    if gender not in {"female", "male", "neutral"}:
        raise ValueError(f"unsupported SMPL-H gender {value!r}")
    return gender


def _load_scalar(archive: np.lib.npyio.NpzFile, field: str, description: str) -> object:
    values = np.asarray(archive[field])
    if values.size != 1:
        raise ValueError(f"SMPL-H {description} must be scalar, got shape {values.shape}")
    return values.reshape(()).item()


def _validate_smplh_motion(
    pose_aa: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    fps: float,
) -> None:
    if pose_aa.ndim != 2 or pose_aa.shape[1] != 72:
        raise ValueError(f"pose_aa must have shape (frames, 72), got {pose_aa.shape}")
    if len(pose_aa) < 3:
        raise ValueError(f"SMPL-H motion must contain at least 3 frames, got {len(pose_aa)}")
    if trans.shape != (len(pose_aa), 3):
        raise ValueError(f"trans must have shape ({len(pose_aa)}, 3), got {trans.shape}")
    if betas.ndim != 1:
        raise ValueError(f"betas must be a vector or single-row matrix, got {betas.shape}")
    if betas.size < 10:
        raise ValueError(f"betas must contain at least 10 coefficients, got {betas.size}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"motion frame rate must be positive and finite, got {fps!r}")
    if not (np.isfinite(pose_aa).all() and np.isfinite(trans).all() and np.isfinite(betas).all()):
        raise ValueError("SMPL-H motion contains non-finite pose, translation, or shape values")


__all__ = ["load_smplh_motion"]
