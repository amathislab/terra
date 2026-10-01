"""Fail-closed diagnostics for discontinuous retargeted trajectories."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

DEFAULT_MAX_CONTROL_ROOT_SPEED_M_S = 3.0
DEFAULT_MAX_NATIVE_ROOT_SPEED_RATIO = 2.0
DEFAULT_MAX_LANDMARK_ERROR_M = 0.30
DEFAULT_RETRY_STEP_SIZE = 0.10


def retarget_stability(
    result: Any,
    *,
    max_control_root_speed_m_s: float = DEFAULT_MAX_CONTROL_ROOT_SPEED_M_S,
    max_native_root_speed_ratio: float = DEFAULT_MAX_NATIVE_ROOT_SPEED_RATIO,
    max_landmark_error_m: float = DEFAULT_MAX_LANDMARK_ERROR_M,
) -> dict[str, float | bool]:
    """Measure the final, post-repair trajectory and decide whether a slow retry is required.

    The retry gate is conjunctive. Fast root motion alone may be intentional and a large
    error at one distal landmark alone may reflect imperfect source fitting. Their
    combination is the observed solver-bifurcation signature: a discontinuous output and
    a landmark that is hundreds of millimetres from its target.
    """

    if not np.isfinite(max_control_root_speed_m_s) or max_control_root_speed_m_s <= 0.0:
        raise ValueError("max_control_root_speed_m_s must be finite and positive")
    if not np.isfinite(max_native_root_speed_ratio) or max_native_root_speed_ratio <= 1.0:
        raise ValueError("max_native_root_speed_ratio must be finite and greater than one")
    if not np.isfinite(max_landmark_error_m) or max_landmark_error_m <= 0.0:
        raise ValueError("max_landmark_error_m must be finite and positive")
    qpos = np.asarray(result.trajectory.data.qpos, dtype=float)
    frequency = float(result.trajectory.info.frequency)
    position_error = np.asarray(result.analysis["pos_error"], dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] < 3 or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError("final retargeted qpos must be a finite non-empty (T, nq>=3) array")
    if position_error.ndim != 2 or not position_error.size or not np.isfinite(position_error).all():
        raise ValueError("retargeted pos_error must be a finite non-empty matrix")
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("retargeted trajectory frequency must be finite and positive")
    root_step = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1)
    max_root_step = float(root_step.max(initial=0.0))
    max_root_speed = frequency * max_root_step
    native_envelope = result.analysis.get("numerical_envelope")
    if isinstance(native_envelope, str) and native_envelope.startswith("__terra_json__:"):
        native_envelope = json.loads(native_envelope.removeprefix("__terra_json__:"))
    if not isinstance(native_envelope, dict):
        raise ValueError("retargeted analysis must contain the native numerical_envelope")
    native_frequency = float(result.analysis.get("native_fps"))
    native_root_step = float(native_envelope["max_root_step_m"])
    if not np.isfinite(native_frequency) or native_frequency <= 0.0:
        raise ValueError("retargeted native_fps must be finite and positive")
    if not np.isfinite(native_root_step) or native_root_step < 0.0:
        raise ValueError("native max_root_step_m must be finite and non-negative")
    native_root_speed = native_frequency * native_root_step
    effective_speed_limit = max(
        max_control_root_speed_m_s,
        max_native_root_speed_ratio * native_root_speed,
    )
    max_error = float(position_error.max())
    mean_error = float(position_error.mean())
    requires_retry = max_root_speed > effective_speed_limit and max_error > max_landmark_error_m
    return {
        "max_control_root_step_m": max_root_step,
        "max_control_root_speed_m_s": max_root_speed,
        "native_root_speed_m_s": native_root_speed,
        "landmark_error_mean_m": mean_error,
        "landmark_error_max_m": max_error,
        "root_speed_limit_m_s": float(max_control_root_speed_m_s),
        "native_root_speed_ratio_limit": float(max_native_root_speed_ratio),
        "effective_root_speed_limit_m_s": effective_speed_limit,
        "landmark_error_limit_m": float(max_landmark_error_m),
        "requires_retry": bool(requires_retry),
    }


def validate_stability_retry(primary: dict[str, Any], retry: dict[str, Any]) -> None:
    """Require a retry to clear the gate without materially regressing mean tracking."""

    if not primary.get("requires_retry"):
        raise ValueError("a stability retry was requested for an output that passed the gate")
    if retry.get("requires_retry"):
        raise ValueError("stability retry still exceeds the discontinuity gate")
    primary_mean = float(primary["landmark_error_mean_m"])
    retry_mean = float(retry["landmark_error_mean_m"])
    if retry_mean > 1.25 * primary_mean + 0.005:
        raise ValueError(
            "stability retry cleared the discontinuity but materially regressed mean tracking: "
            f"{primary_mean:.4f} -> {retry_mean:.4f} m"
        )


def apply_stability_policy(
    result: Any,
    *,
    config: Mapping[str, object] | None,
    retry: Callable[[dict[str, object]], Any],
) -> Any:
    """Validate one TERRA result and perform the single conservative retry policy.

    ``retry`` receives a complete copy of the caller's method configuration with
    the conservative step size applied. Keeping the rerun callback outside this
    module lets the same final-output policy cover SMPL-H, C3D, dataset, and CLI
    orchestration without teaching the numerical diagnostic about input formats.
    """

    overrides = dict(config or {})
    primary = retarget_stability(result)
    result.analysis["stability_check"] = primary
    if not primary["requires_retry"]:
        return result

    configured_step = float(overrides.get("step_size", 0.2))
    if configured_step <= DEFAULT_RETRY_STEP_SIZE:
        raise ValueError("retargeted output failed the stability gate at the conservative step size")

    retry_result = retry(overrides | {"step_size": DEFAULT_RETRY_STEP_SIZE})
    retry_report = retarget_stability(retry_result)
    validate_stability_retry(primary, retry_report)
    retry_result.analysis.update(
        stability_check=retry_report,
        stability_retry={
            "triggered": True,
            "primary": primary,
            "retry_step_size": DEFAULT_RETRY_STEP_SIZE,
        },
    )
    return retry_result


__all__ = [
    "DEFAULT_MAX_CONTROL_ROOT_SPEED_M_S",
    "DEFAULT_MAX_LANDMARK_ERROR_M",
    "DEFAULT_MAX_NATIVE_ROOT_SPEED_RATIO",
    "DEFAULT_RETRY_STEP_SIZE",
    "apply_stability_policy",
    "retarget_stability",
    "validate_stability_retry",
]
