"""Validation and normalization for the public C3D fitting configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np

SUPPORTED_C3D_OPTIONS = frozenset(
    {
        "clear_cache",
        "converted_c3d_name",
        "device",
        "enforce_knee_hinge",
        "gender",
        "head_marker_corr_path",
        "least_avail_markers",
        "n_ref_frames",
        "optimize_toes",
        "pose_body_prior_path",
        "seed",
        "stage1_iters",
        "stage1_shape_solver",
        "stage1_state_path",
        "stage2_iters",
        "stage2_marker_weight_overrides",
        "stage2_solver",
        "stage2_torso_frame_weight",
        "strict_frame_picking",
        "surface_model_type",
        "target_fps",
        "wrist_markers_on_stick",
    }
)

_MANAGED_OPTIONS = frozenset(
    {
        "c3d_fit_model_path",
        "cache_root",
        "gmr_config",
        "logger",
        "retarget_smpl_model_path",
        "retargeting_method",
        "terra_config",
    }
)

DEFAULT_C3D_OPTIONS: dict[str, object] = {
    "clear_cache": False,
    "device": "cpu",
    "enforce_knee_hinge": False,
    "gender": "neutral",
    "least_avail_markers": 1.0,
    "n_ref_frames": 12,
    "optimize_toes": True,
    "seed": 100,
    "stage1_iters": 320,
    "stage1_shape_solver": "joint_dogleg_jax",
    "stage1_state_path": None,
    "stage2_iters": 80,
    "stage2_marker_weight_overrides": None,
    "stage2_solver": "frame_lbfgs",
    "stage2_torso_frame_weight": 0.0,
    "strict_frame_picking": True,
    "surface_model_type": "smplx",
    "target_fps": None,
    "wrist_markers_on_stick": False,
}

_GENDERS = frozenset({"female", "male", "neutral"})
_SURFACE_MODELS = frozenset({"smplh", "smplx"})
_STAGE1_SHAPE_SOLVERS = frozenset({"joint_dogleg", "joint_dogleg_jax"})
_STAGE2_SOLVERS = frozenset({"batched_lbfgs", "frame_lbfgs"})
_DEVICE = re.compile(r"(?:cpu|cuda(?::\d+)?)")


def resolve_c3d_options(
    c3d_options: Mapping[str, object] | None,
    *,
    c3d_model_path: str | Path | None,
    smpl_model_path: str | Path | None,
    cache_root: str | Path | None,
) -> dict[str, object]:
    """Validate user overrides and add paths controlled by TERRA's launcher.

    Validation happens before importing or invoking the C3D fitter so malformed
    configuration fails cheaply at the public boundary.
    """

    supplied = dict(c3d_options or {})
    managed = _MANAGED_OPTIONS & supplied.keys()
    if managed:
        names = ", ".join(sorted(managed))
        raise ValueError(f"c3d_options cannot override managed option(s): {names}")

    unknown = supplied.keys() - SUPPORTED_C3D_OPTIONS
    if unknown:
        names = ", ".join(sorted(unknown))
        supported = ", ".join(sorted(SUPPORTED_C3D_OPTIONS))
        raise ValueError(f"unknown c3d_options field(s): {names}; supported fields: {supported}")

    options = dict(DEFAULT_C3D_OPTIONS)
    options.update(supplied)
    _validate_scalar_options(options)
    _validate_marker_weights(options)
    _resolve_option_paths(options)

    explicit_options = {
        "c3d_fit_model_path": c3d_model_path,
        "retarget_smpl_model_path": smpl_model_path,
        "cache_root": cache_root,
    }
    options.update(
        (name, str(Path(value).expanduser().resolve())) for name, value in explicit_options.items() if value is not None
    )
    return options


def _normalize_choice(
    options: dict[str, object],
    name: str,
    choices: frozenset[str],
) -> str:
    value = options[name]
    if not isinstance(value, str):
        raise ValueError(f"c3d_options.{name} must be a string")
    normalized = value.casefold()
    if normalized not in choices:
        supported = ", ".join(sorted(choices))
        raise ValueError(f"c3d_options.{name} must be one of {supported}; got {normalized!r}")
    options[name] = normalized
    return normalized


def _validate_scalar_options(options: dict[str, object]) -> None:
    _normalize_choice(options, "gender", _GENDERS)
    _normalize_choice(options, "surface_model_type", _SURFACE_MODELS)
    _normalize_choice(options, "stage1_shape_solver", _STAGE1_SHAPE_SOLVERS)
    _normalize_choice(options, "stage2_solver", _STAGE2_SOLVERS)

    boolean_fields = (
        "clear_cache",
        "enforce_knee_hinge",
        "optimize_toes",
        "strict_frame_picking",
        "wrist_markers_on_stick",
    )
    for name in boolean_fields:
        if not isinstance(options[name], bool):
            raise ValueError(f"c3d_options.{name} must be a boolean")

    for name in ("n_ref_frames", "stage1_iters", "stage2_iters"):
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"c3d_options.{name} must be a positive integer")
        options[name] = int(value)

    seed = options["seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("c3d_options.seed must be an integer between 0 and 2**32 - 1")
    options["seed"] = int(seed)

    target_fps = options["target_fps"]
    if target_fps is not None:
        if (
            isinstance(target_fps, bool)
            or not isinstance(target_fps, Real)
            or not np.isfinite(target_fps)
            or target_fps <= 0
        ):
            raise ValueError("c3d_options.target_fps must be positive and finite or null")
        options["target_fps"] = float(target_fps)

    marker_fraction = options["least_avail_markers"]
    if (
        isinstance(marker_fraction, bool)
        or not isinstance(marker_fraction, Real)
        or not np.isfinite(marker_fraction)
        or not 0 < marker_fraction <= 1
    ):
        raise ValueError("c3d_options.least_avail_markers must be in (0, 1]")
    options["least_avail_markers"] = float(marker_fraction)

    torso_weight = options["stage2_torso_frame_weight"]
    if (
        isinstance(torso_weight, bool)
        or not isinstance(torso_weight, Real)
        or not np.isfinite(torso_weight)
        or torso_weight < 0
    ):
        raise ValueError("c3d_options.stage2_torso_frame_weight must be finite and non-negative")
    options["stage2_torso_frame_weight"] = float(torso_weight)

    device = options["device"]
    if not isinstance(device, str) or _DEVICE.fullmatch(device.casefold()) is None:
        raise ValueError("c3d_options.device must be 'cpu', 'cuda', or 'cuda:<index>'")
    options["device"] = device.casefold()


def _validate_marker_weights(options: dict[str, object]) -> None:
    overrides = options["stage2_marker_weight_overrides"]
    if overrides is None:
        return
    if not isinstance(overrides, Mapping):
        raise ValueError("c3d_options.stage2_marker_weight_overrides must be an object or null")
    normalized: dict[str, float] = {}
    for label, weight in overrides.items():
        if not isinstance(label, str) or not label:
            raise ValueError("C3D marker-weight labels must be non-empty strings")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"C3D marker weight for {label!r} must be positive and finite")
        normalized[label] = float(weight)
    options["stage2_marker_weight_overrides"] = normalized


def _resolve_option_paths(options: dict[str, object]) -> None:
    converted_name = options.get("converted_c3d_name")
    if converted_name is not None and not isinstance(converted_name, (str, Path)):
        raise ValueError("c3d_options.converted_c3d_name must be a string or path")
    for name in ("head_marker_corr_path", "pose_body_prior_path", "stage1_state_path"):
        value = options.get(name)
        if value is None:
            continue
        if not isinstance(value, (str, Path)):
            raise ValueError(f"c3d_options.{name} must be a string or path")
        resolved = Path(value).expanduser().resolve()
        if name == "stage1_state_path" and not resolved.is_file():
            raise FileNotFoundError(f"C3D Stage-I state not found: {resolved}")
        options[name] = str(resolved)


__all__ = ["DEFAULT_C3D_OPTIONS", "SUPPORTED_C3D_OPTIONS", "resolve_c3d_options"]
