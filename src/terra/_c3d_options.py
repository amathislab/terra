"""Validation and normalization for the public C3D fitting configuration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

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


def resolve_c3d_options(
    c3d_options: Mapping[str, object] | None,
    *,
    c3d_model_path: str | Path | None,
    smpl_model_path: str | Path | None,
    cache_root: str | Path | None,
) -> dict[str, object]:
    """Forward fitter options and add paths controlled by TERRA's launcher."""

    supplied = dict(c3d_options or {})
    managed = _MANAGED_OPTIONS & supplied.keys()
    if managed:
        names = ", ".join(sorted(managed))
        raise ValueError(f"c3d_options cannot override managed option(s): {names}")

    options = dict(DEFAULT_C3D_OPTIONS)
    options.update(supplied)
    for name in ("gender", "surface_model_type", "stage1_shape_solver", "stage2_solver", "device"):
        if isinstance(options[name], str):
            options[name] = options[name].casefold()
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


__all__ = ["DEFAULT_C3D_OPTIONS", "resolve_c3d_options"]
