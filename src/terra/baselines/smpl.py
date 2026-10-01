"""SMPL optimization baseline adapter.

TERRA owns the baseline definition and cache namespace. MuscleMimic supplies the
method-neutral implementation used to fit a motion to its robot model.
"""

from __future__ import annotations

from collections.abc import Mapping

from terra._musclemimic import fit_smpl_motion as _fit_smpl_motion
from terra.baselines._spec import BaselineSpec
from terra.baselines.temporal import resample_smplh_motion

SMPL_BASELINE = BaselineSpec(
    key="smpl",
    label="MuscleMimic SMPL-fit",
    dependency_extra="baselines",
    config={"algorithm": "smpl", "target_fps": None, "skip_steps": True, "visualize": False},
)


def fit_motion(
    env_name,
    robot_conf,
    path_to_smpl_model,
    motion_data,
    path_to_optimized_smpl_shape,
    logger,
    config: Mapping[str, object] | None = None,
    *,
    skip_steps: bool | None = None,
    visualize: bool | None = None,
):
    """Run the MuscleMimic SMPL optimization baseline."""
    supplied = dict(config or {})
    if skip_steps is not None:
        supplied["skip_steps"] = skip_steps
    if visualize is not None:
        supplied["visualize"] = visualize
    unknown = sorted(set(supplied) - set(SMPL_BASELINE.config))
    if unknown:
        raise ValueError(f"unknown MM-SMPL configuration field(s): {', '.join(unknown)}")
    resolved = SMPL_BASELINE.resolved_config(supplied)
    resolved.pop("algorithm")
    target_fps = resolved.pop("target_fps")
    skip_steps = resolved.pop("skip_steps")
    visualize = resolved.pop("visualize")
    if not isinstance(skip_steps, bool) or not isinstance(visualize, bool):
        raise ValueError("MM-SMPL skip_steps and visualize must be booleans")

    report = {}
    if target_fps is not None:
        motion_data, report = resample_smplh_motion(motion_data, target_fps)
        # With an exact target rate, every resampled input frame enters the optimizer.
        skip_steps = False

    trajectory, analysis = _fit_smpl_motion(
        env_name,
        robot_conf,
        path_to_smpl_model,
        motion_data,
        path_to_optimized_smpl_shape,
        logger,
        skip_steps=skip_steps,
        visualize=visualize,
    )
    if report:
        analysis = dict(analysis) | report
    return trajectory, analysis


__all__ = ["SMPL_BASELINE", "fit_motion"]
