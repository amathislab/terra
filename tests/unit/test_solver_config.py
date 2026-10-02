"""Configuration validation and scene defaults for the solver."""

import logging

import numpy as np
import pytest

from terra.assembly import SolveContext, SolveStage
from terra.defaults import (
    DEFAULT_FLAT_FOOT_ORIENT_WEIGHT,
    DEFAULT_FLAT_SELF_COLLISION_WEIGHT,
    DEFAULT_MTP_SMOOTH_WEIGHT,
    DEFAULT_SELF_COLLISION_WEIGHT,
    DEFAULT_TERRAIN_PENETRATION_TOLERANCE,
)
from terra.profiles import (
    MATCHED_TERRA_CORE_OVERRIDES,
    SolverConfig,
    active_qp_terms,
    all_active_qp_terms,
    resolve_method_profile,
    resolve_solver_config,
)


def test_unknown_solver_configuration_key_is_rejected():
    with pytest.raises(ValueError, match=r"unknown TERRA solver configuration.*step_szie"):
        resolve_solver_config("terra", {"step_szie": 0.1})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("solver_backend", "unregistered"),
        ("foot_mode", "sticky"),
        ("clearance_mode", "automatic"),
    ],
)
def test_controlling_solver_modes_are_validated_during_resolution(key, value):
    with pytest.raises(ValueError, match=key):
        resolve_solver_config("terra", {key: value})


def test_scene_resolution_materializes_profile_defaults():
    flat = resolve_solver_config("terra").for_scene(on_terrain=False)
    terrain = resolve_solver_config("terra").for_scene(on_terrain=True)

    assert flat.penetration_tolerance == 1e-3
    assert flat.foot_orient_weight == 0.0
    assert flat.selfpen_weight == DEFAULT_FLAT_SELF_COLLISION_WEIGHT
    assert flat.mtp_smooth_weight is None
    assert flat.sole_offset_mode == "off"
    assert flat.clearance_mode == "off"
    assert flat.ground_range == (-10.0, 10.0)
    assert flat.ground_size == 10

    assert terrain.penetration_tolerance == DEFAULT_TERRAIN_PENETRATION_TOLERANCE
    assert terrain.foot_orient_weight == 0.0
    assert terrain.selfpen_weight == DEFAULT_SELF_COLLISION_WEIGHT
    assert terrain.mtp_smooth_weight == DEFAULT_MTP_SMOOTH_WEIGHT
    assert terrain.sole_offset_mode == "on"
    assert terrain.clearance_mode == "source"
    assert terrain.ground_range == (-3.0, 3.0)
    assert terrain.ground_size == 8


@pytest.mark.parametrize("on_terrain", [False, True])
def test_typed_resolution_preserves_active_term_selection(on_terrain):
    overrides = {
        "orient_weight": 2.0,
        "foot_velocity_weight": 0.0,
        "foot_velocity_tracking_weight": 0.0,
        "stance_height_weight": 25.0,
        "selfpen_mode": "off",
    }
    mapping = resolve_method_profile("terra", overrides)
    mapping.setdefault("foot_orient_mode", "on")
    mapping.setdefault("foot_orient_weight", 2.0 if on_terrain else DEFAULT_FLAT_FOOT_ORIENT_WEIGHT)
    typed = resolve_solver_config("terra", overrides).for_scene(on_terrain=on_terrain)

    assert active_qp_terms(typed, on_terrain=on_terrain) == active_qp_terms(
        mapping,
        on_terrain=on_terrain,
    )
    assert all_active_qp_terms(typed, on_terrain=on_terrain) == all_active_qp_terms(
        mapping,
        on_terrain=on_terrain,
    )


def test_omniretarget_profile_stays_contribution_free_after_typed_resolution():
    config = resolve_solver_config(
        "omniretarget",
        {"orient_weight": 999.0, "selfpen_mode": "legs", "posthoc_repair": True},
    ).for_scene(on_terrain=True)

    assert active_qp_terms(config) == []
    assert config.stance_height_weight == 0.0
    assert config.foot_anchor_weight == 50.0
    assert config.torso_orient_weight == 1.0
    assert config.posthoc_repair is False
    assert config.posthoc_tendon_repair is False


def test_matched_terra_core_keeps_pipeline_smoothing_and_repair():
    config = resolve_solver_config("terra", MATCHED_TERRA_CORE_OVERRIDES).for_scene(on_terrain=True)

    assert active_qp_terms(config) == ["mtp_smoothing"]
    assert all_active_qp_terms(config) == [
        "interaction_mesh",
        "joint_limits",
        "omniretarget_object_nonpenetration",
        "omniretarget_foot_sticking",
        "trunk_q_regularizer",
        "mtp_smoothing",
    ]
    assert config.solver_backend == "native_clarabel"
    assert config.warmup_frames > 0
    assert config.mtp_smooth_weight == DEFAULT_MTP_SMOOTH_WEIGHT
    assert config.posthoc_repair is True
    assert config.posthoc_tendon_repair is True


def test_solve_context_rejects_out_of_order_stage_transition():
    context = SolveContext(
        config=SolverConfig.from_mapping(),
        logger=logging.getLogger("test_solver_stage"),
        model=None,
        terrain=None,
        fps=100.0,
        joints_mapping={},
        human_joints=np.zeros((1, 1, 3)),
        smpl_rotations=np.zeros((1, 1, 3, 3)),
        scene={},
    )

    context.advance(SolveStage.PREPARED, SolveStage.CONSTRAINTS_ATTACHED)
    with pytest.raises(RuntimeError, match="invalid solver stage transition"):
        context.advance(SolveStage.PREPARED, SolveStage.SOLVED)


def test_public_default_weights_match_terrain_profile():
    from terra import defaults
    from terra.profiles import TERRA_TERRAIN_DEFAULTS

    pairs = {
        "orient_weight": defaults.DEFAULT_ORIENT_WEIGHT,
        "foot_anchor_weight": defaults.DEFAULT_FOOT_ANCHOR_WEIGHT,
        "foot_velocity_weight": defaults.DEFAULT_FOOT_VELOCITY_WEIGHT,
        "foot_velocity_tracking_weight": defaults.DEFAULT_FOOT_VELOCITY_TRACKING_WEIGHT,
        "stance_height_weight": defaults.DEFAULT_STANCE_HEIGHT_WEIGHT,
        "coupler_weight": defaults.DEFAULT_COUPLER_WEIGHT,
    }
    assert pairs == {key: TERRA_TERRAIN_DEFAULTS[key] for key in pairs}
