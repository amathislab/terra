"""Tests for TERRA's terrain-aware extensions to the OmniRetarget QP.

Every optional cost is linearized into one shared least-squares representation used by
the legacy CVXPY, condensed CVXPY, and native Clarabel backends.

These guard silent failures. A wrong frame or sign in the orientation Jacobian or the
coupler linearisation does not raise: the QP still solves, the cost still decreases, and
the retargeted motion is merely worse in a way only a full solve plus a diagnostic sweep
would reveal. Same for the contact ramp - a ramp that fails to reach zero at the edges
reintroduces exactly the engagement step it exists to remove, and nothing downstream would
complain.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from terra.assembly import SolveDiagnostics
from terra.baselines.omniretarget import OMNIRETARGET_INSTALLED
from terra.contacts import contact_ramp, source_foot_contact, source_foot_sticking
from terra.profiles import SolverConfig
from terra.retargeter import TerraRetargeter

mujoco = pytest.importorskip("mujoco")

# `_orientation_linearisation` calls `_build_transform_qdot_to_qvel_fast`, which lives on
# the OmniRetarget base class, so these need OmniRetarget importable.
requires_omni = pytest.mark.skipif(
    not OMNIRETARGET_INSTALLED, reason="OmniRetarget (holosoma_retargeting) not on the path"
)

PROBE_BODIES = ["torso", "humerus_l", "femur_r", "head"]


def test_myofullbody_site_calibration_schema_matches_cache_site_order():
    from musclemimic.environments.humanoids.myofullbody import MyoFullBody
    from terra.constants import MYOFULLBODY_SITE_CALIBRATION

    environment_mapping = MyoFullBody.body2sites_for_mimic.fget(None)
    expected = [(site, body) for site, _joint, body in MYOFULLBODY_SITE_CALIBRATION]

    assert expected == [(site, body) for body, site in environment_mapping.items()]


def test_site_calibration_rotates_local_offsets_per_frame():
    from loco_mujoco.smpl import SMPLH_BONE_ORDER_NAMES
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import MYOFULLBODY_SITE_CALIBRATION
    from terra.source import apply_site_calibration

    n_sites = len(MYOFULLBODY_SITE_CALIBRATION)
    joints = np.zeros((2, len(SMPLH_DEMO_JOINTS), 3))
    rotations = np.tile(np.eye(3), (2, len(SMPLH_BONE_ORDER_NAMES), 1, 1))
    position_offsets = np.zeros((n_sites, 3))
    rotation_offsets = np.tile(np.eye(3), (n_sites, 1, 1))

    head_site = next(
        index for index, (_site, joint, _body) in enumerate(MYOFULLBODY_SITE_CALIBRATION) if joint == "Head"
    )
    position_offsets[head_site] = [0.1, 0.0, 0.0]
    head_bone = SMPLH_BONE_ORDER_NAMES.index("Head")
    rotations[1, head_bone] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    calibrated = apply_site_calibration(
        joints,
        rotations,
        position_offsets,
        rotation_offsets,
        np.tile(np.eye(3), (len(SMPLH_BONE_ORDER_NAMES), 1, 1)),
    )

    head_demo = SMPLH_DEMO_JOINTS.index("Head")
    np.testing.assert_allclose(calibrated[0, head_demo], [-0.1, 0.0, 0.0])
    np.testing.assert_allclose(calibrated[1, head_demo], [0.0, -0.1, 0.0], atol=1e-15)
    untouched = [index for index in range(len(SMPLH_DEMO_JOINTS)) if index != head_demo]
    np.testing.assert_array_equal(calibrated[:, untouched], joints[:, untouched])


def test_landmark_normalization_records_exact_motion_only_floor_transform():
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.source import _landmark_normalization

    joints = np.zeros((3, len(SMPLH_DEMO_JOINTS), 3), dtype=float)
    left = SMPLH_DEMO_JOINTS.index("L_Toe")
    right = SMPLH_DEMO_JOINTS.index("R_Toe")
    joints[:, left, 2] = [0.08, -0.04, 0.02]
    joints[:, right, 2] = [0.03, 0.01, 0.07]

    record = _landmark_normalization(joints, 0.75)

    assert record["source_floor_z_m"] == pytest.approx(-0.04)
    assert record["uniform_scale"] == pytest.approx(0.75)
    assert record["source_to_normalized_translation_m"] == pytest.approx([0.0, 0.0, 0.03])


def test_numerical_output_envelope_accepts_ordinary_gait_and_reports_extrema():
    from terra.pipeline import validate_numerical_output

    source = np.column_stack([np.arange(5) * 0.02, np.zeros(5), np.ones(5)])
    qpos = np.zeros((5, 8))
    qpos[:, :3] = source + np.array([0.01, -0.02, 0.10])
    qpos[:, 3] = 1.0

    report = validate_numerical_output(qpos, source)

    assert report["max_root_step_m"] == pytest.approx(0.02)
    assert report["max_root_source_deviation_m"] == pytest.approx(np.sqrt(0.0105))


@pytest.mark.parametrize("failure", ["step", "source", "finite"])
def test_numerical_output_envelope_rejects_escaped_solve(failure):
    from terra.pipeline import validate_numerical_output

    source = np.zeros((4, 3))
    qpos = np.zeros((4, 8))
    qpos[:, 3] = 1.0
    if failure == "step":
        qpos[2:, 0] = 2.0
    elif failure == "source":
        qpos[:, 1] = 8.0
    else:
        qpos[1, 7] = np.nan

    with pytest.raises(ValueError, match=r"numerical envelope|non-finite"):
        validate_numerical_output(qpos, source)


def test_condensed_laplacian_qp_matches_auxiliary_formulation():
    """Eliminating the equality-bound Laplacian variable must preserve the QP optimum."""
    import cvxpy as cp
    from scipy import sparse

    from terra._qp import _native_clarabel_qp

    rng = np.random.default_rng(41)
    n_dof, n_laplacian = 9, 36
    jac = rng.normal(size=(n_laplacian, n_dof))
    lap0 = rng.normal(size=n_laplacian)
    target = rng.normal(size=n_laplacian)
    sqrt_weight = rng.uniform(0.2, 2.0, size=n_laplacian)
    regularizer = rng.uniform(0.1, 1.0, size=n_dof)

    dqa_aux = cp.Variable(n_dof)
    lap_aux = cp.Variable(n_laplacian)
    common_aux = [dqa_aux >= -0.15, dqa_aux <= 0.15, cp.SOC(0.3, dqa_aux)]
    aux_problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(cp.multiply(sqrt_weight, lap_aux - target))
            + cp.sum_squares(cp.multiply(regularizer, dqa_aux))
        ),
        [*common_aux, jac @ dqa_aux - lap_aux == -lap0],
    )
    aux_problem.solve(solver=cp.CLARABEL)

    dqa_condensed = cp.Variable(n_dof)
    condensed_problem = cp.Problem(
        cp.Minimize(
            cp.sum_squares(cp.multiply(sqrt_weight, jac @ dqa_condensed + lap0 - target))
            + cp.sum_squares(cp.multiply(regularizer, dqa_condensed))
        ),
        [dqa_condensed >= -0.15, dqa_condensed <= 0.15, cp.SOC(0.3, dqa_condensed)],
    )
    condensed_problem.solve(solver=cp.CLARABEL)

    bounds = np.vstack([np.eye(n_dof), -np.eye(n_dof)])
    bound_rhs = np.r_[np.full(n_dof, -0.15), np.full(n_dof, -0.15)]
    native_x, native_cost, _ = _native_clarabel_qp(
        n_dof,
        [
            (jac, target - lap0, sqrt_weight**2),
            (np.eye(n_dof), np.zeros(n_dof), regularizer**2),
        ],
        (bounds, bound_rhs),
        0.3,
    )
    native_sparse_x, native_sparse_cost, _ = _native_clarabel_qp(
        n_dof,
        [
            (jac, target - lap0, sqrt_weight**2),
            (np.eye(n_dof), np.zeros(n_dof), regularizer**2),
        ],
        (sparse.csr_matrix(bounds), bound_rhs),
        0.3,
    )

    assert aux_problem.status == cp.OPTIMAL
    assert condensed_problem.status == cp.OPTIMAL
    np.testing.assert_allclose(dqa_condensed.value, dqa_aux.value, atol=2e-7, rtol=2e-7)
    assert condensed_problem.value == pytest.approx(aux_problem.value, abs=2e-7, rel=2e-7)
    np.testing.assert_allclose(native_x, dqa_aux.value, atol=2e-6, rtol=2e-6)
    assert native_cost == pytest.approx(aux_problem.value, abs=2e-7, rel=2e-7)
    np.testing.assert_array_equal(native_sparse_x, native_x)
    assert native_sparse_cost == native_cost


def test_identity_least_squares_specialization_is_bit_exact():
    """Analytic identity accumulation must reproduce dense identity products exactly."""
    from terra._qp import (
        _add_weighted_identity_least_squares,
        _add_weighted_least_squares,
    )

    rng = np.random.default_rng(84)
    n_dof = 89
    target = rng.normal(size=n_dof)
    for weight in (0.73, rng.uniform(0.1, 2.0, size=n_dof)):
        expected_hessian = rng.normal(size=(n_dof, n_dof))
        actual_hessian = expected_hessian.copy()
        expected_linear = rng.normal(size=n_dof)
        actual_linear = expected_linear.copy()
        expected_constant = _add_weighted_least_squares(
            expected_hessian,
            expected_linear,
            2.5,
            np.eye(n_dof),
            target,
            weight,
        )
        actual_constant = _add_weighted_identity_least_squares(actual_hessian, actual_linear, 2.5, target, weight)
        np.testing.assert_array_equal(actual_hessian, expected_hessian)
        np.testing.assert_array_equal(actual_linear, expected_linear)
        assert actual_constant == expected_constant


def test_native_csc_fast_assembly_is_bit_exact():
    """Direct CSC builders must reproduce SciPy's former dense conversion arrays."""
    from scipy import sparse

    from terra._qp import (
        _dense_inequality_soc_csc,
        _symmetric_upper_csc,
    )

    rng = np.random.default_rng(94)
    for n_dof, n_rows in ((1, 0), (9, 17), (89, 188)):
        hessian = rng.normal(size=(n_dof, n_dof))
        hessian[rng.random(hessian.shape) < 0.35] = 0.0
        expected_p = sparse.csc_matrix(np.triu(0.5 * (hessian + hessian.T)))
        actual_p = _symmetric_upper_csc(hessian)

        inequalities = rng.normal(size=(n_rows, n_dof))
        inequalities[rng.random(inequalities.shape) < 0.85] = 0.0
        expected_a = sparse.csc_matrix(np.vstack([-inequalities, np.zeros((1, n_dof)), -np.eye(n_dof)]))
        actual_a = _dense_inequality_soc_csc(inequalities)

        for expected, actual in ((expected_p, actual_p), (expected_a, actual_a)):
            assert actual.shape == expected.shape
            np.testing.assert_array_equal(actual.data, expected.data)
            np.testing.assert_array_equal(actual.indices, expected.indices)
            np.testing.assert_array_equal(actual.indptr, expected.indptr)


def test_fast_target_laplacian_is_bit_exact_for_both_weight_modes():
    """Removing dead uniform-weight distances must preserve OmniRetarget's arithmetic."""
    from holosoma_retargeting.src import interaction_mesh_retargeter as omniretarget

    from terra._interaction_mesh import _calculate_laplacian_coordinates

    rng = np.random.default_rng(52)
    vertices = rng.normal(size=(37, 3))
    adjacency = []
    for index in range(len(vertices)):
        candidates = np.delete(np.arange(len(vertices)), index)
        degree = index % 12
        adjacency.append(list(rng.choice(candidates, size=degree, replace=False)))

    for uniform_weight in (True, False):
        expected = omniretarget.calculate_laplacian_coordinates(vertices, adjacency, uniform_weight=uniform_weight)
        actual = _calculate_laplacian_coordinates(vertices, adjacency, uniform_weight=uniform_weight)
        np.testing.assert_array_equal(actual, expected)


def test_fast_adjacency_preserves_omniretarget_neighbor_order():
    """The explicit six-edge loop must preserve even Python set iteration order."""
    from holosoma_retargeting.src import interaction_mesh_retargeter as omniretarget

    from terra._interaction_mesh import _get_adjacency_list

    tetrahedra = np.array(
        [
            [17, 3, 21, 8],
            [8, 3, 29, 17],
            [29, 4, 21, 17],
            [21, 4, 8, 3],
            [8, 4, 29, 3],
        ],
        dtype=np.int32,
    )
    expected = omniretarget.get_adjacency_list(tetrahedra, 30)
    actual = _get_adjacency_list(tetrahedra, 30)
    assert actual == expected


def test_dense_least_squares_assembly_matches_sparse_oracle():
    """The fast dense normal equations must match the retained sparse implementation."""
    from scipy import sparse

    from terra._qp import _add_weighted_least_squares

    rng = np.random.default_rng(52)
    jacobian = rng.normal(size=(23, 11))
    target = rng.normal(size=23)
    weights = rng.uniform(0.1, 4.0, size=23)
    results = []
    for representation in (jacobian, sparse.csr_matrix(jacobian)):
        hessian = np.zeros((11, 11))
        linear = np.zeros(11)
        constant = _add_weighted_least_squares(hessian, linear, 0.0, representation, target, weights)
        results.append((hessian, linear, constant))

    np.testing.assert_allclose(results[0][0], results[1][0], atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-13, rtol=2e-13)
    assert results[0][2] == pytest.approx(results[1][2], abs=2e-13, rel=2e-13)


@requires_omni
def test_interaction_laplacian_cache_is_qualified_by_topology(monkeypatch):
    """Vertex motion reuses uniform operators; an adjacency change invalidates them."""
    from holosoma_retargeting.src import interaction_mesh_retargeter as _imr

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    original = _imr.calculate_laplacian_matrix
    calls = 0

    def counted(vertices, adj_list):
        nonlocal calls
        calls += 1
        return original(vertices, adj_list)

    monkeypatch.setattr(_imr, "calculate_laplacian_matrix", counted)
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    adjacency = [[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]]
    first = retargeter._interaction_laplacian_operators(vertices, adjacency)
    second = retargeter._interaction_laplacian_operators(vertices + 100.0, adjacency)

    assert calls == 1
    assert second[0] is first[0]
    assert second[1] is first[1]

    changed = [[1], [0, 2], [1, 3], [2, 4], [3]]
    third = retargeter._interaction_laplacian_operators(vertices, changed)
    assert calls == 2
    assert third[0] is not first[0]
    assert third[1] is not first[1]


def test_interaction_laplacian_cache_uses_identity_only_in_validated_frame_scope(monkeypatch):
    """SQP fast hits skip content conversion; direct calls still detect in-place edits."""
    from holosoma_retargeting.src import interaction_mesh_retargeter as _imr

    class CountingAdjacency(list):
        iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    original = _imr.calculate_laplacian_matrix
    calls = 0

    def counted(vertices, adj_list):
        nonlocal calls
        calls += 1
        return original(vertices, adj_list)

    monkeypatch.setattr(_imr, "calculate_laplacian_matrix", counted)
    vertices = np.arange(15, dtype=float).reshape(5, 3)
    adjacency = CountingAdjacency([[1, 2], [0, 2], [0, 1, 3], [2, 4], [3]])
    retargeter._laplacian_topology_scope_active = True
    retargeter._laplacian_scoped_adjacency = None
    retargeter._laplacian_scoped_vertex_count = None

    first = retargeter._interaction_laplacian_operators(vertices, adjacency)
    iterations_after_validation = adjacency.iterations
    second = retargeter._interaction_laplacian_operators(vertices + 1.0, adjacency)
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert adjacency.iterations == iterations_after_validation
    assert calls == 1

    retargeter._laplacian_topology_scope_active = False
    adjacency[0] = [1]
    third = retargeter._interaction_laplacian_operators(vertices, adjacency)
    assert calls == 2
    assert third[0] is not first[0]
    assert third[1] is not first[1]


def test_foot_orientation_weight_does_not_weaken_other_sites():
    from terra.assembly import _orientation_site_weights

    names = ["head_mimic", "left_ankle_mimic", "left_toes_mimic", "right_wrist_mimic"]
    weights = _orientation_site_weights(names, orient_weight=1.0, foot_orient_weight=0.25)
    assert weights.tolist() == [1.0, 0.25, 0.25, 1.0]
    assert _orientation_site_weights(names, 0.7, None).tolist() == [0.7] * len(names)


def test_upper_orientation_weight_does_not_weaken_pelvis_legs_or_feet():
    from terra.assembly import _orientation_site_weights

    names = ["pelvis_mimic", "upper_body_mimic", "left_hand_mimic", "left_ankle_mimic"]
    weights = _orientation_site_weights(names, orient_weight=1.0, foot_orient_weight=0.25, upper_orient_weight=0.05)
    assert weights.tolist() == [1.0, 0.05, 0.05, 0.25]


def test_position_torso_target_is_separate_from_arm_orientation_authority():
    from scipy.spatial.transform import Rotation

    from terra.assembly import (
        _orientation_site_weights,
        position_torso_orientation_targets,
    )

    names = ["Pelvis", "L_Shoulder", "R_Shoulder"]
    joints = np.zeros((2, 3, 3))
    joints[:, names.index("L_Shoulder")] = [0.0, 0.2, 1.0]
    joints[:, names.index("R_Shoulder")] = [0.0, -0.2, 1.0]
    yaw = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    joints[1, 1:] = np.einsum("ij,kj->ki", yaw, joints[1, 1:])
    reference = Rotation.from_euler("x", 10, degrees=True).as_matrix()

    targets = position_torso_orientation_targets(joints, names, reference)

    np.testing.assert_allclose(targets[0], reference, atol=1e-12)
    np.testing.assert_allclose(targets[1], yaw @ reference, atol=1e-12)
    weights = _orientation_site_weights(
        ["upper_body_mimic", "left_hand_mimic", "pelvis_mimic"],
        orient_weight=1.0,
        foot_orient_weight=0.0,
        upper_orient_weight=0.0,
        torso_orient_weight=2.0,
    )
    assert weights.tolist() == [2.0, 0.0, 1.0]


def test_source_foot_sticking_matches_horizontal_speed_at_any_rate_and_ignores_height():
    names = ["L_Toe", "R_Toe"]

    def sticking(fps, x):
        joints = np.zeros((4, 2, 3))
        joints[:, :, 2] = 2.0  # a slow airborne foot is damped, never absolutely anchored
        joints[:, 0, 0] = x
        joints[:, 1, 0] = x
        return source_foot_sticking(joints, names, names, fps, speed_ms=0.3, release_guard_frames=0)

    at_100_hz = sticking(100.0, [0.0, 0.0015, 0.0055, 0.0070])
    at_50_hz = sticking(50.0, [0.0, 0.0030, 0.0110, 0.0140])
    expected = np.array([False, True, False, True])
    for name in names:
        assert np.array_equal(at_100_hz[name], expected)
        assert np.array_equal(at_50_hz[name], expected)


def test_source_foot_sticking_guards_the_first_release_sample_for_resampling():
    joints = np.zeros((6, 1, 3))
    # Slow on frames 1--3, then fast.  Only the first fast sample is guarded.
    joints[:, 0, 0] = [0.0, 0.001, 0.002, 0.003, 0.013, 0.023]
    sticking = source_foot_sticking(joints, ["L_Toe"], ["L_Toe"], 100.0, speed_ms=0.3, release_guard_frames=1)
    assert sticking["L_Toe"].tolist() == [False, True, True, True, True, False]


@pytest.mark.parametrize("fps,speed", [(0.0, 0.3), (100.0, 0.0), (np.inf, 0.3)])
def test_source_foot_sticking_rejects_invalid_physical_units(fps, speed):
    with pytest.raises(ValueError):
        source_foot_sticking(np.zeros((2, 1, 3)), ["L_Toe"], ["L_Toe"], fps, speed)


@pytest.mark.parametrize("guard", [-1, 0.5, True])
def test_source_foot_sticking_rejects_invalid_release_guard(guard):
    with pytest.raises(ValueError, match="release_guard_frames"):
        source_foot_sticking(
            np.zeros((2, 1, 3)),
            ["L_Toe"],
            ["L_Toe"],
            100.0,
            release_guard_frames=guard,
        )


def test_foot_orientation_weight_default_is_low_only_on_flat():
    from terra.assembly import _default_foot_orientation_weight

    assert _default_foot_orientation_weight(False, 1.0) == 0.1
    assert _default_foot_orientation_weight(True, 1.0) == 1.0
    assert _default_foot_orientation_weight(True, 0.25) == 0.25


@pytest.mark.parametrize("value", [-0.1, np.nan, np.inf])
def test_foot_orientation_weight_rejects_invalid_authority(value):
    from terra.assembly import _orientation_site_weights

    with pytest.raises(ValueError, match="foot_orient_weight"):
        _orientation_site_weights(["left_ankle_mimic"], 1.0, value)


class _Stub(TerraRetargeter):
    """Only the attributes `_orientation_linearisation` touches - no solver, no XML reload."""

    def __init__(self, model):
        self._initialize_terra_state()
        self.robot_model = model
        self.robot_data = mujoco.MjData(model)
        self.nq = model.nq
        self.q_a_indices = np.arange(7, model.nq)
        self.nq_a = len(self.q_a_indices)
        self.has_dynamic_object = False


@requires_omni
def test_native_retargeting_step_moves_a_real_mujoco_body_toward_target():
    """Exercise the assembled native objective and Clarabel with a MuJoCo Jacobian."""
    from terra._sqp import _InequalityConstraints, _LaplacianLinearization

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 1">
              <freejoint/>
              <geom type="sphere" size="0.05" mass="1"/>
              <body name="pivot">
                <joint name="hinge" type="hinge" axis="0 0 1"/>
                <geom type="sphere" size="0.05" mass="1"/>
                <body name="tip" pos="1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    retargeter = _Stub(model)
    retargeter._solver_backend = "native_clarabel"
    retargeter.Q_diag = np.array([0.01])
    retargeter.smooth_weight = 0.01
    retargeter.track_nominal_indices = []
    q = model.qpos0.copy()
    jacobians, positions, _ = retargeter._calc_manipulator_jacobians(q, {"tip": "tip"})
    initial_position = positions["tip"]
    target_position = initial_position + np.array([0.0, 0.1, 0.0])
    laplacian = _LaplacianLinearization(
        jacobian=jacobians["tip"],
        current=initial_position,
        target=target_position,
        row_scale=np.ones(3),
    )
    constraints = _InequalityConstraints(retargeter.nq_a)
    constraints.add_variable_bounds(np.array([-0.2]), np.array([0.2]))

    result = retargeter._try_native_condensed_solve(
        q,
        q[retargeter.q_a_indices],
        np.zeros(retargeter.nq_a),
        laplacian,
        constraints,
        [],
        {"w_nominal_tracking": 0.0, "q_a_nominal": None, "verbose": False, "frame_idx": 0},
        trust_radius=0.2,
    )

    assert result is not None
    updated_q, objective = result
    _, updated_positions, _ = retargeter._calc_manipulator_jacobians(updated_q, {"tip": "tip"})
    assert updated_q[7] > 0.0
    assert np.linalg.norm(updated_positions["tip"] - target_position) < np.linalg.norm(
        initial_position - target_position
    )
    assert np.isfinite(objective)
    assert retargeter._native_fallback_count == 0


@requires_omni
@pytest.mark.parametrize("method_profile", ["omniretarget", "terra"])
def test_omniretarget_static_scene_binding_selects_terrain_box_collisions(
    monkeypatch, method_profile
):
    """The inherited object constraint must see fitted ramps in matched-core mode."""
    from terra.pipeline import _bind_omniretarget_scene_collision

    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        contype=1,
        conaffinity=1,
    )
    spec.worldbody.add_geom(
        name="terrain_box_ramp",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0, 0, 0.1],
        size=[0.5, 0.5, 0.1],
        contype=1,
        conaffinity=1,
    )
    body = spec.worldbody.add_body(name="probe")
    body.add_freejoint()
    body.add_geom(
        name="probe_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.1, 0, 0],
        contype=1,
        conaffinity=1,
    )
    model = spec.compile()
    retargeter = _Stub(model)
    retargeter.object_name = "ground"
    retargeter.collision_detection_threshold = 0.1
    retargeter.penetration_tolerance = 0.005
    environment = retargeter._constraints.environment
    environment.geom_ids = None
    retargeter._geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "" for geom_id in range(model.ngeom)
    ]
    retargeter._geom_names_model_cache = model
    monkeypatch.setattr(
        retargeter,
        "_compute_jacobian_for_contact_relative",
        lambda *_args: np.zeros((3, retargeter.nq_a)),
    )
    qpos = np.array(model.qpos0, dtype=float)
    qpos[2] = 0.2
    qpos[3] = 1.0
    terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_box_ramp")

    before, _ = retargeter._update_jacobians_and_phis_from_q(qpos)
    assert not any(terrain_id in pair for pair in before)

    ctx = SimpleNamespace(
        config=SolverConfig.from_mapping(
            {
                "method_profile": method_profile,
                "foot_mode": "omniretarget",
                "activate_obj_non_penetration": True,
            }
        ).for_scene(on_terrain=True),
        diagnostics=SolveDiagnostics(),
        on_terrain=True,
        model=model,
        logger=SimpleNamespace(info=lambda *_args: None),
    )
    _bind_omniretarget_scene_collision(ctx, retargeter)
    after, _ = retargeter._update_jacobians_and_phis_from_q(qpos)

    assert any(terrain_id in pair for pair in after)
    assert retargeter.object_name == "terrain_box"
    assert ctx.diagnostics.omniretarget_collision_geom_count == 1


@requires_omni
def test_retargeter_constructor_owns_optional_state(monkeypatch):
    """Every solver instance must own its mutable targets and diagnostics."""
    base = TerraRetargeter.__mro__[1]
    calls = []

    def record_base_init(instance, **kwargs):
        calls.append((instance, kwargs))

    monkeypatch.setattr(base, "__init__", record_base_init)
    constants = SimpleNamespace()
    first = TerraRetargeter(constants, None)
    second = TerraRetargeter(constants, None)

    assert [instance for instance, _kwargs in calls] == [first, second]
    assert calls[0][1]["task_constants"] is constants
    assert calls[0][1]["object_urdf_path"] is None
    first_state = first._constraints
    second_state = second._constraints
    mutable_pairs = (
        (first_state.objective.clearance.geoms, second_state.objective.clearance.geoms),
        (first_state.objective.clearance.targets, second_state.objective.clearance.targets),
        (first_state.objective.clearance.activation, second_state.objective.clearance.activation),
        (first_state.objective.route.targets, second_state.objective.route.targets),
        (first_state.objective.route.activation, second_state.objective.route.activation),
        (first_state.objective.stance_height.targets, second_state.objective.stance_height.targets),
        (first_state.objective.stance_height.activation, second_state.objective.stance_height.activation),
        (first_state.objective.seat_contact.activation, second_state.objective.seat_contact.activation),
        (first_state.objective.self_collision.pairs, second_state.objective.self_collision.pairs),
        (first_state.environment.relaxed_frames, second_state.environment.relaxed_frames),
        (first._native_fallback_frames, second._native_fallback_frames),
    )
    assert all(first_value is not second_value for first_value, second_value in mutable_pairs)

    first_state.objective.route.targets[3] = np.ones(1)
    first_state.objective.self_collision.pairs.append((1, 2))
    first._native_fallback_frames.add(7)
    assert second_state.objective.route.targets == {}
    assert second_state.objective.self_collision.pairs == []
    assert second._native_fallback_frames == set()
    assert not second._has_attached_quadratic_terms()


@pytest.fixture(scope="module")
def probed_model():
    """MyoFullBody with a probe site on each test body, at a deliberately non-identity pose.

    Non-identity because a site frame that differs from its body frame is the case the
    implementation must handle; the shipped mimic sites happen to be identity today, which
    would let a body-frame bug pass unnoticed.
    """
    from musclemimic_models import get_xml_path

    from musclemimic.utils.retarget.msk_metrics import apply_spec_changes

    spec = mujoco.MjSpec.from_file(str(get_xml_path("myofullbody")))
    spec = apply_spec_changes(spec)
    for i, body in enumerate(PROBE_BODIES):
        quat = Rotation.from_euler("xyz", [0.3 * (i + 1), -0.2 * (i + 1), 0.1]).as_quat()
        spec.body(body).add_site(
            name=f"probe_{body}",
            pos=[0.01 * (i + 1), 0.02, -0.01],
            quat=[quat[3], quat[0], quat[1], quat[2]],
        )
    model = spec.compile()
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"probe_{b}") for b in PROBE_BODIES])
    assert (ids >= 0).all()
    return model, ids


@pytest.fixture(scope="module")
def random_pose(probed_model):
    model, _ = probed_model
    rng = np.random.default_rng(0)
    q = np.zeros(model.nq)
    for j in range(model.njnt):
        if model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            a = int(model.jnt_qposadr[j])
            lo, hi = model.jnt_range[j]
            q[a] = rng.uniform(lo, hi) * 0.4 if model.jnt_limited[j] else rng.normal(0, 0.2)
    quat = Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_quat()
    q[3:7] = [quat[3], quat[0], quat[1], quat[2]]
    q[2] = 1.0
    return q


@requires_omni
def test_manipulator_point_jacobians_share_one_forward_pass(probed_model, monkeypatch):
    """A link batch shares one kinematics update and one velocity transform."""
    model, _ = probed_model
    stub = _Stub(model)
    q = np.array(model.qpos0, dtype=float)
    original = mujoco.mj_forward
    base_retargeter = TerraRetargeter.__mro__[1]
    original_transform = base_retargeter._build_transform_qdot_to_qvel_fast
    forward_calls = 0
    transform_calls = 0

    def counted_forward(*args):
        nonlocal forward_calls
        forward_calls += 1
        return original(*args)

    def counted_transform(self):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(self)

    monkeypatch.setattr(mujoco, "mj_forward", counted_forward)
    monkeypatch.setattr(base_retargeter, "_build_transform_qdot_to_qvel_fast", counted_transform)
    jacobians, positions, _ = stub._calc_manipulator_jacobians(
        q,
        links={name: name for name in PROBE_BODIES},
        obj_frame=False,
    )

    assert forward_calls == 1
    assert transform_calls == 1
    assert set(jacobians) == set(PROBE_BODIES)
    assert set(positions) == set(PROBE_BODIES)
    assert all(np.isfinite(jacobian).all() for jacobian in jacobians.values())

    stub._calc_manipulator_jacobians(q, links={name: name for name in PROBE_BODIES})
    assert forward_calls == 1
    assert transform_calls == 1

    stub.robot_data = mujoco.MjData(model)
    stub._calc_manipulator_jacobians(q, links={name: name for name in PROBE_BODIES})
    assert forward_calls == 2
    assert transform_calls == 2


@requires_omni
def test_contact_jacobian_reuses_fully_overwritten_position_scratch(probed_model):
    """MuJoCo must overwrite the reusable buffer; no rotational output is required."""
    model, _ = probed_model
    stub = _Stub(model)
    q = np.array(model.qpos0, dtype=float)
    stub._ensure_forward(q)
    stub._contact_forward_is_current = True
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PROBE_BODIES[0])
    point = np.array([0.013, -0.021, 0.034])

    expected = stub._calc_contact_jacobian_from_point(body_id, point)
    scratch = stub._contact_jacobian_position_scratch
    scratch[:] = np.nan
    actual = stub._calc_contact_jacobian_from_point(body_id, point)

    assert stub._contact_jacobian_position_scratch is scratch
    assert not np.isnan(scratch).any()
    np.testing.assert_array_equal(actual, expected)


def _site_rotations(model, data, q, site_ids):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    return np.array([data.site_xmat[s].reshape(3, 3).copy() for s in site_ids])


@requires_omni
def test_orientation_jacobian_matches_finite_differences(probed_model, random_pose):
    """`jac` must predict the world angular velocity a joint perturbation actually causes."""
    model, site_ids = probed_model
    stub = _Stub(model)
    n = len(site_ids)
    stub.attach_orientation_targets(site_ids, np.tile(np.eye(3), (1, n, 1, 1)), np.ones(n), PROBE_BODIES)

    error, jac = stub._orientation_linearisation(random_pose, 0)
    r0 = _site_rotations(model, stub.robot_data, random_pose, site_ids)
    expected_error = np.concatenate(
        [
            Rotation.from_matrix(target @ current.T).as_rotvec()
            for target, current in zip(stub._constraints.objective.orientation.targets[0], r0, strict=True)
        ]
    )
    np.testing.assert_array_equal(error, expected_error)

    eps = 1e-6
    worst = 0.0
    for col, qi in enumerate(stub.q_a_indices):
        qp = random_pose.copy()
        qp[qi] += eps
        r1 = _site_rotations(model, stub.robot_data, qp, site_ids)
        for i in range(n):
            fd = Rotation.from_matrix(r1[i] @ r0[i].T).as_rotvec() / eps
            worst = max(worst, float(np.abs(fd - jac[3 * i : 3 * i + 3, col]).max()))
    assert worst < 1e-5, f"Jacobian disagrees with finite differences by {worst:.2e}"


@requires_omni
def test_orientation_residual_vanishes_at_target(probed_model, random_pose):
    """Pins the direction of the error rotation: zero error when already on target."""
    model, site_ids = probed_model
    stub = _Stub(model)
    n = len(site_ids)
    stub.attach_orientation_targets(site_ids, np.tile(np.eye(3), (1, n, 1, 1)), np.ones(n), PROBE_BODIES)
    r0 = _site_rotations(model, stub.robot_data, random_pose, site_ids)
    stub._constraints.objective.orientation.targets = r0[None]

    e, _ = stub._orientation_linearisation(random_pose, 0)
    assert np.abs(e).max() < 1e-9


@requires_omni
def test_orientation_step_reduces_error(probed_model, random_pose):
    """End-to-end sign check: solving `jac dq = e` must move sites toward their targets."""
    model, site_ids = probed_model
    stub = _Stub(model)
    n = len(site_ids)
    rng = np.random.default_rng(1)
    stub.attach_orientation_targets(site_ids, np.tile(np.eye(3), (1, n, 1, 1)), np.ones(n), PROBE_BODIES)

    r0 = _site_rotations(model, stub.robot_data, random_pose, site_ids)
    perturb = np.array([Rotation.from_rotvec(rng.normal(0, 0.15, 3)).as_matrix() for _ in range(n)])
    stub._constraints.objective.orientation.targets = np.einsum("kij,kjl->kil", perturb, r0)[None]

    e, jac = stub._orientation_linearisation(random_pose, 0)
    dq = np.linalg.lstsq(jac, e, rcond=None)[0]
    dq *= min(1.0, 0.2 / (np.linalg.norm(dq) + 1e-12))  # respect the solver's trust region

    q_new = random_pose.copy()
    q_new[stub.q_a_indices] += dq
    r_new = _site_rotations(model, stub.robot_data, q_new, site_ids)

    before = np.linalg.norm(e)
    after = np.linalg.norm(
        [
            Rotation.from_matrix(stub._constraints.objective.orientation.targets[0, i] @ r_new[i].T).as_rotvec()
            for i in range(n)
        ]
    )
    assert after < before, f"step increased orientation error: {before:.4f} -> {after:.4f}"


@requires_omni
def test_foot_anchor_is_captured_from_the_previous_frame_not_the_current_iterate(probed_model):
    """The anchor must come from `q_t_last`, never from the in-progress SQP iterate.

    `iterate` calls the term builder up to 50 times per frame with a partially-solved `q`.
    Capturing the anchor from that would pin the foot wherever the solver happened to be
    mid-solve, for the whole contact.

    Today it happens to be safe: OmniRetarget seeds `iterate` with the previous frame's
    solution, so on a capture frame `q` and `q_t_last` coincide, and reading either is
    byte-identical. This test pins the *intent* rather than that coincidence - it passes
    two deliberately different poses, which OmniRetarget's loop never does, so it fails if the
    capture is ever wired back to `q`.
    """
    model, _ = probed_model
    stub = _Stub(model)
    stub.foot_links = {"left_foot": "calcn_l", "right_foot": "calcn_r"}
    stub.attach_foot_anchoring({"left_foot": np.ones(4, bool), "right_foot": np.ones(4, bool)}, 50.0, 0)

    # Two clearly different configurations: one stands for the converged previous frame,
    # the other for a half-solved iterate.
    q_last = np.zeros(model.nq)
    q_last[2] = 1.0
    q_last[3:7] = [1.0, 0.0, 0.0, 0.0]
    q_iterate = q_last.copy()
    q_iterate[0] += 0.5  # displace the whole body, so both feet move with it

    def foot_xy(q):
        return {
            k: v[:2].copy()
            for k, v in stub._calc_manipulator_jacobians(q, links=stub.foot_links, obj_frame=False)[1].items()
        }

    expected, wrong = foot_xy(q_last), foot_xy(q_iterate)
    assert not np.allclose(expected["left_foot"], wrong["left_foot"]), "test poses do not differ"

    stub._foot_anchor_terms(q_iterate, q_last, frame_idx=0)

    for label in stub.foot_links:
        assert np.allclose(stub._constraints.objective.foot_anchor.anchor[label], expected[label]), (
            f"{label} anchored to the current iterate instead of the previous frame"
        )


@requires_omni
def test_foot_velocity_term_penalizes_displacement_from_previous_frame(probed_model):
    model, _ = probed_model
    stub = _Stub(model)
    stub.foot_links = {"left_foot": "calcn_l", "right_foot": "calcn_r"}
    inactive_anchor = {label: np.zeros(2, bool) for label in stub.foot_links}
    velocity_contact = {label: np.array([False, True]) for label in stub.foot_links}
    stub.attach_foot_anchoring(
        inactive_anchor,
        weight=50.0,
        ramp_frames=8,
        velocity_contact=velocity_contact,
        velocity_weight=75.0,
        velocity_tolerance_m=0.02,
    )

    q_last = np.zeros(model.nq)
    q_last[2] = 1.0
    q_last[3] = 1.0
    q_iterate = q_last.copy()
    q_iterate[0] += 0.05

    terms = stub._foot_anchor_terms(q_iterate, q_last, frame_idx=1)
    assert len(terms) == 2
    for weight, jacobian, offset in terms:
        assert weight == pytest.approx(75.0)
        assert jacobian.shape == (2, stub.nq_a)
        assert offset == pytest.approx([0.03, 0.0], abs=1e-8)


@requires_omni
def test_foot_velocity_tracking_and_excess_terms_are_independent(probed_model):
    model, _ = probed_model
    stub = _Stub(model)
    stub.foot_links = {"left_foot": "calcn_l", "right_foot": "calcn_r"}
    velocity_contact = {label: np.array([False, True]) for label in stub.foot_links}
    stub.attach_foot_anchoring(
        {label: np.zeros(2, bool) for label in stub.foot_links},
        weight=50.0,
        ramp_frames=8,
        velocity_contact=velocity_contact,
        velocity_weight=75.0,
        velocity_tracking_weight=10.0,
        velocity_tolerance_m=0.02,
    )
    q_last = np.zeros(model.nq)
    q_last[2] = 1.0
    q_last[3] = 1.0
    q_iterate = q_last.copy()
    q_iterate[0] += 0.05
    terms = stub._foot_anchor_terms(q_iterate, q_last, frame_idx=1)
    assert [weight for weight, _, _ in terms] == [10.0, 75.0, 10.0, 75.0]
    assert terms[0][2] == pytest.approx([0.05, 0.0], abs=1e-8)
    assert terms[1][2] == pytest.approx([0.03, 0.0], abs=1e-8)


@requires_omni
def test_foot_velocity_term_is_inert_inside_speed_dead_zone(probed_model):
    model, _ = probed_model
    stub = _Stub(model)
    stub.foot_links = {"left_foot": "calcn_l", "right_foot": "calcn_r"}
    inactive_anchor = {label: np.zeros(2, bool) for label in stub.foot_links}
    velocity_contact = {label: np.array([False, True]) for label in stub.foot_links}
    stub.attach_foot_anchoring(
        inactive_anchor,
        weight=50.0,
        ramp_frames=8,
        velocity_contact=velocity_contact,
        velocity_weight=1000.0,
        velocity_tolerance_m=0.003,
    )

    q_last = np.zeros(model.nq)
    q_last[2] = 1.0
    q_last[3] = 1.0
    q_iterate = q_last.copy()
    q_iterate[0] += 0.002

    assert stub._foot_anchor_terms(q_iterate, q_last, frame_idx=1) == []


@requires_omni
def test_foot_velocity_term_uses_forefoot_site_rotation(probed_model):
    model, _ = probed_model
    stub = _Stub(model)
    stub.foot_links = {"left_foot": "calcn_l", "right_foot": "calcn_r"}
    contact = {label: np.array([False, True]) for label in stub.foot_links}
    stub.attach_foot_anchoring(
        {label: np.zeros(2, bool) for label in stub.foot_links},
        weight=50.0,
        ramp_frames=8,
        velocity_contact=contact,
        velocity_sites={"left_foot": "LTOE", "right_foot": "RTOE"},
        velocity_weight=1000.0,
    )

    q_last = model.qpos0.copy()
    stub._foot_anchor_terms(q_last, q_last, frame_idx=0)
    site_jac, site_pos = stub._foot_velocity_kinematics(q_last)
    assert set(site_jac) == set(stub.foot_links)
    assert all(jac.shape == (2, stub.nq_a) for jac in site_jac.values())
    assert all(pos.shape == (2,) for pos in site_pos.values())


@requires_omni
def test_combined_foot_kinematics_matches_omniretarget_body_helper(probed_model):
    model, _ = probed_model
    stub = _Stub(model)
    stub.foot_links = {"left_foot": "calcn_l", "right_foot": "calcn_r"}
    contact = {label: np.array([False, True]) for label in stub.foot_links}
    stub.attach_foot_anchoring(
        contact,
        weight=50.0,
        ramp_frames=8,
        velocity_contact=contact,
        velocity_sites={"left_foot": "LTOE", "right_foot": "RTOE"},
        velocity_weight=75.0,
    )
    q = model.qpos0.copy()
    q[0] += 0.12
    q[2] += 0.04
    expected_j, expected_p, _ = stub._calc_manipulator_jacobians(q, links=stub.foot_links, obj_frame=False)
    actual_j, actual_p, _, _ = stub._foot_kinematics(q)
    for label in stub.foot_links:
        assert actual_j[label] == pytest.approx(expected_j[label], abs=1e-12)
        assert actual_p[label] == pytest.approx(expected_p[label], abs=1e-12)


@requires_omni
@pytest.mark.parametrize(
    "weight,tracking,tolerance",
    [(-1.0, 0.0, 0.0), (1.0, -1.0, 0.0), (1.0, 0.0, -0.001), (np.nan, 0.0, 0.0)],
)
def test_foot_velocity_term_rejects_invalid_parameters(probed_model, weight, tracking, tolerance):
    model, _ = probed_model
    stub = _Stub(model)
    with pytest.raises(ValueError):
        stub.attach_foot_anchoring(
            {},
            weight=50.0,
            ramp_frames=8,
            velocity_weight=weight,
            velocity_tracking_weight=tracking,
            velocity_tolerance_m=tolerance,
        )


@requires_omni
def test_clearance_target_applies_to_soles_only_and_follows_the_frame(probed_model):
    """The requirement must reach the sole geoms, and no others, at the frame being solved.

    Applied to the wrong geom set it becomes a body-wide standoff that lifts the whole
    robot off its own terrain; read at the wrong frame it asks a planted foot for a swing's
    clearance. Neither raises - both come back as a solved QP and a subtly wrong motion.
    """
    model, _ = probed_model
    stub = _Stub(model)
    stub.collision_detection_threshold = 0.1

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    left = [g for g, n in enumerate(names) if n.startswith(("l_foot_col", "l_bofoot_col"))]
    right = [g for g, n in enumerate(names) if n.startswith(("r_foot_col", "r_bofoot_col"))]
    shin = next(g for g, n in enumerate(names) if n.startswith("l_tibia"))
    assert left and right, "test needs the sole geoms to resolve"

    inert = -np.inf
    targets = {"l": np.array([inert, 0.02, 0.04]), "r": np.full(3, inert)}
    stub.attach_foot_clearance({"l": left, "r": right}, targets)

    stub._current_frame = 2
    assert stub._clearance_target(left[0]) == pytest.approx(0.04)
    assert stub._clearance_target(right[0]) == inert, "the other foot must not be constrained"
    assert stub._clearance_target(shin) == inert, "only the soles carry a clearance requirement"

    stub._current_frame = 0
    assert stub._clearance_target(left[0]) == inert, "a planted foot must be free to touch down"

    # Past the end - the warm-up and any trailing frame must clamp, not wrap to frame 0.
    stub._current_frame = 99
    assert stub._clearance_target(left[0]) == pytest.approx(0.04)


@requires_omni
def test_clearance_cost_is_one_sided_and_points_upward(probed_model):
    """A foot already clearing its margin must contribute nothing.

    Two-sided, the term would pull a foot back *down* to the requested height - turning a
    minimum clearance into a target one, and fighting every step the source picks up higher
    than the margin asks for. The sign matters just as much: the shortfall has to be met by
    *increasing* the distance, and a flipped Jacobian would press the sole into the surface
    while the cost reported it improving.
    """
    model, _ = probed_model
    stub = _Stub(model)
    stub.collision_detection_threshold = 0.1
    stub.penetration_tolerance = 1e-3

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left = [g for g, n in enumerate(names) if n.startswith(("l_foot_col", "l_bofoot_col"))]
    stub._constraints.environment.geom_ids = {floor}

    q = np.array(model.qpos0, dtype=float)
    stub.robot_data.qpos[:] = q
    mujoco.mj_forward(model, stub.robot_data)
    standing = min(mujoco.mj_geomDistance(model, stub.robot_data, g, floor, 1.0, None) for g in left)
    assert standing < 0.06, "fixture pose has the feet nowhere near the floor"

    # A height the pose already meets: no terms at all, so nothing pulls the foot back down.
    stub.attach_foot_clearance({"l": left}, {"l": np.array([max(standing - 0.005, 0.0)])})
    stub._current_frame = 0
    assert stub._clearance_terms(q) == [], "a foot already at its height must cost nothing"

    # A height it does not meet: every term must ask for a positive lift.
    required = standing + 0.03
    stub.attach_foot_clearance({"l": left}, {"l": np.array([required])})
    terms = stub._clearance_terms(q)
    assert terms, "a sole below its required height must produce terms"
    for weight, jrow, shortfall in terms:
        assert weight > 0
        assert 0 < shortfall <= required + 1e-9, "shortfall is the height minus the current one"
        assert jrow.shape == (stub.nq_a,)

    # The reported shortfall must shrink as the foot is actually raised.
    lifted = q.copy()
    lifted[2] += 0.02
    worst_before = max(s for _, _, s in terms)
    lifted_terms = stub._clearance_terms(lifted)
    worst_after = max((s for _, _, s in lifted_terms), default=0.0)
    assert worst_after < worst_before, (
        f"raising the body must reduce the shortfall: {worst_before:.4f} -> {worst_after:.4f}"
    )


def test_self_collision_shortfall_is_capped_per_iteration(monkeypatch):
    """A newly deep inter-leg pair must not rearrange the entire leg in one QP step."""
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='left'><geom name='left_geom' type='sphere' "
        "size='.02'/></body><body name='right'><geom name='right_geom' type='sphere' "
        "size='.02'/></body></worldbody></mujoco>"
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter.collision_detection_threshold = 0.1
    retargeter._current_frame = 0
    retargeter._compute_jacobian_for_contact_relative = lambda *args: np.zeros(retargeter.robot_model.nq)
    retargeter.attach_self_collision(
        [("left", "right")],
        tolerance=0.002,
        weight=20000.0,
        max_recovery_per_iter=0.002,
    )
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: -0.010)
    terms = retargeter._self_collision_terms(retargeter.robot_model.qpos0.copy())
    assert len(terms) == 1
    assert terms[0][2] == pytest.approx(0.002), "the true 12 mm shortfall must be capped"

    for invalid in (0.0, -0.01, np.inf, np.nan):
        with pytest.raises(ValueError, match="max_recovery_per_iter"):
            retargeter.attach_self_collision([("left", "right")], max_recovery_per_iter=invalid)


def test_collision_prefilter_caches_names_and_invalidates_on_model_swap(monkeypatch):
    """Static geom names should be resolved once per compiled MuJoCo model."""
    model_xml = (
        "<mujoco><worldbody>"
        "<geom name='floor' type='plane' size='1 1 .1'/>"
        "<body><geom name='probe' type='sphere' size='.02'/></body>"
        "</worldbody></mujoco>"
    )
    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(model_xml)
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)

    original_id2name = mujoco.mj_id2name
    calls = []

    def counted_id2name(model, object_type, object_id):
        calls.append((model, object_type, object_id))
        return original_id2name(model, object_type, object_id)

    monkeypatch.setattr(mujoco, "mj_id2name", counted_id2name)
    threshold = 0.1
    first = retargeter._prefilter_pairs_with_mj_collision(threshold)
    second = retargeter._prefilter_pairs_with_mj_collision(threshold)
    assert second == first
    assert len(calls) == retargeter.robot_model.ngeom

    replacement = mujoco.MjModel.from_xml_string(model_xml)
    retargeter.robot_model = replacement
    retargeter.robot_data = mujoco.MjData(replacement)
    retargeter._prefilter_pairs_with_mj_collision(threshold)
    assert len(calls) == 2 * replacement.ngeom


def test_clearance_shortfall_is_capped_per_iteration(probed_model):
    """A large shortfall must be clamped to `max_recovery_per_iter`, not demanded whole.

    Mid-swing over a beam or box top, non-penetration can be inactive (nothing close
    enough to trigger it yet), so an uncapped clearance shortfall is the only force acting
    and gets undone in one unopposed linearised step - traced to a 56.9 mm single-iteration
    pelvis jump on BEAM01 (`omniretarget-tendon-jump-is-pelvis-pop` memory). The cap plays
    the same role `attach_environment_geoms`'s `max_recovery_per_iter` already plays for
    non-penetration: spread a large correction over more iterations instead of one.
    """
    model, _ = probed_model
    stub = _Stub(model)
    stub.collision_detection_threshold = 0.1
    stub.penetration_tolerance = 1e-3

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left = [g for g, n in enumerate(names) if n.startswith(("l_foot_col", "l_bofoot_col"))]

    q = np.array(model.qpos0, dtype=float)
    stub.robot_data.qpos[:] = q
    mujoco.mj_forward(model, stub.robot_data)
    standing = min(mujoco.mj_geomDistance(model, stub.robot_data, g, floor, 1.0, None) for g in left)

    # A shortfall well past the cap: every term's reported shortfall must sit at the cap,
    # not at the true (larger) distance still owed.
    required = standing + 0.20
    stub.attach_foot_clearance({"l": left}, {"l": np.array([required])}, max_recovery_per_iter=0.01)
    stub._current_frame = 0
    terms = stub._clearance_terms(q)
    assert terms, "a sole far below its required height must still produce terms"
    for _, _, shortfall in terms:
        assert shortfall == pytest.approx(0.01), f"shortfall must be clamped to the recovery cap, got {shortfall:.4f}"

    # A shortfall already under the cap must be reported exactly, unclamped.
    small_required = standing + 0.005
    stub.attach_foot_clearance({"l": left}, {"l": np.array([small_required])}, max_recovery_per_iter=0.01)
    small_terms = stub._clearance_terms(q)
    assert small_terms, "a sole below a small required height must still produce terms"
    for _, _, shortfall in small_terms:
        assert 0 < shortfall < 0.01, f"an already-small shortfall must not be inflated: {shortfall:.4f}"

    for invalid in (0.0, -0.01, np.inf, np.nan):
        with pytest.raises(ValueError, match="max_recovery_per_iter"):
            stub.attach_foot_clearance(
                {"l": left},
                {"l": np.array([small_required])},
                max_recovery_per_iter=invalid,
            )


def test_clearance_finite_edges_ramp_effective_weight(probed_model):
    """A finite target edge must ease the row authority in and out."""
    model, _ = probed_model
    stub = _Stub(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    left = [g for g, n in enumerate(names) if n.startswith(("l_foot_col", "l_bofoot_col"))]
    q = np.array(model.qpos0, dtype=float)
    stub.robot_data.qpos[:] = q
    mujoco.mj_forward(model, stub.robot_data)
    standing = min(mujoco.mj_geomDistance(model, stub.robot_data, g, floor, 1.0, None) for g in left)
    targets = np.array([-np.inf, standing + 0.02, standing + 0.02, -np.inf])
    activation = contact_ramp(np.isfinite(targets), 4)
    stub.attach_foot_clearance(
        {"l": left},
        {"l": targets},
        weight=1200.0,
        activation_by_side={"l": activation},
    )

    stub._current_frame = 0
    assert stub._clearance_terms(q) == []
    stub._current_frame = 1
    first = stub._clearance_terms(q)
    stub._current_frame = 2
    last = stub._clearance_terms(q)
    assert first and last
    assert all(w == pytest.approx(1200.0 * activation[1]) for w, _, _ in first)
    assert all(w == pytest.approx(1200.0 * activation[2]) for w, _, _ in last)
    assert activation[1] == activation[2] < 1.0, "a short interval must taper symmetrically"
    stub._current_frame = 3
    assert stub._clearance_terms(q) == []


def test_smooth_surface_uses_the_same_duration_at_common_source_rates():
    """Zero must be exact; 50 ms must represent the same temporal edge at each FPS.

    `TerrainSpec.height_near` is a hard step at a box's reach-grown footprint edge: traced
    on BEAM01 (`omniretarget-tendon-jump-is-pelvis-pop` memory) to a swing-clearance
    requirement jumping 300+ mm in one frame with no counterpart in the source motion,
    which itself descends smoothly through the same frame. Smoothing spreads the same total
    step over more frames instead of asking the solver to absorb it in one.
    """
    from terra.clearance import _smooth_surface

    step = 0.30
    surface = np.concatenate([np.zeros(20), np.full(20, step)])

    assert np.array_equal(_smooth_surface(surface, 100.0, 0.0), surface), (
        "zero duration must be the raw read, unchanged"
    )

    transition_durations = []
    for fps in (50.0, 100.0, 120.0):
        n = int(fps)
        sampled = np.concatenate([np.zeros(n), np.full(n, step)])
        smoothed = _smooth_surface(sampled, fps, 0.05)
        assert smoothed[0] == pytest.approx(0.0)
        assert smoothed[-1] == pytest.approx(step)
        changed = np.flatnonzero((smoothed > 1e-12) & (smoothed < step - 1e-12))
        transition_durations.append(len(changed) / fps)
        assert np.max(np.abs(np.diff(smoothed))) < step / 2
    assert max(transition_durations) - min(transition_durations) <= 1 / 50.0

    for duration in (-0.01, np.nan, np.inf):
        with pytest.raises(ValueError, match="surface ramp duration"):
            _smooth_surface(surface, 100.0, duration)
    with pytest.raises(ValueError, match="positive fps"):
        _smooth_surface(surface, 0.0, 0.05)


def test_clearance_asks_a_foot_to_clear_a_step_before_it_reaches_it():
    """The lookahead is the whole fix for a foot riding up a riser instead of over it.

    Expressed through contacts, the requirement vanishes exactly where it is needed: a foot
    short of a box touches only its end face, whose normal is horizontal. Measured, that was
    13 frames of forefoot dragging up the beam's end cap while the source cleared the top by
    20-37 mm. The requirement has to be vertical, against the terrain the foot is *about* to
    cross - which is what the reach in `sole_clearance_targets` gives it.
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.clearance import sole_clearance_targets

    names = list(SMPLH_DEMO_JOINTS)
    step_top = 0.10
    step = TerrainSpec(boxes=(BoxSpec(pos=(1.0, 0.0, step_top / 2), size=(0.25, 0.30, step_top / 2)),))

    # A left foot on the floor, at rest long enough to give the offset estimator a datum,
    # then swinging forward to 0.05 m short of the step's near edge with its sole 0.14 m up.
    hold, glide = 120, 60
    joints = np.zeros((hold + glide, len(names), 3))
    joints[:, :, 2] = 1.20
    for j in ("L_Toe", "L_Ankle", "R_Toe", "R_Ankle"):
        joints[:, names.index(j), 0] = 0.0
        joints[:, names.index(j), 2] = 0.0
    for j in ("L_Toe", "L_Ankle"):
        joints[hold:, names.index(j), 0] = np.linspace(0.0, 0.70, glide)
        joints[hold:, names.index(j), 2] = np.linspace(0.0, 0.14, glide)

    targets = sole_clearance_targets(joints, names, 100.0, step, lookahead=0.12, fraction=0.7)

    # 0.05 m short of the step (x = 0.70, near edge at 0.75) the reach has already found it,
    # so the requirement stands on the step's top rather than on the floor.
    assert targets["l"][-1] > step_top, (
        f"a foot short of a step must already be clearing its top, got {targets['l'][-1]:.3f}"
    )
    # ... but never more than the source's own sole reached.
    assert targets["l"][-1] <= 0.14 + 1e-9

    # The right foot never moves and never approaches anything: no requirement at all.
    assert not np.isfinite(targets["r"]).any(), "a still foot must be left unconstrained"

    # The same step, out of reach: the requirement is the floor plus the source's own margin,
    # nowhere near the step's top.
    far = TerrainSpec(boxes=(BoxSpec(pos=(9.0, 0.0, step_top / 2), size=(0.25, 0.30, step_top / 2)),))
    away = sole_clearance_targets(joints, names, 100.0, far, lookahead=0.12, fraction=0.7)
    assert away["l"][-1] == pytest.approx(0.7 * 0.14, abs=1e-9), "a distant step must not enter the requirement"


def _step_up_motion(names, step_top=0.20, hold=120, glide=80, land=40, reach=0.95):
    """A left foot planted on the floor, swinging up onto a step whose near edge is x=0.75.

    Its sole rises with its forward travel, so through the middle of the swing the foot is
    already within reach of the step and still *below* its top - which is what a real ascent
    looks like, and exactly where the fraction-of-source requirement goes inert.
    """
    n = hold + glide + land
    joints = np.zeros((n, len(names), 3))
    joints[:, :, 2] = 1.20
    for j in ("L_Toe", "L_Ankle", "R_Toe", "R_Ankle"):
        joints[:, names.index(j), 0] = 0.0
        joints[:, names.index(j), 2] = 0.0
    for j in ("L_Toe", "L_Ankle"):
        joints[hold : hold + glide, names.index(j), 0] = np.linspace(0.0, reach, glide)
        joints[hold : hold + glide, names.index(j), 2] = np.linspace(0.0, step_top, glide)
        joints[hold + glide :, names.index(j), 0] = reach
        joints[hold + glide :, names.index(j), 2] = step_top
    contact = np.zeros(n, dtype=bool)
    contact[:hold] = True
    contact[hold + glide :] = True
    return joints, {"l": contact, "r": np.ones(n, dtype=bool)}


def test_the_swing_minimum_keeps_the_requirement_alive_where_the_source_grazes_a_step():
    """Without it the requirement is -inf over the frames a foot jams against a riser.

    The source's foot is a toe joint and an ankle joint; the robot's is a 0.25 m body under
    a hard non-penetration constraint. So the robot's forefoot reaches the riser while the
    source's toe is still short of it, and the pull towards the target is along the face
    normal with no tangential part to slide on. Measured on the retargeted subset, that is a
    foot frozen at exactly the -1.0 mm non-penetration tolerance for 17-35 frames while the
    source's own foot travels up to 0.7 m, then a single-frame catch-up: 644 mm of tracking
    lag taken up at 31 m/s on `EKUT/EKUT/234/SS1D104_poses`.
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.clearance import sole_clearance_targets

    names = list(SMPLH_DEMO_JOINTS)
    top = 0.20
    step = TerrainSpec(boxes=(BoxSpec(pos=(1.0, 0.0, top / 2), size=(0.25, 0.30, top / 2)),))
    joints, contact = _step_up_motion(names, step_top=top)
    kw = {"lookahead": 0.12, "fraction": 0.7}

    # Mid-swing, already within reach of the step and still below its top.
    k = 120 + 60
    assert joints[k, names.index("L_Toe"), 0] > 0.75 - 0.12
    assert joints[k, names.index("L_Toe"), 2] < top

    without = sole_clearance_targets(joints, names, 100.0, step, **kw)
    assert not np.isfinite(without["l"][k]), "this only tests anything if the fraction-of-source rule goes inert here"

    with_min = sole_clearance_targets(
        joints,
        names,
        100.0,
        step,
        **kw,
        min_clearance=0.03,
        contact=contact,
        ramp_frames=12,
    )
    # 30 mm above where the source's own sole was at that frame, not above the step: the
    # ask has to stay small, or it is paid for everywhere else (see the test below).
    source_sole = joints[k, names.index("L_Toe"), 2]
    assert with_min["l"][k] == pytest.approx(source_sole + 0.03, abs=2e-3), (
        f"expected the source's own sole ({source_sole:.3f}) plus 30 mm, got {with_min['l'][k]:.3f}"
    )
    # And it is off wherever the foot is planted, or the stance foot lifts off its surface.
    assert not np.isfinite(with_min["l"][0])
    assert not np.isfinite(with_min["r"]).any()


def test_the_swing_minimum_never_asks_a_descending_foot_to_climb_back_up():
    """Why the minimum is measured from the source's sole and not from the surface.

    Stepping *down*, the surface within reach is the tread being stepped off, a whole riser
    above the one being aimed at. A minimum applied to *that* would ask the foot to hold the
    height it is leaving. Measured from the source's own sole it descends with the source,
    and the same rule needs no special case for descents.

    This is not a hypothetical: `surface + min_clearance` was built first and measured over
    the non-flat subset, where it cost 10 of 26 `stairs_up` motions and took median
    body-through-a-box-top from 2.3 mm to 13.3 mm.
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.clearance import sole_clearance_targets

    names = list(SMPLH_DEMO_JOINTS)
    top = 0.20
    tread = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, top / 2), size=(0.30, 0.30, top / 2)),))

    # A left foot standing on the tread, then swinging forward and down onto the floor
    # beyond its far edge at x = 0.30.
    hold, glide = 120, 80
    joints = np.zeros((hold + glide, len(names), 3))
    joints[:, :, 2] = 1.20
    for j in ("L_Toe", "L_Ankle", "R_Toe", "R_Ankle"):
        joints[:, names.index(j), 0] = 0.0
        joints[:, names.index(j), 2] = top
    for j in ("L_Toe", "L_Ankle"):
        joints[hold:, names.index(j), 0] = np.linspace(0.0, 0.45, glide)
        joints[hold:, names.index(j), 2] = np.linspace(top, 0.0, glide)
    contact = np.zeros(len(joints), dtype=bool)
    contact[:hold] = True
    contact[-6:] = True  # the landing, which is what says where the foot is going
    schedule = {"l": contact, "r": np.ones(len(joints), dtype=bool)}
    kw = {"lookahead": 0.12, "fraction": 0.7}

    with_min = sole_clearance_targets(
        joints,
        names,
        100.0,
        tread,
        **kw,
        min_clearance=0.03,
        contact=schedule,
        ramp_frames=12,
    )
    mid = hold + glide // 2
    source_sole = joints[mid, names.index("L_Toe"), 2]
    assert with_min["l"][mid] <= source_sole + 0.03 + 1e-9, (
        f"the minimum lifted a descending foot to {with_min['l'][mid]:.3f} m, more than "
        f"30 mm above the source's own sole at {source_sole:.3f} m"
    )
    # Never the tread being stepped off, which is what the surface-relative form asked for.
    finite = np.where(np.isfinite(with_min["l"]), with_min["l"], -np.inf)
    assert finite[hold + 12 : hold + glide].max() < top, (
        "a descending foot was asked to stay at the height of the tread it is leaving"
    )


def test_swing_target_lift_translates_the_whole_foot_only_beside_the_riser():
    """The route term preserves pitch and vanishes in stance and above the tread."""
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.clearance import swing_foot_target_offsets

    names = list(SMPLH_DEMO_JOINTS)
    top = 0.20
    step = TerrainSpec(boxes=(BoxSpec(pos=(1.0, 0.0, top / 2), size=(0.25, 0.30, top / 2)),))
    joints, contact = _step_up_motion(names, step_top=top)
    offsets = swing_foot_target_offsets(
        joints,
        names,
        100.0,
        step,
        {"L_Toe": contact["l"], "R_Toe": contact["r"]},
        lift=0.03,
        lookahead=0.12,
        ramp_frames=12,
    )

    # Mid-swing and beside the step, both landmarks receive this one offset at the caller;
    # representing it per side makes it impossible for ankle and toe to disagree.
    k = 120 + 65
    assert offsets["l"][k] == pytest.approx(0.03)
    assert offsets["l"][0] == 0.0
    assert not offsets["r"].any()
    active = np.flatnonzero(offsets["l"] > 0.0)
    assert offsets["l"][active[0]] == pytest.approx(0.03 / 12)
    assert offsets["l"][active[-1]] == pytest.approx(0.03 / 12)
    assert np.all(np.diff(offsets["l"][active[:12]]) > 0.0), (
        "the final swing-and-riser mask must ramp in, including a mid-swing riser edge"
    )

    # Once the source sole reaches the tread top, it is no longer beside the riser and the
    # route term gets out of the way even before the contact schedule changes.
    assert offsets["l"][120 + 80 - 1] == 0.0


def test_swing_target_lift_is_inert_on_flat_ground():
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from terra.clearance import swing_foot_target_offsets

    names = list(SMPLH_DEMO_JOINTS)
    joints, contact = _step_up_motion(names)
    offsets = swing_foot_target_offsets(
        joints,
        names,
        100.0,
        None,
        {"L_Toe": contact["l"], "R_Toe": contact["r"]},
        lift=0.03,
    )
    assert not offsets["l"].any()
    assert not offsets["r"].any()


def test_explicit_foot_route_targets_only_contribute_on_finite_frames():
    """The strong route term is absent outside the narrow window selected by OmniRetarget."""
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='foot'><freejoint/><geom type='sphere' size='.02'/></body></worldbody></mujoco>"
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter._build_transform_qdot_to_qvel_fast = lambda: np.eye(retargeter.robot_model.nv, retargeter.robot_model.nq)
    retargeter.attach_foot_route({"foot": np.array([-np.inf, 0.10])}, weight=123.0)

    q = retargeter.robot_model.qpos0.copy()
    retargeter._current_frame = 0
    assert retargeter._foot_route_terms(q) == []
    retargeter._current_frame = 1
    terms = retargeter._foot_route_terms(q)
    assert len(terms) == 1
    weight, row, residual = terms[0]
    assert weight == 123.0
    assert row.shape == (retargeter.robot_model.nq,)
    assert residual == pytest.approx(0.10)


def test_annotated_stance_height_contributes_only_with_contact_authority():
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='foot'><freejoint/><geom type='sphere' size='.02'/></body></worldbody></mujoco>"
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter._build_transform_qdot_to_qvel_fast = lambda: np.eye(retargeter.robot_model.nv, retargeter.robot_model.nq)
    retargeter.attach_foot_stance_height(
        {"foot": np.array([0.10, 0.10])},
        {"foot": np.array([0.0, 0.5])},
        weight=1000.0,
        max_recovery_per_iter=0.004,
    )
    q = retargeter.robot_model.qpos0.copy()
    retargeter._current_frame = 0
    assert retargeter._foot_stance_height_terms(q) == []

    retargeter._current_frame = 1
    weight, row, residual = retargeter._foot_stance_height_terms(q)[0]
    assert weight == pytest.approx(500.0)
    assert row.shape == (retargeter.robot_model.nq,)
    assert residual == pytest.approx(0.004)


def test_seat_contact_uses_exact_signed_geom_distance_and_rest_authority():
    """Chair calibration must lower the glute toward the chair, only while active."""
    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name='terrain_box_seat_0' type='box' pos='0 0 .05' size='.4 .4 .05'/>
            <body name='pelvis' pos='0 0 .20'>
              <freejoint/>
              <geom name='r_pelvis_col' type='sphere' size='.02'/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter._build_transform_qdot_to_qvel_fast = lambda: np.eye(retargeter.robot_model.nv, retargeter.robot_model.nq)
    retargeter.attach_seat_contact(
        ("r_pelvis_col",),
        {"terrain_box_seat_0": np.array([0.0, 0.5, 1.0])},
        weight=2000.0,
        clearance=0.0,
        max_recovery_per_iter=0.01,
    )
    q = retargeter.robot_model.qpos0.copy()

    retargeter._current_frame = 0
    assert retargeter._seat_contact_terms(q) == []
    retargeter._current_frame = 1
    weight, row, residual = retargeter._seat_contact_terms(q)[0]
    assert weight == pytest.approx(1000.0)
    assert row[2] == pytest.approx(1.0)
    assert residual == pytest.approx(-0.01), "the capped target step lowers the hovering glute"


def test_seat_contact_assembly_is_chair_only_and_baseline_safe():
    """No seat geom means no solver mutation; the OmniRetarget profile remains off."""
    import logging

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.assembly import SolveContext, attach_seat_contact_targets
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS

    joints = np.zeros((80, len(SMPLH_DEMO_JOINTS), 3))
    joints[:, SMPLH_DEMO_JOINTS.index("Pelvis")] = [0.5, 0.0, 0.45]
    for name, xy in {
        "L_Toe": (-0.1, 0.1),
        "R_Toe": (-0.1, -0.1),
        "L_Ankle": (0.0, 0.1),
        "R_Ankle": (0.0, -0.1),
    }.items():
        joints[:, SMPLH_DEMO_JOINTS.index(name), :2] = xy

    class Capture:
        def attach_seat_contact(self, *args, **kwargs):
            self.call = (args, kwargs)

    def context(terrain, profile="terra"):
        return SolveContext(
            config={"method_profile": profile},
            logger=logging.getLogger("test_seat_contact_scope"),
            model=None,
            terrain=terrain,
            fps=100.0,
            joints_mapping={},
            human_joints=joints,
            smpl_rotations=np.empty((80, 0, 3, 3)),
            scene={},
        )

    ordinary = TerrainSpec(boxes=(BoxSpec(pos=(0.5, 0.0, 0.1), size=(0.3, 0.3, 0.1), name="terrain_box_platform_0"),))
    capture = Capture()
    ordinary_ctx = context(ordinary)
    attach_seat_contact_targets(ordinary_ctx, capture)
    assert not hasattr(capture, "call")
    assert ordinary_ctx.diagnostics.seat_contact_active is False

    chair = TerrainSpec(boxes=(BoxSpec(pos=(0.5, 0.0, 0.1), size=(0.3, 0.3, 0.1), name="terrain_box_seat_0"),))
    chair_capture = Capture()
    chair_ctx = context(chair)
    attach_seat_contact_targets(chair_ctx, chair_capture)
    assert chair_ctx.diagnostics.seat_contact_active is True
    assert chair_ctx.diagnostics.seat_contact_rest_count == 1
    assert chair_capture.call[0][0] == ("r_pelvis_col", "l_pelvis_col")
    assert np.max(chair_capture.call[0][1]["terrain_box_seat_0"]) == pytest.approx(1.0)

    baseline_capture = Capture()
    baseline_ctx = context(chair, profile="omniretarget")
    attach_seat_contact_targets(baseline_ctx, baseline_capture)
    assert not hasattr(baseline_capture, "call")
    assert baseline_ctx.config.seat_contact_mode == "off"


def test_new_route_row_ramps_target_and_effective_qp_weight():
    """A route row must not apply full authority to the foot's old residual at its edge."""
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='foot'><freejoint/><geom type='sphere' size='.02'/></body></worldbody></mujoco>"
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter._build_transform_qdot_to_qvel_fast = lambda: np.eye(retargeter.robot_model.nv, retargeter.robot_model.nq)
    targets = np.array([-np.inf, 0.025, 0.10])
    activation = np.array([0.0, 0.25, 1.0])
    retargeter.attach_foot_route({"foot": targets}, weight=120.0, activation_by_body={"foot": activation})
    q = retargeter.robot_model.qpos0.copy()

    retargeter._current_frame = 1
    weight, _row, residual = retargeter._foot_route_terms(q)[0]
    assert weight == pytest.approx(30.0)
    assert residual == pytest.approx(0.025), "the requested lift itself must also be ramped"

    retargeter._current_frame = 2
    weight, _row, residual = retargeter._foot_route_terms(q)[0]
    assert weight == pytest.approx(120.0)
    assert residual == pytest.approx(0.10)


def test_route_recovery_cap_bounds_large_residuals_without_inflating_small_ones():
    """Route caps signed residuals without inflating a small residual."""
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='ankle'><freejoint/>"
        "<geom type='sphere' size='.02'/><body name='toe' pos='.1 0 0'>"
        "<geom type='sphere' size='.02'/></body></body></worldbody></mujoco>"
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter._build_transform_qdot_to_qvel_fast = lambda: np.eye(retargeter.robot_model.nv, retargeter.robot_model.nq)
    q = retargeter.robot_model.qpos0.copy()
    q[2] = 0.20
    retargeter._current_frame = 0

    retargeter.attach_foot_route(
        {"ankle": np.array([0.10]), "toe": np.array([0.205])},
        weight=120.0,
        max_recovery_per_iter=0.01,
    )
    terms = retargeter._foot_route_terms(q)
    assert len(terms) == 2
    assert terms[0][2] == pytest.approx(-0.01)
    assert terms[1][2] == pytest.approx(0.005)


def test_route_weight_activation_does_not_inflate_or_shrink_the_residual_cap():
    """Weight authority and the bounded residual have independent, explicit roles."""
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='foot'><freejoint/><geom type='sphere' size='.02'/></body></worldbody></mujoco>"
    )
    retargeter.robot_data = mujoco.MjData(retargeter.robot_model)
    retargeter.q_a_indices = np.arange(retargeter.robot_model.nq)
    retargeter.nq_a = retargeter.robot_model.nq
    retargeter._build_transform_qdot_to_qvel_fast = lambda: np.eye(retargeter.robot_model.nv, retargeter.robot_model.nq)
    retargeter.attach_foot_route(
        {"foot": np.array([0.10])},
        weight=120.0,
        activation_by_body={"foot": np.array([0.25])},
        max_recovery_per_iter=0.01,
    )
    q = retargeter.robot_model.qpos0.copy()
    q[2] = 0.20
    retargeter._current_frame = 0

    weight, _row, residual = retargeter._foot_route_terms(q)[0]
    assert weight == pytest.approx(30.0)
    assert residual == pytest.approx(-0.01)


def test_route_recovery_cap_validation_is_explicit():
    import mujoco

    from terra.retargeter import TerraRetargeter

    retargeter = object.__new__(TerraRetargeter)
    retargeter._initialize_terra_state()
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='foot'><freejoint/><geom type='sphere' size='.02'/></body></worldbody></mujoco>"
    )
    for cap in [0.0, -0.1, np.inf, np.nan]:
        with pytest.raises(ValueError, match="max_recovery_per_iter"):
            retargeter.attach_foot_route({"foot": np.ones(2)}, 100.0, max_recovery_per_iter=cap)


def _tendon_jump_probe():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom type='sphere' size='.01'/>"
        "<site name='origin' pos='0 0 0'/>"
        "<body pos='1 0 0'><joint name='hinge' axis='0 0 1'/>"
        "<geom type='sphere' size='.01'/><site name='moving' pos='1 0 0'/>"
        "</body></body></worldbody>"
        "<tendon><spatial name='probe'><site site='origin'/><site site='moving'/>"
        "</spatial></tendon></mujoco>"
    )
    qpos = np.repeat(model.qpos0[None], 5, axis=0)
    qpos[2, model.jnt_qposadr[1]] = 1.5
    return model, qpos


def _materialized_tendon_jump_probe():
    """Build a tendon jump whose tracked child body moves with the driving hinge."""
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='Pelvis'><freejoint/>"
        "<geom type='sphere' size='.01'/><site name='origin'/>"
        "<body name='L_Hip' pos='1 0 0'>"
        "<joint name='hinge' axis='0 0 1' limited='true' range='-120 120'/>"
        "<geom type='sphere' size='.01'/>"
        "<body name='L_Knee' pos='1 0 0'><geom type='sphere' size='.01'/>"
        "<site name='moving' pos='1 0 0'/></body></body></body></worldbody>"
        "<tendon><spatial name='probe'><site site='origin'/><site site='moving'/>"
        "</spatial></tendon></mujoco>"
    )
    qpos = np.repeat(model.qpos0[None], 5, axis=0)
    qpos[2, model.jnt_qposadr[1]] = 1.5
    names = ("Pelvis", "L_Knee")
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
    joint_indices = [SMPLH_DEMO_JOINTS.index(name) for name in names]
    targets = np.zeros((len(qpos), len(SMPLH_DEMO_JOINTS), 3))
    data = mujoco.MjData(model)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        targets[frame, joint_indices] = data.xpos[body_ids]
    return model, qpos, targets, dict(zip(names, names, strict=True))


def test_bounded_tendon_repair_removes_one_isolated_pose_and_honours_validator():
    from terra.tendon_repair import (
        adaptive_tendon_events,
        repair_tendon_discontinuities,
        tendon_lengths,
    )

    model, qpos = _tendon_jump_probe()
    assert adaptive_tendon_events(model, tendon_lengths(model, qpos), threshold=0.05)

    repaired, frames, remaining = repair_tendon_discontinuities(model, qpos, threshold=0.05, max_changed_frames=1)
    assert frames == [2]
    assert not remaining
    assert repaired[2] == pytest.approx(qpos[1])

    rejected, frames, remaining = repair_tendon_discontinuities(
        model,
        qpos,
        validator=lambda _candidate, _frames: False,
        threshold=0.05,
        max_changed_frames=1,
    )
    assert frames == []
    assert remaining
    assert rejected == pytest.approx(qpos)


def test_bounded_tendon_repair_preserves_a_free_root_exactly():
    """A global rigid transform cannot fix an internal length and must never be smoothed."""
    from terra.tendon_repair import repair_tendon_discontinuities

    model, qpos = _tendon_jump_probe()
    qpos[:, :3] = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.01, 0.02, 0.03],
            [0.08, 0.04, 0.10],
            [0.09, 0.06, 0.12],
            [0.11, 0.08, 0.15],
        ]
    )
    qpos[:, 3:7] = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.999, 0.0, 0.045, 0.0],
            [0.990, 0.0, 0.141, 0.0],
            [0.980, 0.0, 0.199, 0.0],
            [0.970, 0.0, 0.243, 0.0],
        ]
    )
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    root = qpos[:, :7].copy()

    repaired, frames, remaining = repair_tendon_discontinuities(model, qpos, threshold=0.05, max_changed_frames=1)

    assert frames == [2]
    assert not remaining
    assert np.array_equal(repaired[:, :7], root)


def test_tendon_repair_edits_only_its_jacobian_chain_and_projects_couplers():
    """A shoulder-like event must not interpolate an unrelated leg-like joint."""
    import mujoco

    from terra.tendon_repair import (
        _coupled_coordinate_closure,
        _interpolate_span,
        _project_couplers,
        _tendon_qpos_dependencies,
    )

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom type='sphere' size='.01'/>"
        "<site name='origin'/>"
        "<body pos='1 0 0'><joint name='driver' axis='0 0 1'/>"
        "<geom type='sphere' size='.01'/><site name='moving' pos='1 0 0'/></body>"
        "<body pos='0 1 0'><joint name='coupled' axis='0 0 1'/>"
        "<geom type='sphere' size='.01'/></body>"
        "<body pos='0 -1 0'><joint name='unrelated' axis='0 0 1'/>"
        "<geom type='sphere' size='.01'/></body>"
        "</body></worldbody>"
        "<equality><joint joint1='coupled' joint2='driver' "
        "polycoef='0 2 0 0 0'/></equality>"
        "<tendon><spatial name='probe'><site site='origin'/><site site='moving'/>"
        "</spatial></tendon></mujoco>"
    )
    data = mujoco.MjData(model)
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    driver = int(model.jnt_qposadr[1])
    coupled = int(model.jnt_qposadr[2])
    unrelated = int(model.jnt_qposadr[3])
    qpos[2, driver] = 1.0
    qpos[1, unrelated] = 0.73

    dependencies = _tendon_qpos_dependencies(model, data, qpos[2], 0)
    coordinates, rows = _coupled_coordinate_closure(model, dependencies)
    candidate = _interpolate_span(qpos, 0, 2, coordinates=coordinates)
    _project_couplers(candidate, range(1, 2), rows)

    assert set(dependencies) == {driver}
    assert set(coordinates) == {driver, coupled}
    assert candidate[1, unrelated] == qpos[1, unrelated]
    assert candidate[1, coupled] == pytest.approx(2.0 * candidate[1, driver])


def test_bounded_tendon_repair_validation_is_explicit():
    from terra.tendon_repair import repair_tendon_discontinuities

    model, qpos = _tendon_jump_probe()
    with pytest.raises(ValueError, match="threshold"):
        repair_tendon_discontinuities(model, qpos, threshold=0.0)
    with pytest.raises(ValueError, match="max_changed_frames"):
        repair_tendon_discontinuities(model, qpos, max_changed_frames=-1)
    with pytest.raises(ValueError, match="max_local_changed_frames"):
        repair_tendon_discontinuities(model, qpos, max_local_changed_frames=-1)


def test_materialized_repair_targets_use_the_exact_endpoint_aligned_grid():
    """Tracking targets must share the trajectory materializer's output timestamps."""
    from terra.postprocess import resample_frame_values

    source = np.arange(12, dtype=float).reshape(6, 2)
    output = resample_frame_values(source, 5)

    assert output.shape == (5, 2)
    assert output[0] == pytest.approx(source[0])
    assert output[-1] == pytest.approx(source[-1])
    assert resample_frame_values(source, len(source)) == pytest.approx(source)
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="output_frames"):
            resample_frame_values(source, invalid)


@pytest.mark.parametrize(("tracking_budget", "accepted"), [(10.0, True), (0.0, False)])
def test_materialized_tendon_repair_enforces_tracking_quality_gate(caplog, tracking_budget, accepted):
    """A storage-precision tendon repair is accepted only within its tracking budget."""
    import logging
    from types import SimpleNamespace

    from terra.postprocess import repair_materialized_tendon_transitions

    model, qpos, targets, joints_mapping = _materialized_tendon_jump_probe()
    ctx = SimpleNamespace(
        on_terrain=True,
        config=SolverConfig.from_mapping(
            {
                "posthoc_tracking_mean_regression": tracking_budget,
                "posthoc_tracking_max_regression": tracking_budget,
                "posthoc_tendon_max_frames": 1,
                "posthoc_tendon_max_local_frames": 1,
            }
        ).for_scene(on_terrain=True),
        logger=logging.getLogger("test_materialized_tendon_repair"),
        model=model,
        joints_mapping=joints_mapping,
    )

    with caplog.at_level(logging.INFO):
        repaired = repair_materialized_tendon_transitions(ctx, qpos, targets, "off")

    if accepted:
        assert np.flatnonzero(np.any(repaired != qpos, axis=1)).tolist() == [2]
        assert repaired[2, model.jnt_qposadr[1]] == pytest.approx(0.0)
        assert "interpolated frame(s) [2]" in caplog.text
    else:
        np.testing.assert_array_equal(repaired, qpos)
        assert "no bounded quality-safe interpolation was found" in caplog.text


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"posthoc_pen_threshold": np.nan}, "posthoc_pen_threshold"),
        ({"posthoc_tracking_mean_regression": -0.01}, "posthoc_tracking_mean_regression"),
        ({"posthoc_tendon_threshold": 0.0}, "posthoc_tendon_threshold"),
        ({"posthoc_tendon_max_frames": 1.5}, "posthoc_tendon_max_frames"),
        ({"posthoc_tendon_max_local_frames": True}, "posthoc_tendon_max_local_frames"),
    ],
)
def test_materialized_tendon_repair_policy_rejects_invalid_limits(config, message):
    from terra.postprocess import _materialized_tendon_repair_policy

    with pytest.raises(ValueError, match=message):
        _materialized_tendon_repair_policy(SolverConfig.from_mapping(config))


def test_materialized_tendon_repair_validates_input_shapes_and_mapping():
    from terra.postprocess import _materialized_landmark_mapping, _materialized_repair_inputs

    model, qpos, targets, _mapping = _materialized_tendon_jump_probe()
    with pytest.raises(ValueError, match=r"qpos.*shape"):
        _materialized_repair_inputs(model, qpos[:, :-1], targets)
    with pytest.raises(ValueError, match=r"targets.*shape"):
        _materialized_repair_inputs(model, qpos, targets[..., 0])
    with pytest.raises(ValueError, match="does not exist"):
        _materialized_landmark_mapping(model, {"Pelvis": "missing"}, targets.shape[1])


def test_materialized_tendon_joint_limit_gate_handles_ball_joint_angles():
    from terra.postprocess import _joint_limits_satisfied

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><joint type='ball' limited='true' range='0 30'/>"
        "<geom type='sphere' size='.1'/></body></worldbody></mujoco>"
    )
    qpos = model.qpos0.copy()
    inside = np.radians(20.0)
    qpos[:4] = [np.cos(inside / 2.0), np.sin(inside / 2.0), 0.0, 0.0]
    assert _joint_limits_satisfied(model, qpos)

    outside = np.radians(40.0)
    qpos[:4] = [np.cos(outside / 2.0), np.sin(outside / 2.0), 0.0, 0.0]
    assert not _joint_limits_satisfied(model, qpos)


def test_flat_landmark_analysis_keeps_the_historical_extra_target_tail():
    """A source-rate target may be one sample longer after velocity differencing."""
    from types import SimpleNamespace

    import mujoco

    from terra.postprocess import landmark_error

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='Pelvis'><freejoint/>"
        "<geom type='sphere' size='.01'/></body></worldbody></mujoco>"
    )
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    ctx = SimpleNamespace(
        model=model,
        joints_mapping={"Pelvis": "Pelvis"},
        human_joints=np.zeros((4, 1, 3)),
        logger=SimpleNamespace(info=lambda _message: None),
    )

    error = landmark_error(ctx, mujoco.MjData(model), qpos, ["Pelvis"])

    assert error.shape == (3, 1)
    assert not error.any()
    with pytest.raises(ValueError, match="must match qpos"):
        landmark_error(
            ctx,
            mujoco.MjData(model),
            qpos,
            ["Pelvis"],
            target_joints=np.zeros((4, 1, 3)),
        )


def test_bounded_tendon_repair_does_not_hide_a_secondary_event_in_the_ema(monkeypatch):
    """An earlier repair must not erase a still-large later transition from bookkeeping."""
    from types import SimpleNamespace

    import terra.tendon_repair as repair

    # Treat the last qpos coordinate as one synthetic tendon length. The first repair can
    # raise the adaptive detector's EMA enough to hide the second 11% physical step even
    # though it has not changed; the repair must retain that transition and continue.
    monkeypatch.setattr(repair, "tendon_lengths", lambda _model, poses: np.asarray(poses[:, 7:8], dtype=float))
    model = SimpleNamespace(ntendon=1, tendon_length0=np.ones(1))
    values = np.array(
        [
            -0.0064740161,
            -0.0079681357,
            -0.0069840754,
            0.1088240128,
            0.1135959335,
            0.1100222535,
            0.1110858493,
            0.1079406327,
            -0.0054772150,
            -0.0055421687,
            -0.0066589204,
            -0.0118134753,
        ]
    )
    qpos = np.zeros((len(values), 8))
    qpos[:, 3] = 1.0
    qpos[:, 7] = values

    repaired, frames, remaining = repair.repair_tendon_discontinuities(
        model, qpos, threshold=0.05, max_changed_frames=6
    )

    assert frames
    assert not remaining
    raw = np.abs(np.diff(repair.tendon_lengths(model, repaired)[:, 0]))
    assert raw[[2, 7]].max() <= 0.05, "both originally flagged steps must be physically fixed"


def _probe_over_floor():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='floor' type='plane' size='1 1 .1'/>"
        "<body name='probe' pos='0 0 .02'><freejoint/>"
        "<geom name='probe_geom' type='sphere' size='.02'/></body></worldbody></mujoco>"
    )
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    return model, floor


def test_posthoc_repair_interpolates_only_an_isolated_collision_that_it_resolves():
    from terra.collisions import repair_isolated_terrain_penetrations

    model, floor = _probe_over_floor()
    qpos = np.repeat(model.qpos0[None], 5, axis=0)
    qpos[2, 2] = -0.03  # one bad frame between exactly supported poses
    repaired, frames = repair_isolated_terrain_penetrations(model, qpos, [floor], threshold=0.005, max_run=1)
    assert frames == [2]
    assert repaired[2, 2] == pytest.approx(qpos[1, 2])
    assert np.array_equal(repaired[[0, 1, 3, 4]], qpos[[0, 1, 3, 4]])


def test_posthoc_repair_leaves_a_systematic_collision_run_visible():
    from terra.collisions import repair_isolated_terrain_penetrations

    model, floor = _probe_over_floor()
    qpos = np.repeat(model.qpos0[None], 5, axis=0)
    qpos[2:4, 2] = -0.03
    repaired, frames = repair_isolated_terrain_penetrations(model, qpos, [floor], threshold=0.005, max_run=1)
    assert frames == []
    assert np.array_equal(repaired, qpos)


def test_posthoc_repair_accepts_a_three_frame_collision_run():
    from terra.collisions import repair_isolated_terrain_penetrations

    model, floor = _probe_over_floor()
    qpos = np.repeat(model.qpos0[None], 7, axis=0)
    qpos[2:5, 2] = -0.03
    repaired, frames = repair_isolated_terrain_penetrations(model, qpos, [floor], threshold=0.005, max_run=3)
    assert frames == [2, 3, 4]
    assert np.allclose(repaired[2:5, 2], qpos[1, 2])


def test_posthoc_repair_phase_shifts_an_interpolation_only_collision():
    """A rotating foot can cross a stair nose between two collision-free poses."""
    from terra.collisions import repair_isolated_terrain_penetrations

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<geom name='obstacle' type='sphere' pos='.7 0 0' size='.1'/>"
        "<body name='root'><freejoint/><inertial pos='0 0 0' mass='1' diaginertia='.01 .01 .01'/>"
        "<body name='arm'>"
        "<joint name='sweep' type='hinge' axis='0 0 1'/>"
        "<geom name='sweeper' type='capsule' fromto='0 0 0 1 0 0' size='.05'/>"
        "</body></body></worldbody></mujoco>"
    )
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
    qpos = np.repeat(model.qpos0[None], 5, axis=0)
    qpos[1, 7], qpos[3, 7] = -0.4, 0.4
    qpos[2, 7] = 0.0  # the nominal midpoint intersects the obstacle

    repaired, frames = repair_isolated_terrain_penetrations(model, qpos, [obstacle], threshold=0.005, max_run=1)

    assert frames == [2]
    assert repaired[2, 7] != pytest.approx(0.0), "the unsafe temporal midpoint was retained"
    assert abs(repaired[2, 7]) < 0.4, "repair must remain inside the neighboring poses"


def test_posthoc_repair_checks_isolated_self_collision_too():
    from terra.collisions import repair_isolated_terrain_penetrations

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<body name='left' pos='-.05 0 0'><freejoint/>"
        "<geom name='left_geom' type='sphere' size='.02'/></body>"
        "<body name='right' pos='.05 0 0'><freejoint/>"
        "<geom name='right_geom' type='sphere' size='.02'/></body>"
        "</worldbody></mujoco>"
    )
    qpos = np.repeat(model.qpos0[None], 3, axis=0)
    qpos[1, :3] = qpos[1, 7:10]  # put the two spheres on top of one another
    repaired, frames = repair_isolated_terrain_penetrations(
        model,
        qpos,
        [],
        threshold=0.005,
        max_run=1,
        self_collision_body_pairs=(("left", "right"),),
    )
    assert frames == [1]
    assert np.allclose(repaired[1], qpos[0])


def test_posthoc_repair_rejects_a_candidate_that_deepens_secondary_collision():
    """Fixing one collision class must not introduce a violation of the other class."""
    from terra.collisions import _repair_collision_runs

    qpos = np.array([[-1.0], [10.0], [1.0]])
    original = qpos.copy()
    repaired = set()

    def primary(pose):
        return float(pose[0] > 5.0)

    def secondary(pose):
        return float(abs(pose[0]) <= 2.0)

    _repair_collision_runs(
        qpos,
        repaired,
        primary,
        secondary,
        threshold=0.1,
        max_run=1,
        quaternion_slices=(),
    )

    assert repaired == set()
    np.testing.assert_array_equal(qpos, original)


def test_collision_pose_interpolation_handles_every_quaternion_joint():
    """Equivalent signs on multiple ball joints must not create zero quaternions."""
    from terra.collisions import _interpolate_pose, _quaternion_qpos_slices

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        "<body><joint type='ball'/><geom type='sphere' size='.02' mass='1'/>"
        "<body pos='.1 0 0'><joint type='ball'/><geom type='sphere' size='.02' mass='1'/></body>"
        "</body></worldbody></mujoco>"
    )
    before = model.qpos0.copy()
    after = before.copy()
    quaternion_slices = _quaternion_qpos_slices(model)
    for quaternion_slice in quaternion_slices:
        after[quaternion_slice] *= -1.0

    midpoint = _interpolate_pose(before, after, 0.5, quaternion_slices)

    assert quaternion_slices == (slice(0, 4), slice(4, 8))
    for quaternion_slice in quaternion_slices:
        np.testing.assert_allclose(midpoint[quaternion_slice], before[quaternion_slice])
        assert np.linalg.norm(midpoint[quaternion_slice]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("threshold", "max_run", "message"),
    [
        (np.nan, 1, "threshold"),
        (-0.01, 1, "threshold"),
        (0.005, -1, "max_run"),
        (0.005, 1.5, "max_run"),
        (0.005, True, "max_run"),
    ],
)
def test_posthoc_repair_rejects_invalid_numerical_limits(threshold, max_run, message):
    from terra.collisions import repair_isolated_terrain_penetrations

    model, floor = _probe_over_floor()
    qpos = np.repeat(model.qpos0[None], 3, axis=0)

    with pytest.raises(ValueError, match=message):
        repair_isolated_terrain_penetrations(
            model,
            qpos,
            [floor],
            threshold=threshold,
            max_run=max_run,
        )


def test_posthoc_repair_rejects_malformed_trajectory_and_geom_ids():
    from terra.collisions import repair_isolated_terrain_penetrations

    model, _floor = _probe_over_floor()
    with pytest.raises(ValueError, match=r"qpos.*shape"):
        repair_isolated_terrain_penetrations(model, np.zeros((3, model.nq - 1)), [])
    with pytest.raises(ValueError, match="environment geom IDs"):
        repair_isolated_terrain_penetrations(model, np.repeat(model.qpos0[None], 3, axis=0), [model.ngeom])


# The lookahead above is why the margin the constraint enforces and the margin measured
# from the source have to be read off the same surface. Measured with `height_at` and
# enforced with `height_near`, a foot standing on a tread is asked to reach the top of the
# tread *above* it: over the 60 stair motions of the non-flat subset that put 167 of 587
# footfalls exactly one riser too high, on ascents and descents alike.


def test_source_clearance_is_read_off_the_surface_the_constraint_enforces_against():
    """A stance foot on a staircase must ask for no clearance at all.

    The two halves are only comparable at one `lookahead`. At 0 the source reads a clean
    stance - sole on its own tread, clearance zero - and the constraint then adds that zero
    to the tread *above*, which is within a foot length. The margin has to go negative in
    stance for the constraint to be inert there, which is what the shared reach gives.
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.clearance import source_sole_clearance
    from terra.defaults import DEFAULT_CLEARANCE_LOOKAHEAD

    names = list(SMPLH_DEMO_JOINTS)
    riser, run = 0.20, 0.30
    stairs = TerrainSpec(
        boxes=tuple(
            BoxSpec(pos=((i + 0.5) * run, 0.0, (i + 1) * riser / 2), size=(run / 2, 0.5, (i + 1) * riser / 2))
            for i in range(4)
        )
    )

    # A stance on the floor and then one in the middle of tread 1 (top 0.40), with the toe
    # joints on the surface and the ankles a realistic 0.09 m inside the foot. The floor
    # stance is what gives `joint_surface_offsets` a datum: measured on the tread alone,
    # each joint's offset would absorb the whole riser. Each phase runs well past the
    # detector's 0.3 s local-minimum window, or neither reads as a stance at all.
    fps, hold, step = 100.0, 100, 10
    # Clear of the flight, then near the front of tread 1 - where a foot that fills its
    # tread actually stands, and where the tread above is within a foot length.
    stance_x = {0: -0.40, 1: 1.85 * run}
    stance_z = {0: 0.0, 1: 2 * riser}
    joints = np.zeros((2 * hold + step, len(names), 3))
    for phase in (0, 1):
        sl = slice(0, hold) if phase == 0 else slice(hold + step, None)
        joints[sl, :, 0] = stance_x[phase]
        joints[sl, :, 2] = stance_z[phase] + 1.20
        for toe in ("L_Toe", "R_Toe"):
            joints[sl, names.index(toe), 2] = stance_z[phase]
        for ankle in ("L_Ankle", "R_Ankle"):
            joints[sl, names.index(ankle), 2] = stance_z[phase] + 0.09
    # The transition itself must be fast enough to fail the speed gate, or the two phases
    # merge into one run and neither is a stance.
    joints[hold : hold + step] = joints[hold - 1]
    joints[hold : hold + step, :, 0] = np.linspace(stance_x[0], stance_x[1], step)[:, None]

    on_tread = slice(hold + step, None)
    at_foot = source_sole_clearance(joints, names, fps, stairs, lookahead=0.0)
    shared = source_sole_clearance(joints, names, fps, stairs, lookahead=DEFAULT_CLEARANCE_LOOKAHEAD)
    for side in ("l", "r"):
        assert at_foot[side][on_tread].max() == pytest.approx(0.0, abs=1e-9), (
            "measured under the foot, a stance reads as zero clearance - which the "
            "constraint would then require above the tread above"
        )
        # Exactly one riser of slack, which is the error this removes from every
        # intermediate footfall of a flight.
        assert shared[side][on_tread].max() == pytest.approx(-riser, abs=1e-9), (
            f"{side}: a stance foot must ask for no clearance over the step ahead of it, "
            f"got {shared[side][on_tread].max() * 1000:.0f} mm"
        )


def test_clearance_never_asks_for_more_height_than_the_source_reached():
    """The invariant that makes the requirement safe to enforce at the robot's own geoms.

    The requirement used to be a *clearance* the solver added to a surface it read at the
    robot's sole geoms. A sole straddling a tread nose is within a foot length of the step
    above while the source's toe joint is not, so a margin of a tenth of a millimetre became
    a demand to stand on the next tread: on `KIT/167/upstairs01_poses` that lifted the whole
    body 120-270 mm off the floor for 140 frames, and the guard on the margin could not
    catch it because the margin was positive, just tiny.

    Resolving the height on the source side makes that unrepresentable - whatever the robot
    is doing, it is never asked to go above where the source's own sole was.
    """
    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS

    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.clearance import _sole_clearance_and_surface, sole_clearance_targets

    names = list(SMPLH_DEMO_JOINTS)
    riser, run = 0.20, 0.30
    stairs = TerrainSpec(
        boxes=tuple(
            BoxSpec(pos=((i + 0.5) * run, 0.0, (i + 1) * riser / 2), size=(run / 2, 0.5, (i + 1) * riser / 2))
            for i in range(4)
        )
    )

    # A foot on the floor at the foot of the flight, then climbing it: the case where the
    # robot's sole and the source's toe joint sit on opposite sides of a tread nose.
    rng = np.random.default_rng(0)
    n = 400
    joints = np.zeros((n, len(names), 3))
    joints[:, :, 2] = 1.20
    x = np.linspace(-0.5, 1.3, n)
    for j, phase in (("L_Toe", 0.0), ("L_Ankle", 0.0), ("R_Toe", np.pi), ("R_Ankle", np.pi)):
        joints[:, names.index(j), 0] = x
        lift = 0.25 * np.clip(np.sin(np.linspace(0, 6 * np.pi, n) + phase), 0, None)
        joints[:, names.index(j), 2] = stairs.height_at(x, np.zeros(n)) + lift
    joints[:100, [names.index(j) for j in ("L_Toe", "L_Ankle", "R_Toe", "R_Ankle")], 2] = 0.0
    joints[:100, :, 0] += rng.normal(scale=1e-4, size=(100, len(names)))  # break exact ties

    targets = sole_clearance_targets(joints, names, 100.0, stairs, lookahead=0.12, fraction=0.7, cap=0.05)
    pairs = _sole_clearance_and_surface(joints, names, 100.0, stairs, lookahead=0.12)
    for side, required in targets.items():
        clearance, surface = pairs[side]
        source_sole = surface + clearance
        active = np.isfinite(required)
        assert (required[active] <= source_sole[active] + 1e-9).all(), (
            f"{side}: the requirement rose above the source's own sole by up to "
            f"{(required[active] - source_sole[active]).max() * 1000:.1f} mm"
        )
        # And it is inert wherever the source demonstrated nothing, rather than collapsing
        # onto whatever surface happens to be in reach.
        assert not np.isfinite(required[clearance <= 0]).any()


@requires_omni
def test_mtp_moves_no_tracked_landmark_and_so_must_be_damped(probed_model):
    """The reason the toes flap: `mtp_angle` is invisible to the whole landmark cost.

    The `toes_*` body origin *is* the mtp joint centre, so rotating it moves no tracked body
    at all while moving the sole geoms several mm. Nothing in the objective resolves it until
    the swing-clearance cost pushes on the forefoot geoms, and then dorsiflexing the toes is
    the cheapest way to satisfy it. If a future landmark set ever makes mtp observable this
    test fails, which is the signal that the damping is no longer load-bearing.
    """
    from terra.constants import MTP_JOINTS, SMPLH_TO_MYOFULLBODY
    from terra.robot import mtp_qpos_indices

    model, _ = probed_model
    data = mujoco.MjData(model)
    tracked = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in SMPLH_TO_MYOFULLBODY.values()]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]

    assert mtp_qpos_indices(model).size == len(MTP_JOINTS), "not every toe joint resolved"

    for joint in MTP_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        adr = int(model.jnt_qposadr[jid])
        side = joint[-1]  # its own foot's geoms, not the other one's
        sole = [g for g, n in enumerate(names) if n.startswith((f"{side}_foot_col", f"{side}_bofoot_col"))]
        assert sole, f"no sole geoms for {joint}"

        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        before_bodies = data.xpos[tracked].copy()
        before_soles = data.geom_xpos[sole].copy()

        data.qpos[adr] += 0.2
        mujoco.mj_forward(model, data)

        assert np.abs(data.xpos[tracked] - before_bodies).max() < 1e-9, (
            f"{joint} now moves a tracked landmark; the damping was justified by it not doing so"
        )
        assert np.abs(data.geom_xpos[sole] - before_soles).max() > 1e-3, (
            f"{joint} must still move its sole geoms, or the clearance cost could not drive it"
        )


@requires_omni
def test_clearance_needs_a_floor_geom_to_measure_height_from(probed_model):
    """Height comes from the distance to the floor plane; without one it would silently be 0."""
    model, _ = probed_model
    stub = _Stub(model)
    stub.robot_model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'><freejoint/>"
        "<geom name='g' type='sphere' size='0.1'/></body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="floor"):
        stub.attach_foot_clearance({"l": [0]}, {"l": np.array([0.02])})


def test_contact_ramp_is_zero_outside_contact_and_tapers_inside():
    contact = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    ramp = contact_ramp(contact, 4)

    assert (ramp[~contact] == 0).all(), "a ramp outside contact would constrain a swinging foot"
    assert 0.0 <= ramp.min() and ramp.max() <= 1.0
    assert ramp[2] < ramp[5], "must ease in after touchdown"
    assert ramp[11] < ramp[8], "must ease out before lift-off"
    assert ramp[5:9].max() == pytest.approx(1.0), "must reach full strength in mid-stance"


def test_contact_ramp_short_contact_stays_symmetric():
    """A contact shorter than 2x the ramp must taper both ends, not saturate on one."""
    ramp = contact_ramp(np.array([0, 1, 1, 1, 0], dtype=bool), 4)
    assert ramp[1] == pytest.approx(ramp[3]), "short contact must be symmetric"
    assert ramp.max() < 1.0


def test_contact_ramp_zero_frames_is_binary():
    contact = np.array([0, 1, 1, 0], dtype=bool)
    assert (contact_ramp(contact, 0) == contact.astype(float)).all()


def test_contact_ramp_preserves_censored_recording_boundaries():
    """A clip boundary is not evidence of a touchdown or liftoff transition."""
    contact = np.array([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=bool)
    ramp = contact_ramp(contact, 4, preserve_clipped_boundaries=True)

    assert ramp[0] == pytest.approx(1.0)
    assert ramp[5] < 1.0, "the observed first liftoff must still ease out"
    assert ramp[7] < 1.0, "the observed last touchdown must still ease in"
    assert ramp[-1] == pytest.approx(1.0)


def test_contact_ramp_default_keeps_symmetric_legacy_semantics():
    contact = np.ones(5, dtype=bool)
    ramp = contact_ramp(contact, 4)
    assert ramp[0] == pytest.approx(ramp[-1])
    assert ramp[0] < ramp[2]


def test_contact_ramp_can_hold_support_until_observed_release():
    contact = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    ramp = contact_ramp(contact, 4, release_ramp_frames=1)

    assert ramp[2] < 1.0, "touchdown must retain its gradual engagement"
    assert ramp[5] == pytest.approx(1.0)
    assert ramp[11] == pytest.approx(1.0), "support must persist through annotated stance"
    assert ramp[12] == pytest.approx(0.0)


def test_contact_ramp_rejects_negative_release_duration():
    with pytest.raises(ValueError, match="release_ramp_frames"):
        contact_ramp(np.ones(5, dtype=bool), 4, release_ramp_frames=-1)


def test_source_foot_contact_requires_slow_and_low():
    """Height matters as much as speed: OmniRetarget's speed-only test is what over-triggers."""
    demo = ["Pelvis", "L_Toe", "R_Toe"]
    fps = 100.0
    t = np.arange(60)
    joints = np.zeros((60, 3, 3))
    joints[:, 0, 2] = 1.0
    # Left toe: stationary but held high - lifted, not planted.
    joints[:, 1] = [0.0, 0.0, 0.50]
    # Right toe: on the floor, moving fast for the first half then still.
    joints[:, 2, 2] = 0.0
    joints[:, 2, 0] = np.where(t < 30, t * 0.02, 30 * 0.02)

    contact = source_foot_contact(joints, demo, ["L_Toe", "R_Toe"], fps)

    assert not contact["L_Toe"].any(), "a high, still foot must not count as contact"
    assert not contact["R_Toe"][:25].any(), "a fast foot must not count as contact"
    assert contact["R_Toe"][40:].all(), "a low, still foot must count as contact"


def test_source_foot_contact_sees_a_footfall_on_raised_terrain():
    """ "Low" against the motion's global minimum makes every elevated stance read as swing.

    The failure is silent - no error, just a foot that is never anchored - and it costs the
    whole point of the constraint on exactly the motions terrain exists for. Measured on
    `go_over_beam08`, stances on the beam slipped 87-111 mm against 3-34 mm on the floor.
    """
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec

    demo = ["Pelvis", "L_Toe", "R_Toe"]
    joints = np.zeros((60, 3, 3))
    joints[:, 0, 2] = 1.0
    # Left toe planted on a 0.10 m beam; right toe planted on the floor beside it.
    joints[:, 1] = [0.0, 0.0, 0.102]
    joints[:, 2] = [0.0, 1.0, 0.0]

    beam = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.05), size=(1.0, 0.2, 0.05)),))

    flat = source_foot_contact(joints, demo, ["L_Toe", "R_Toe"], 100.0)
    assert not flat["L_Toe"].any(), "precondition: the flat datum cannot see this footfall"

    on_terrain = source_foot_contact(joints, demo, ["L_Toe", "R_Toe"], 100.0, terrain=beam)
    assert on_terrain["L_Toe"].all(), "a foot planted on the beam is in contact"
    assert on_terrain["R_Toe"].all(), "a foot on the floor beside the beam is still in contact"


@requires_omni
def test_sole_offset_lifts_the_foot_targets_by_the_geometry_mismatch(probed_model):
    """The correction is robot-minus-human, and it is what a swing clearance is made of.

    Both sides of it are measured, so a sign error or a swapped subtraction produces a
    correction of the right magnitude pointing the wrong way - which on a beam is the
    difference between clearing the surface and being driven into it.
    """
    from terra.constants import SMPLH_TO_MYOFULLBODY
    from terra.contacts import foot_sole_offsets, robot_sole_offsets

    model, _ = probed_model
    robot = robot_sole_offsets(model, SMPLH_TO_MYOFULLBODY)
    assert set(robot) == {"L_Toe", "R_Toe", "L_Ankle", "R_Ankle"}
    assert all(v > 0 for v in robot.values()), "every tracked foot body sits above the sole"
    assert robot["L_Ankle"] > robot["L_Toe"], "the ankle sits higher above the sole than the toe"

    # A human standing still, whose toe centre sits `h_toe` above the floor and ankle
    # `h_ankle`. `detect_stance_events` needs a run of slow frames, which standing gives.
    demo = ["Pelvis", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    h = {"L_Toe": 0.002, "R_Toe": 0.002, "L_Ankle": 0.052, "R_Ankle": 0.052}
    joints = np.zeros((60, len(demo), 3))
    joints[:, 0, 2] = 1.0
    for name, height in h.items():
        joints[:, demo.index(name), 2] = height

    offsets = foot_sole_offsets(model, joints, demo, 100.0, SMPLH_TO_MYOFULLBODY)

    for name, human_height in h.items():
        assert offsets[name] == pytest.approx(robot[name] - human_height, abs=1e-6), (
            f"{name}: the correction must be the robot's own offset minus the human's"
        )
        assert offsets[name] > 0, f"{name}: this robot's foot is deeper than this human's"


@requires_omni
def test_sole_offset_uses_flat_reference_when_nonflat_clip_never_reaches_floor(probed_model):
    from terra.constants import SMPLH_TO_MYOFULLBODY
    from terra.contacts import foot_sole_offsets, robot_sole_offsets

    model, _ = probed_model
    robot = robot_sole_offsets(model, SMPLH_TO_MYOFULLBODY)
    demo = ["Pelvis", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    reference = {"L_Toe": 0.002, "R_Toe": 0.002, "L_Ankle": 0.052, "R_Ankle": 0.052}
    joints = np.zeros((60, len(demo), 3))
    joints[:, 0, 2] = 1.0
    for name, height in reference.items():
        joints[:, demo.index(name), 2] = height + 0.20

    offsets = foot_sole_offsets(
        model,
        joints,
        demo,
        100.0,
        SMPLH_TO_MYOFULLBODY,
        human_offsets=reference,
    )

    for name, height in reference.items():
        assert offsets[name] == pytest.approx(robot[name] - height, abs=1e-6)


@requires_omni
def test_shifted_flat_reference_does_not_create_clearance_during_stance(probed_model):
    """A calibrated anatomical offset must follow the landmark target shift.

    Otherwise a robot-minus-human target correction is counted again as physical sole
    clearance, making the clearance term active throughout a flat planted stance.
    """
    import logging

    from terra.assembly import SolveContext, apply_sole_offsets
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.clearance import source_sole_clearance
    from terra.constants import SMPLH_TO_MYOFULLBODY

    model, _ = probed_model
    reference = {"L_Toe": 0.002, "R_Toe": 0.002, "L_Ankle": 0.052, "R_Ankle": 0.052}
    joints = np.zeros((60, len(SMPLH_DEMO_JOINTS), 3))
    joints[:, SMPLH_DEMO_JOINTS.index("Pelvis"), 2] = 1.0
    for name, height in reference.items():
        joints[:, SMPLH_DEMO_JOINTS.index(name), 2] = height

    ctx = SolveContext(
        config={"sole_offset_mode": "on", "source_sole_offsets": reference},
        logger=logging.getLogger("test_shifted_flat_reference"),
        model=model,
        terrain=None,
        fps=100.0,
        joints_mapping=SMPLH_TO_MYOFULLBODY,
        human_joints=joints,
        smpl_rotations=np.empty((60, 0, 3, 3)),
        scene={},
    )
    apply_sole_offsets(ctx)

    assert ctx.shifted_sole_offsets is not None
    clearance = source_sole_clearance(
        ctx.human_joints,
        list(SMPLH_DEMO_JOINTS),
        ctx.fps,
        source_offsets=ctx.shifted_sole_offsets,
    )
    np.testing.assert_allclose(clearance["l"], 0.0, atol=1e-6)
    np.testing.assert_allclose(clearance["r"], 0.0, atol=1e-6)


@requires_omni
def test_kinematic_stance_height_targets_support_not_floating_source(probed_model):
    """Kinematic contact must correct, not reproduce, a raised SMPL-H foot fit."""
    import logging

    from terra.assembly import SolveContext, attach_stance_height_targets
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import SMPLH_TO_MYOFULLBODY
    from terra.contacts import robot_sole_offsets

    model, _ = probed_model
    joints = np.zeros((12, len(SMPLH_DEMO_JOINTS), 3))
    for name in ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle"):
        joints[:, SMPLH_DEMO_JOINTS.index(name), 2] = 0.123

    class Capture:
        def attach_foot_stance_height(self, targets, activation, weight, recovery):
            self.targets = targets
            self.activation = activation
            self.weight = weight
            self.recovery = recovery

    capture = Capture()
    ctx = SolveContext(
        config={"stance_height_weight": 200.0},
        logger=logging.getLogger("test_support_referenced_stance"),
        model=model,
        terrain=None,
        fps=100.0,
        joints_mapping=SMPLH_TO_MYOFULLBODY,
        human_joints=joints,
        smpl_rotations=np.empty((12, 0, 3, 3)),
        scene={},
    )
    contact = {"L_Toe": np.ones(12, dtype=bool), "R_Toe": np.ones(12, dtype=bool)}
    attach_stance_height_targets(ctx, capture, contact)

    expected = robot_sole_offsets(model, SMPLH_TO_MYOFULLBODY)
    for joint, body in SMPLH_TO_MYOFULLBODY.items():
        if joint in expected:
            np.testing.assert_allclose(capture.targets[body], expected[joint])
            assert capture.activation[body][0] == pytest.approx(1.0)
            assert capture.activation[body][-1] == pytest.approx(1.0)
            assert not np.allclose(capture.targets[body], 0.123)


@requires_omni
def test_stance_height_source_support_tracks_the_loaded_end_of_the_foot(probed_model):
    """Heel rise and toe rise must not turn one side-level event into two flat feet."""
    import logging

    from loco_mujoco.core.terrain import TerrainSpec
    from terra.assembly import SolveContext, attach_stance_height_targets
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import SMPLH_TO_MYOFULLBODY

    model, _ = probed_model
    joints = np.zeros((8, len(SMPLH_DEMO_JOINTS), 3))
    # The left toe is the loaded probe and the left ankle/heel proxy has risen; the
    # right side is the converse heel-strike configuration.
    joints[:, SMPLH_DEMO_JOINTS.index("L_Toe"), 2] = 0.002
    joints[:, SMPLH_DEMO_JOINTS.index("L_Ankle"), 2] = 0.050
    joints[:, SMPLH_DEMO_JOINTS.index("R_Toe"), 2] = 0.050
    joints[:, SMPLH_DEMO_JOINTS.index("R_Ankle"), 2] = 0.002

    class Capture:
        def attach_foot_stance_height(self, targets, activation, weight, recovery):
            self.activation = activation

    capture = Capture()
    ctx = SolveContext(
        config={
            "stance_height_weight": 200.0,
            "stance_height_probe_mode": "source_support",
            "stance_height_probe_tolerance_m": 0.015,
            "stance_height_ramp_frames": 1,
        },
        logger=logging.getLogger("test_source_support_probe"),
        model=model,
        terrain=TerrainSpec(),
        fps=100.0,
        joints_mapping=SMPLH_TO_MYOFULLBODY,
        human_joints=joints,
        smpl_rotations=np.empty((8, 0, 3, 3)),
        scene={},
        shifted_sole_offsets=dict.fromkeys(("L_Toe", "R_Toe", "L_Ankle", "R_Ankle"), 0.0),
    )
    contact = {"L_Toe": np.ones(8, dtype=bool), "R_Toe": np.ones(8, dtype=bool)}
    attach_stance_height_targets(ctx, capture, contact)

    assert np.all(capture.activation["toes_l"] == 1.0)
    assert np.all(capture.activation["talus_l"] == 0.0)
    assert np.all(capture.activation["toes_r"] == 0.0)
    assert np.all(capture.activation["talus_r"] == 1.0)


@pytest.mark.parametrize(
    ("contact", "warmup", "expected"),
    [
        ([1, 1, 1, 1, 1, 1, 0, 0], 2, [1, 1, 1, 1, 1, 0, 0, 0]),
        ([0, 0, 1, 1, 1, 1], 0, [0, 0, 0, 1, 1, 1]),
        ([0, 1, 1, 1, 1, 0], 0, [0, 0, 1, 1, 0, 0]),
    ],
)
def test_midstance_mask_preserves_censored_recording_boundaries(contact, warmup, expected):
    """Only observed contact transitions are trimmed from the centered interval."""
    from terra.assembly import _midstance_mask

    actual = _midstance_mask(np.asarray(contact, dtype=bool), warmup, fraction=0.5)

    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=bool))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {"stance_landmark_clearance_cap_m": -0.01},
            "stance_landmark_clearance_cap_m",
        ),
        (
            {"stance_landmark_midstance_both_cap_m": np.nan},
            "stance_landmark_midstance_both_cap_m",
        ),
        (
            {
                "stance_landmark_clearance_cap_m": 0.01,
                "stance_landmark_midstance_fraction": 0.0,
            },
            "stance_landmark_midstance_fraction",
        ),
        (
            {
                "stance_landmark_clearance_cap_m": 0.01,
                "stance_landmark_probe_tolerance_m": "invalid",
            },
            "stance_landmark_probe_tolerance_m",
        ),
    ],
)
def test_stance_landmark_cap_config_rejects_invalid_policy(config, message):
    """Invalid optional policy values fail before any model-dependent work begins."""
    from terra.assembly import _stance_landmark_cap_config

    with pytest.raises(ValueError, match=message):
        _stance_landmark_cap_config(SolverConfig.from_mapping(config))


def test_stance_landmark_cap_config_allows_midstance_only_policy():
    """The both-end correction is independently configurable from the loaded-end cap."""
    from terra.assembly import _stance_landmark_cap_config

    config = _stance_landmark_cap_config(
        SolverConfig.from_mapping(
            {
                "stance_landmark_midstance_both_cap_m": 0.01,
            }
        )
    )

    assert config is not None
    assert config.loaded_cap is None
    assert config.midstance_cap == 0.01


def test_stance_landmark_contact_masks_validate_public_schedule_shape():
    """Malformed contact schedules report the offending toe instead of broadcasting."""
    from terra.assembly import _stance_contact_masks, _stance_landmark_cap_config

    config = _stance_landmark_cap_config(
        SolverConfig.from_mapping(
            {
                "stance_landmark_clearance_cap_m": 0.01,
            }
        )
    )
    assert config is not None

    with pytest.raises(ValueError, match="R_Toe"):
        _stance_contact_masks({"L_Toe": np.zeros(4, dtype=bool)}, 4, 0, config)
    with pytest.raises(ValueError, match=r"L_Toe.*shape \(4,\)"):
        _stance_contact_masks(
            {
                "L_Toe": np.zeros(3, dtype=bool),
                "R_Toe": np.zeros(4, dtype=bool),
            },
            4,
            0,
            config,
        )


@requires_omni
def test_stance_landmark_cap_preserves_unloaded_heel_rise(probed_model):
    """Only the loaded foot end is lowered; real foot roll remains in the landmarks."""
    import logging

    from loco_mujoco.core.terrain import TerrainSpec
    from terra.assembly import (
        SolveContext,
        apply_stance_landmark_clearance_cap,
    )
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import SMPLH_TO_MYOFULLBODY
    from terra.contacts import robot_sole_offsets

    model, _ = probed_model
    offsets = robot_sole_offsets(model, SMPLH_TO_MYOFULLBODY)
    joints = np.zeros((8, len(SMPLH_DEMO_JOINTS), 3))
    joints[:, SMPLH_DEMO_JOINTS.index("L_Toe"), 2] = offsets["L_Toe"] + 0.052
    joints[:, SMPLH_DEMO_JOINTS.index("L_Ankle"), 2] = offsets["L_Ankle"] + 0.100
    joints[:, SMPLH_DEMO_JOINTS.index("R_Toe"), 2] = offsets["R_Toe"] + 0.100
    joints[:, SMPLH_DEMO_JOINTS.index("R_Ankle"), 2] = offsets["R_Ankle"] + 0.052
    original = joints.copy()
    ctx = SolveContext(
        config={
            "stance_landmark_clearance_cap_m": 0.020,
            "stance_landmark_probe_tolerance_m": 0.015,
        },
        logger=logging.getLogger("test_stance_landmark_cap"),
        model=model,
        terrain=TerrainSpec(),
        fps=100.0,
        joints_mapping=SMPLH_TO_MYOFULLBODY,
        human_joints=joints,
        smpl_rotations=np.empty((8, 0, 3, 3)),
        scene={},
        shifted_sole_offsets=offsets,
    )
    contact = {"L_Toe": np.ones(8, dtype=bool), "R_Toe": np.ones(8, dtype=bool)}
    apply_stance_landmark_clearance_cap(ctx, contact)

    np.testing.assert_allclose(
        ctx.human_joints[:, SMPLH_DEMO_JOINTS.index("L_Toe"), 2],
        offsets["L_Toe"] + 0.020,
    )
    np.testing.assert_allclose(
        ctx.human_joints[:, SMPLH_DEMO_JOINTS.index("R_Ankle"), 2],
        offsets["R_Ankle"] + 0.020,
    )
    np.testing.assert_allclose(
        ctx.human_joints[:, SMPLH_DEMO_JOINTS.index("L_Ankle"), 2],
        original[:, SMPLH_DEMO_JOINTS.index("L_Ankle"), 2],
    )
    np.testing.assert_allclose(
        ctx.human_joints[:, SMPLH_DEMO_JOINTS.index("R_Toe"), 2],
        original[:, SMPLH_DEMO_JOINTS.index("R_Toe"), 2],
    )


@requires_omni
def test_stance_landmark_midstance_cap_lowers_both_foot_ends(probed_model):
    """The optional midstance handoff removes conversion tip-toe without changing roll edges."""
    import logging

    from loco_mujoco.core.terrain import TerrainSpec
    from terra.assembly import SolveContext, apply_stance_landmark_clearance_cap
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.constants import SMPLH_TO_MYOFULLBODY
    from terra.contacts import robot_sole_offsets

    model, _ = probed_model
    offsets = robot_sole_offsets(model, SMPLH_TO_MYOFULLBODY)
    joints = np.zeros((8, len(SMPLH_DEMO_JOINTS), 3))
    for joint in ("L_Toe", "R_Toe"):
        joints[:, SMPLH_DEMO_JOINTS.index(joint), 2] = offsets[joint] + 0.052
    for joint in ("L_Ankle", "R_Ankle"):
        joints[:, SMPLH_DEMO_JOINTS.index(joint), 2] = offsets[joint] + 0.100
    original = joints.copy()
    ctx = SolveContext(
        config={
            "stance_landmark_clearance_cap_m": 0.010,
            "stance_landmark_probe_tolerance_m": 0.015,
            "stance_landmark_midstance_both_cap_m": 0.010,
            "stance_landmark_midstance_fraction": 0.5,
        },
        logger=logging.getLogger("test_stance_landmark_midstance_cap"),
        model=model,
        terrain=TerrainSpec(),
        fps=100.0,
        joints_mapping=SMPLH_TO_MYOFULLBODY,
        human_joints=joints,
        smpl_rotations=np.empty((8, 0, 3, 3)),
        scene={},
        shifted_sole_offsets=offsets,
    )
    contact = np.zeros(8, dtype=bool)
    contact[2:6] = True
    apply_stance_landmark_clearance_cap(ctx, {"L_Toe": contact, "R_Toe": contact})

    toe = ctx.human_joints[:, SMPLH_DEMO_JOINTS.index("L_Toe"), 2]
    ankle = ctx.human_joints[:, SMPLH_DEMO_JOINTS.index("L_Ankle"), 2]
    np.testing.assert_allclose(toe[2:6], offsets["L_Toe"] + 0.010)
    np.testing.assert_allclose(ankle[3:5], offsets["L_Ankle"] + 0.010)
    np.testing.assert_allclose(ankle[[2, 5]], original[[2, 5], SMPLH_DEMO_JOINTS.index("L_Ankle"), 2])


def _coupler_residual_deg(model, couplers, q):
    """Max |dependent - poly(independent)| over all couplers, in degrees."""
    worst = 0.0
    for dep, indep, poly in couplers:
        x, y = q[indep], q[dep]
        pred = poly[0] + poly[1] * x + poly[2] * x**2 + poly[3] * x**3 + poly[4] * x**4
        worst = max(worst, abs(np.degrees(y - pred)))
    return worst


@requires_omni
def test_joint_couplers_found_and_well_formed(probed_model):
    """MyoFullBody drives its lumbar chain and shoulder girdle through joint equalities."""
    from terra._musclemimic import joint_couplers

    model, _ = probed_model
    couplers = joint_couplers(model)

    assert len(couplers) > 40, f"expected the full coupler set, found {len(couplers)}"
    for dep, indep, poly in couplers:
        assert 0 <= dep < model.nq and 0 <= indep < model.nq
        assert dep != indep
        assert poly.shape == (5,)
    assert len({d for d, _, _ in couplers}) == len(couplers), "a joint is driven twice"


@requires_omni
def test_coupler_step_reduces_violation(probed_model, random_pose):
    """The linearisation must actually pull a violating pose back onto the manifold.

    This is the whole fix: OmniRetarget optimises qpos with no knowledge of the couplers,
    which let the lumbar chain leave the manifold and produced the bimodal ~12.4 tendon
    flip. A sign error here would leave the solver just as free while looking wired up.
    """
    from terra._musclemimic import joint_couplers

    model, _ = probed_model
    couplers = joint_couplers(model)
    stub = _Stub(model)
    stub.attach_joint_couplers(couplers, weight=200.0)
    assert stub._constraints.objective.coupler.pairs, "no couplers landed inside the actuated index range"

    before = _coupler_residual_deg(model, couplers, random_pose)
    assert before > 1.0, "test pose does not violate the couplers; nothing to verify"

    amat, bvec = stub._coupler_linearisation(random_pose)
    dq = np.linalg.lstsq(amat, bvec, rcond=None)[0]

    q_new = random_pose.copy()
    q_new[stub.q_a_indices] += dq
    after = _coupler_residual_deg(model, couplers, q_new)

    assert after < before / 10.0, f"coupler step barely helped: {before:.2f} -> {after:.2f} deg"


@requires_omni
def test_batched_coupler_linearisation_is_bit_exact(probed_model, random_pose):
    """Static-index batching must reproduce the former scalar row loop exactly."""
    from terra._musclemimic import joint_couplers

    model, _ = probed_model
    stub = _Stub(model)
    stub.attach_joint_couplers(joint_couplers(model), weight=200.0)

    actual_a, actual_b = stub._coupler_linearisation(random_pose)
    expected_a = np.zeros_like(actual_a)
    expected_b = np.zeros_like(actual_b)
    for row, (dependent, independent, polynomial) in enumerate(stub._constraints.objective.coupler.pairs):
        x = random_pose[stub.q_a_indices[independent]]
        y = random_pose[stub.q_a_indices[dependent]]
        powers = np.array([1.0, x, x**2, x**3, x**4])
        derivative = np.array([0.0, 1.0, 2 * x, 3 * x**2, 4 * x**3])
        expected_a[row, dependent] = 1.0
        expected_a[row, independent] = -float(polynomial @ derivative)
        expected_b[row] = float(polynomial @ powers) - y

    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


@requires_omni
def test_coupler_residual_vanishes_on_the_manifold(probed_model, random_pose):
    """A pose that already satisfies the couplers must produce a zero residual."""
    from terra._musclemimic import joint_couplers

    model, _ = probed_model
    couplers = joint_couplers(model)
    stub = _Stub(model)
    stub.attach_joint_couplers(couplers, weight=200.0)

    q = random_pose.copy()
    for dep, indep, poly in couplers:  # project onto the manifold
        x = q[indep]
        q[dep] = poly[0] + poly[1] * x + poly[2] * x**2 + poly[3] * x**3 + poly[4] * x**4

    _amat, bvec = stub._coupler_linearisation(q)
    assert np.abs(bvec).max() < 1e-9


@requires_omni
def test_floor_clearance_measures_airborne_poses():
    """Ground alignment must measure a floating body, not report a fixed number.

    The datum for flat-ground alignment is the minimum of this over the motion, so a value
    that does not track the real height silently translates the whole trajectory by the
    difference. The margin-based implementation this replaced returned its own margin for
    any airborne pose - MuJoCo generates no plane contact once a geom clears the plane,
    whatever the margin is set to - which buried one clip of ten by 0.5 m with nothing
    downstream noticing.
    """
    from terra.collisions import collidable_geoms, floor_clearance

    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        contype=1,
        conaffinity=1,
    )
    body = spec.worldbody.add_body(name="ball")
    body.add_freejoint()
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0], contype=1, conaffinity=1)
    model = spec.compile()
    data = mujoco.MjData(model)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    probes = collidable_geoms(model, exclude=floor_id)
    assert probes, "the sphere should be collidable"

    for height, expected in [(0.1, 0.0), (0.3, 0.2), (1.1, 1.0), (0.05, -0.05)]:
        q = np.zeros(model.nq)
        q[2], q[3] = height, 1.0
        got = floor_clearance(model, data, q, floor_id, probes)
        assert got == pytest.approx(expected, abs=1e-6), (
            f"sphere centred at z={height} should read clearance {expected}, got {got}"
        )


# --- the frame the solver works in -----------------------------------------------------
# OmniRetarget normalises height by dropping the motion so its lowest toe sits at z=0, except
# that a motion whose lowest toe never comes below `mat_height` is taken to be standing on
# a 0.1 m mat and is left where it is. That rule describes OmniRetarget's own capture
# setup. On AMASS the same signature is a subject who spends the whole clip on a step or a
# staircase, and the 0.1 m it leaves in is a datum the terrain fitter cannot see: the
# fitter puts its lowest contact level at z=0 by construction, so motion and terrain end up
# in frames 0.1 m apart, and every stance foot hovers.


@requires_omni
def test_a_motion_on_a_step_is_not_mistaken_for_a_motion_on_a_mat():
    from types import SimpleNamespace

    from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
    from holosoma_retargeting.src.utils import preprocess_motion_data

    from terra.constants import NO_MAT_HEIGHT

    names = list(SMPLH_DEMO_JOINTS)
    retargeter = SimpleNamespace(demo_joints=names)

    def walk_on_a_step(height: float) -> np.ndarray:
        joints = np.zeros((20, len(names), 3))
        joints[:, :, 2] = height + 0.9
        for toe in ("L_Toe", "R_Toe"):
            joints[:, names.index(toe), 2] = height
        return joints

    toes = [names.index("L_Toe"), names.index("R_Toe")]
    # 0.15 m up: above the mat threshold, so OmniRetarget leaves 0.1 m of datum behind.
    omniretarget = preprocess_motion_data(walk_on_a_step(0.15), retargeter, ["L_Toe", "R_Toe"], scale=1.0)
    assert omniretarget[:, toes, 2].min() == pytest.approx(0.10)

    ours = preprocess_motion_data(
        walk_on_a_step(0.15), retargeter, ["L_Toe", "R_Toe"], scale=1.0, mat_height=NO_MAT_HEIGHT
    )
    assert ours[:, toes, 2].min() == pytest.approx(0.0)

    # Below the threshold the two agree, which is why flat-ground results are untouched.
    low = walk_on_a_step(0.05)
    assert preprocess_motion_data(low.copy(), retargeter, ["L_Toe", "R_Toe"], scale=1.0)[
        :, toes, 2
    ].min() == pytest.approx(0.0)


# A foot whose lowest footfall is a tread up never sees the lowest surface, so
# `joint_surface_offsets` reads the step height into that joint's offset: measured on
# `KIT/359/downstairs09_poses`, L_Ankle came out at +56 mm and R_Ankle at +264 mm, a
# difference of one riser. Both feet are the same size, so the lower estimate is the sound
# one.


def test_a_foot_that_never_reached_the_lowest_surface_takes_its_partner_s_offset():
    from terra.terrain import paired_sole_offsets

    paired = paired_sole_offsets({"L_Ankle": 0.056, "R_Ankle": 0.264, "L_Toe": 0.010, "R_Toe": 0.227})
    assert paired["L_Ankle"] == pytest.approx(0.056)
    assert paired["R_Ankle"] == pytest.approx(0.056)
    assert paired["R_Toe"] == pytest.approx(0.010)


def test_two_sound_estimates_are_left_alone():
    from terra.terrain import paired_sole_offsets

    # Both feet visit the floor: the sides already agree to a few millimetres, and taking
    # the minimum must not quietly bias a good estimate downward by more than that.
    human = {"L_Ankle": 0.050, "R_Ankle": 0.045, "L_Toe": 0.003, "R_Toe": 0.005}
    paired = paired_sole_offsets(human)
    assert paired["L_Ankle"] == pytest.approx(0.045)
    assert paired["L_Toe"] == pytest.approx(0.003)
    # Nothing moves further than the two sides already disagreed, so a sound pair keeps its
    # estimate to within the spread of its own measurement.
    for left, right in (("L_Ankle", "R_Ankle"), ("L_Toe", "R_Toe")):
        spread = abs(human[left] - human[right])
        assert max(abs(paired[j] - human[j]) for j in (left, right)) <= spread


def test_a_landmark_with_no_partner_is_passed_through():
    from terra.terrain import paired_sole_offsets

    assert paired_sole_offsets({"L_Ankle": 0.05}) == {"L_Ankle": 0.05}


# --- the foot orientation datum --------------------------------------------------------
# `smpl2robot_rot_mat` aligns SMPL's zero pose to the robot's `qpos0`, and those two poses
# agree everywhere except the foot: SMPL's is plantarflexed. The alignment therefore carries
# a constant dorsiflexion into every foot orientation target, and the orientation cost holds
# the forefoot 20-50 mm off the surface through every stance.
#
# The correction is a rotation, so a sign or frame error produces one of the right magnitude
# pointing the wrong way - which doubles the tilt instead of removing it. Both tests below
# check the *corrected target*, not the returned angle, for exactly that reason.

FOOT_SITE_BODIES = {
    "left_ankle_mimic": "talus_l",
    "left_toes_mimic": "toes_l",
    "right_ankle_mimic": "talus_r",
    "right_toes_mimic": "toes_r",
}
#: A flat-footed human standing still: each contact joint at its own height above the floor,
#: held for long enough that `detect_stance_events` sees a stance.
FLAT_STANCE_HEIGHTS = {"L_Toe": 0.002, "R_Toe": 0.002, "L_Ankle": 0.052, "R_Ankle": 0.052}


@pytest.fixture(scope="module")
def foot_site_model():
    """MyoFullBody carrying the four foot mimic sites, at non-identity site frames.

    Non-identity because the correction is expressed in *site* coordinates: a version that
    silently worked in body or world coordinates would pass against the shipped sites, whose
    frames happen to equal their bodies'.
    """
    from musclemimic_models import get_xml_path

    from musclemimic.utils.retarget.msk_metrics import apply_spec_changes

    spec = apply_spec_changes(mujoco.MjSpec.from_file(str(get_xml_path("myofullbody"))))
    for i, (site, body) in enumerate(FOOT_SITE_BODIES.items()):
        quat = Rotation.from_euler("xyz", [0.4 * (i + 1), -0.3, 0.2 * (i + 1)]).as_quat()
        spec.body(body).add_site(name=site, quat=[quat[3], quat[0], quat[1], quat[2]])
    model = spec.compile()
    names = list(FOOT_SITE_BODIES)
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s) for s in names])
    assert (ids >= 0).all()
    return model, ids, names


def _flat_stance_source(n_frames: int = 60, **overrides) -> tuple[np.ndarray, list[str]]:
    """A standing source motion, with any contact joint's height overridden."""
    demo = ["Pelvis", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    heights = FLAT_STANCE_HEIGHTS | overrides
    joints = np.zeros((n_frames, len(demo), 3))
    joints[:, 0, 2] = 1.0
    for name, height in heights.items():
        joints[:, demo.index(name), 2] = height
    return joints, demo


def _tilted_targets(model, site_ids, tilt: Rotation, n_frames: int = 60) -> np.ndarray:
    """(T, K, 3, 3) targets: each site's own flat-footed orientation, tilted in world."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    flat = np.array([data.site_xmat[s].reshape(3, 3) for s in site_ids])
    return np.repeat((tilt.as_matrix() @ flat)[None], n_frames, axis=0)


def _sole_direction(model, site_ids, targets, frame: int = 0) -> np.ndarray:
    """(K, 3) where each site's sole faces under its target, in world."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    down = np.array([0.0, 0.0, -1.0])
    return np.array([targets[frame, k] @ (data.site_xmat[s].reshape(3, 3).T @ down) for k, s in enumerate(site_ids)])


def test_the_correction_turns_a_tilted_foot_target_flat_again(foot_site_model):
    """A known dorsiflexion in, the same rotation out, and a vertical sole after applying it."""
    from terra.contacts import foot_orientation_offsets

    model, site_ids, names = foot_site_model
    tilt_deg = 12.0
    targets = _tilted_targets(model, site_ids, Rotation.from_rotvec([0, np.radians(tilt_deg), 0]))
    joints, demo = _flat_stance_source()

    before = _sole_direction(model, site_ids, targets)
    assert np.allclose(np.degrees(np.arccos(-before[:, 2])), tilt_deg, atol=1e-6), (
        "precondition: every target's sole is tilted by exactly the injected angle"
    )

    corrections = foot_orientation_offsets(model, site_ids, names, targets, joints, demo, 100.0)
    assert set(corrections) == set(names), "every foot site is measurable from a flat stance"

    for site in names:
        angle = np.degrees(np.linalg.norm(Rotation.from_matrix(corrections[site]).as_rotvec()))
        assert angle == pytest.approx(tilt_deg, abs=1e-3), f"{site}: recovers the tilt"
    corrected = targets.copy()
    for k, site in enumerate(names):
        corrected[:, k] = corrected[:, k] @ corrections[site]

    after = _sole_direction(model, site_ids, corrected)
    assert np.allclose(after, [0.0, 0.0, -1.0], atol=1e-9), "the corrected targets put every sole flat on the surface"


def test_a_foot_that_is_never_flat_is_left_uncorrected(foot_site_model):
    """A toe on the floor with the ankle 60 mm up is a toe-off, not a calibration frame.

    Estimating the datum from it would read the source's own toe-off as a model mismatch and
    bake it into every frame of the motion.
    """
    from terra.contacts import flat_stance_frames, foot_orientation_offsets

    model, site_ids, names = foot_site_model
    targets = _tilted_targets(model, site_ids, Rotation.from_rotvec([0, np.radians(12), 0]))
    joints, demo = _flat_stance_source(L_Ankle=0.112)  # left heel up, right foot flat

    frames = flat_stance_frames(joints, demo, 100.0)
    assert not len(frames["l"]), "the left foot is never flat"
    assert len(frames["r"]), "the right foot is, and must not be dragged down with it"

    corrections = foot_orientation_offsets(model, site_ids, names, targets, joints, demo, 100.0)
    assert not {s for s in corrections if s.startswith("left")}
    assert {s for s in corrections if s.startswith("right")} == {"right_ankle_mimic", "right_toes_mimic"}


def _slope_source(slope_deg: float, n_frames: int = 120):
    """A source that stands flat on an incline, then flat on the floor beside it.

    Both halves are calibration frames, so a correction that is a property of the two
    skeletons has to come out the same from either. The floor half is also what makes the
    offsets inside the foot measurable: `flat_stance_frames` estimates them from the lowest
    surface the motion visits, and a clip that never leaves the slope has none.
    """
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec

    pitch = -np.radians(slope_deg)
    length, rise = 3.0, 3.0 * np.tan(np.radians(slope_deg))
    nz = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    ramp = BoxSpec(
        pos=tuple(np.array([length / 2, 0.0, rise / 2]) - 0.5 * nz),
        size=(0.5 * length / np.cos(pitch), 0.6, 0.5),
        pitch=pitch,
    )
    terrain = TerrainSpec(boxes=(ramp,))

    demo = ["Pelvis", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    joints = np.zeros((n_frames, len(demo), 3))
    half = n_frames // 2
    # Above the feet, not at a fixed remote point: `detect_seat_rests` (now read by
    # `flat_stance_frames`) would otherwise see a pelvis standing still 1.5-5.0 m outside
    # the foot hull and mistake this stance for a seat, dropping every calibration frame.
    joints[:half, 0] = [1.5, 0.1, 1.0]
    joints[half:, 0] = [5.0, 0.1, 1.0]
    for name, offset in FLAT_STANCE_HEIGHTS.items():
        x = 1.5 if name.endswith("Ankle") else 1.5 + 0.11  # the toe leads, up the slope
        surface = float(terrain.height_at(x, 0.1))
        joints[:half, demo.index(name)] = [x, 0.1, surface + offset]
        joints[half:, demo.index(name)] = [5.0, 0.1, offset]  # off the ramp, on the floor
    return joints, demo, terrain, ramp.rotation[:, 2]


def test_the_foot_datum_is_the_surface_the_foot_is_on_not_the_world_vertical(foot_site_model):
    """ "Flat on the surface" is a statement about the surface, and a slope is not vertical.

    Calibrating the correction against world-down makes it absorb the incline: the constant
    it returns is the rest-pose mismatch *plus* however much ramp the flat frames sampled,
    and the corrected target then asks for a foot held horizontal on ground that is not. On
    a descent that lifts the forefoot off the surface by the slope angle over a foot length,
    which is the mode the correction exists to remove. Read against each foot's own surface,
    the estimate is the same from the ramp frames and the floor frames alike.
    """
    from terra.contacts import flat_stance_frames, foot_orientation_offsets

    model, site_ids, names = foot_site_model
    mismatch, slope_deg = 12.0, 15.0
    joints, demo, terrain, normal = _slope_source(slope_deg)

    # The target a flat foot on this slope really has: the flat-footed one, laid on the
    # incline, then tilted by the mismatch the correction is supposed to recover.
    axis = np.cross([0.0, 0.0, 1.0], normal)
    lie = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(slope_deg))
    tilt = Rotation.from_rotvec([0, np.radians(mismatch), 0])
    on_slope = _tilted_targets(model, site_ids, tilt * lie, n_frames=len(joints))
    on_floor = _tilted_targets(model, site_ids, tilt, n_frames=len(joints))
    targets = np.concatenate([on_slope[: len(joints) // 2], on_floor[len(joints) // 2 :]])

    frames = flat_stance_frames(joints, demo, 100.0, terrain)
    assert len(frames["l"]) > 20 and (frames["l"] < len(joints) // 2).any(), (
        "precondition: the foot is certified flat on the ramp as well as on the floor"
    )

    corrections = foot_orientation_offsets(model, site_ids, names, targets, joints, demo, 100.0, terrain=terrain)
    assert set(corrections) == set(names)
    for site in names:
        angle = np.degrees(np.linalg.norm(Rotation.from_matrix(corrections[site]).as_rotvec()))
        assert angle == pytest.approx(mismatch, abs=1e-3), (
            f"{site}: {angle:.2f} deg, against the {mismatch:.0f} deg of model mismatch - "
            f"a reading near {mismatch + slope_deg:.0f} or {abs(mismatch - slope_deg):.0f} "
            "is the slope leaking into the constant"
        )

    # And the corrected target puts the sole on the surface rather than level with the world.
    corrected = targets.copy()
    for k, site in enumerate(names):
        corrected[:, k] = corrected[:, k] @ corrections[site]
    on_ramp = _sole_direction(model, site_ids, corrected, frame=0)
    assert np.allclose(on_ramp, -normal, atol=1e-9), "flat on the incline, not on the world"
    assert np.allclose(
        _sole_direction(model, site_ids, corrected, frame=len(joints) - 1), [0.0, 0.0, -1.0], atol=1e-9
    ), "and flat on the floor beside it"


def test_a_foot_at_the_brink_of_a_ramp_is_not_a_calibration_frame(foot_site_model):
    """Heel on the landing, toe over the incline: there is no one plane the sole is on.

    The correction is a constant, so every frame that contributes has to be one where the
    two skeletons' feet are in the same posture. A foot straddling the top of a ramp is not
    - and it is not a rare frame either. `KIT/3/slope_down03_poses` opens with a 260-frame
    stance at exactly that brink, which supplied 73% of the calibration set and pinned the
    constant to its flat-ground value for the whole descent.

    Two surfaces of the *same* tilt agree by definition, so the rule never drops a frame on
    flat-topped terrain, where a foot straddling a riser is the normal way to stand.
    """
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.contacts import _surface_into

    demo = ["Pelvis", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    pitch = -np.radians(15.0)
    nz = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
    rise = 3.0 * np.tan(np.radians(15.0))
    incline = BoxSpec(
        pos=tuple(np.array([1.5, 0.0, rise / 2]) - 0.5 * nz), size=(1.5 / np.cos(pitch), 0.6, 0.5), pitch=pitch
    )
    landing = BoxSpec(pos=(3.3, 0.0, rise / 2), size=(0.3, 0.6, rise / 2))
    ramped = TerrainSpec(boxes=(incline, landing))
    stairs = TerrainSpec(
        boxes=(
            BoxSpec(pos=(0.15, 0.0, 0.10), size=(0.15, 0.5, 0.10)),
            BoxSpec(pos=(0.45, 0.0, 0.20), size=(0.15, 0.5, 0.20)),
        )
    )

    joints = np.zeros((3, len(demo), 3))
    for t, (toe_x, ankle_x) in enumerate([(1.5, 1.39), (2.95, 3.06), (3.3, 3.19)]):
        joints[t, demo.index("L_Toe"), :2] = [toe_x, 0.0]
        joints[t, demo.index("L_Ankle"), :2] = [ankle_x, 0.0]
    _, usable = _surface_into(joints, demo, "l", np.arange(3), ramped)
    assert list(usable) == [True, False, True], "mid-incline and mid-landing are usable; the brink between them is not"

    # The same straddle across a riser, where both surfaces are level: kept.
    joints[0, demo.index("L_Toe"), :2] = [0.39, 0.0]
    joints[0, demo.index("L_Ankle"), :2] = [0.28, 0.0]
    _, usable = _surface_into(joints, demo, "l", np.arange(1), stairs)
    assert bool(usable[0]), "a foot straddling a riser stands on one plane"


def test_a_side_that_is_never_flat_on_terrain_returns_empty_rather_than_raising():
    """No flat-footed frames on one side is an ordinary outcome, not an error.

    `foot_orientation_offsets` already handles it - it logs and skips the correction for that
    side - but it can only do so if it is handed an empty result. The loop builds its answer
    by stacking a list, and stacking nothing raises, so on non-flat terrain the empty case
    reached `np.stack([])` and took the whole motion down. Reachable wherever one foot never
    goes flat on terrain, and common on chairs, where a foot spends the sit tucked under
    itself: `BioMotionLab_NTroje/rub001/0010_sitting2_poses` fits a 59 mm box from its
    footfalls *and* a seat, so the terrain is not flat and the empty path is taken.
    """
    from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
    from terra.contacts import _surface_into

    demo = ["Pelvis", "L_Toe", "R_Toe", "L_Ankle", "R_Ankle"]
    joints = np.zeros((5, len(demo), 3))
    terrain = TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.03), size=(0.3, 0.3, 0.03)),))

    into, usable = _surface_into(joints, demo, "l", np.array([], dtype=int), terrain)
    assert into.shape == (0, 3)
    assert usable.shape == (0,)
    assert into[usable].shape == (0, 3), "the caller indexes one by the other"


def test_an_over_large_estimate_is_discarded_rather_than_clamped(foot_site_model):
    """Past the cap the axis is no longer trustworthy, so there is nothing to clamp along."""
    from terra.contacts import foot_orientation_offsets
    from terra.defaults import MAX_FOOT_ORIENT_OFFSET

    model, site_ids, names = foot_site_model
    over = np.degrees(MAX_FOOT_ORIENT_OFFSET) + 10.0
    targets = _tilted_targets(model, site_ids, Rotation.from_rotvec([0, np.radians(over), 0]))
    joints, demo = _flat_stance_source()

    assert not foot_orientation_offsets(model, site_ids, names, targets, joints, demo, 100.0)


# --- non-penetration, when the QP cannot satisfy it ------------------------------------
# Non-penetration is the only *hard* constraint added here, and the frame it engages on is
# the one that can be jointly unsatisfiable: the body is still in its warm-up pose, and a
# low box under a foot gives the sole geoms top-face and side-face normals at once. OmniRetarget
# raises, which loses the whole motion. `EKUT/EKUT/265/SS2D102_poses` did exactly that once
# its terrain gained a 47 mm pad under the opening stance, at frame 25 of 573 - the frame the
# constraint engages.


@requires_omni
def test_an_infeasible_frame_relaxes_non_penetration_instead_of_losing_the_motion(probed_model, monkeypatch):
    """One infeasible iteration must cost one iteration, not the whole motion."""
    model, _ = probed_model
    r = _Stub(model)
    environment = r._constraints.environment
    environment.geom_ids = {0}
    r._current_frame = 10

    base = TerraRetargeter.__mro__[1]
    seen = []

    def fake(self, *args, **kwargs):
        seen.append(self._constraints.environment.suppressed)
        if len(seen) == 1:
            raise RuntimeError("CVXPY solve failed: infeasible")
        return "solved"

    monkeypatch.setattr(base, "solve_single_iteration", fake, raising=False)

    assert r._solve_or_relax_nonpen() == "solved"
    # First attempt with the constraint, retry without it, and nothing left suppressed.
    assert seen == [False, True]
    assert environment.suppressed is False
    assert environment.relaxed_frames == {10}


@requires_omni
def test_a_solver_error_relaxes_non_penetration_when_no_status_is_available(probed_model, monkeypatch):
    """Clarabel can raise before CVXPY can report the equivalent infeasible status."""
    import cvxpy as cp

    model, _ = probed_model
    r = _Stub(model)
    environment = r._constraints.environment
    environment.geom_ids = {0}
    r._current_frame = 25

    base = TerraRetargeter.__mro__[1]
    seen = []

    def fake(self, *args, **kwargs):
        seen.append(self._constraints.environment.suppressed)
        if len(seen) == 1:
            raise cp.error.SolverError("Solver 'CLARABEL' failed")
        return "solved"

    monkeypatch.setattr(base, "solve_single_iteration", fake, raising=False)

    assert r._solve_or_relax_nonpen() == "solved"
    assert seen == [False, True]
    assert environment.suppressed is False
    assert environment.relaxed_frames == {25}


@requires_omni
def test_a_failure_that_is_not_infeasibility_still_propagates(probed_model, monkeypatch):
    """Only infeasibility is recoverable; anything else has to reach the caller."""
    model, _ = probed_model
    r = _Stub(model)
    base = TerraRetargeter.__mro__[1]
    monkeypatch.setattr(
        base,
        "solve_single_iteration",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("CVXPY solve failed: unbounded")),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="unbounded"):
        r._solve_or_relax_nonpen()
    assert not r.nonpenetration_relaxed_frames


@requires_omni
def test_still_infeasible_without_the_constraint_is_not_swallowed(probed_model, monkeypatch):
    """Suppressing the constraint is one attempt, not an infinite excuse."""
    model, _ = probed_model
    r = _Stub(model)
    base = TerraRetargeter.__mro__[1]
    monkeypatch.setattr(
        base,
        "solve_single_iteration",
        lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("CVXPY solve failed: infeasible")),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="infeasible"):
        r._solve_or_relax_nonpen()
    assert r._constraints.environment.suppressed is False


@requires_omni
def test_suppressing_non_penetration_drops_every_row_and_only_for_that_iteration(probed_model):
    """The suppression has to be total while it lasts, and gone the moment it is over."""
    model, _ = probed_model
    r = _Stub(model)
    environment = r._constraints.environment
    environment.geom_ids = {0}
    environment.engage_from_frame = 0
    r._current_frame = 10

    environment.suppressed = True
    assert r._update_jacobians_and_phis_from_q(np.zeros(model.nq)) == ({}, {})

    # Off again, the frame gate is the only thing left deciding.
    environment.suppressed = False
    environment.engage_from_frame = 50
    assert r._update_jacobians_and_phis_from_q(np.zeros(model.nq)) == ({}, {})


@requires_omni
def test_suppression_also_drops_inherited_omniretarget_object_rows(probed_model, monkeypatch):
    """The clean baseline uses OmniRetarget's object-name collision dispatch."""
    model, _ = probed_model
    r = _Stub(model)
    environment = r._constraints.environment
    environment.geom_ids = None
    base = TerraRetargeter.__mro__[1]
    sentinel = ({("robot", "terrain"): np.ones(model.nv)}, {("robot", "terrain"): -0.01})
    monkeypatch.setattr(
        base,
        "_update_jacobians_and_phis_from_q",
        lambda self, q: sentinel,
        raising=False,
    )

    assert r._update_jacobians_and_phis_from_q(np.zeros(model.nq)) is sentinel
    environment.suppressed = True
    assert r._update_jacobians_and_phis_from_q(np.zeros(model.nq)) == ({}, {})


def test_quadratic_term_has_identical_numeric_and_cvxpy_objectives():
    import cvxpy as cp

    from terra._sqp import _QuadraticTerm

    step_value = np.array([0.2, -0.3])
    jacobian = np.array([[1.0, 2.0], [-0.5, 3.0]])
    target = np.array([0.4, -0.1])
    row_scale = np.array([2.0, 0.5])
    terms = [
        _QuadraticTerm(jacobian, target, 3.0),
        _QuadraticTerm(jacobian, target, row_scale**2, cvxpy_row_scale=row_scale),
    ]

    step = cp.Variable(2)
    step.value = step_value
    for term in terms:
        numeric_jacobian, numeric_target, weight = term.as_native_term()
        residual = numeric_jacobian @ step_value - numeric_target
        expected = float(np.sum(np.asarray(weight) * residual**2))
        assert term.cvxpy_expression(cp, step).value == pytest.approx(expected, abs=1e-14)


@requires_omni
def test_attached_quadratic_terms_preserve_scientific_signs_and_order(probed_model, monkeypatch):
    model, _ = probed_model
    retargeter = _Stub(model)
    n_dof = retargeter.nq_a
    row = np.arange(n_dof, dtype=float)

    objective = retargeter._constraints.objective
    objective.orientation.site_ids = np.array([0])
    objective.orientation.weights = np.array([4.0])
    objective.foot_anchor.contact = {}
    objective.self_collision.pairs = [(0, 1)]
    objective.clearance.geoms = {0: "left"}
    objective.route.targets = {0: np.zeros(1)}
    objective.stance_height.targets = {0: np.zeros(1)}
    objective.seat_contact.glute_geoms = (0,)
    objective.coupler.pairs = [(0, 1, np.zeros(5))]
    objective.coupler.weight = 8.0

    monkeypatch.setattr(
        retargeter,
        "_orientation_linearisation",
        lambda _q, _frame: (np.array([1.0, 2.0, 3.0]), np.vstack([row, row + 1.0, row + 2.0])),
    )
    monkeypatch.setattr(
        retargeter,
        "_foot_anchor_terms",
        lambda _q, _last, _frame: [(2.0, np.vstack([row, row + 1.0]), np.array([4.0, 5.0]))],
    )
    monkeypatch.setattr(retargeter, "_self_collision_terms", lambda _q: [(3.0, row, 6.0)])
    monkeypatch.setattr(retargeter, "_clearance_terms", lambda _q: [(4.0, row, 7.0)])
    monkeypatch.setattr(retargeter, "_foot_route_terms", lambda _q: [(5.0, row, 8.0)])
    monkeypatch.setattr(retargeter, "_foot_stance_height_terms", lambda _q: [(6.0, row, 9.0)])
    monkeypatch.setattr(retargeter, "_seat_contact_terms", lambda _q: [(7.0, row, 10.0)])
    monkeypatch.setattr(retargeter, "_coupler_linearisation", lambda _q: (row[None], np.array([11.0])))

    terms = retargeter._attached_quadratic_terms(
        np.zeros(model.nq),
        np.zeros(model.nq),
        frame_idx=12,
    )

    assert len(terms) == 8
    expected_targets = ([1.0, 2.0, 3.0], [-4.0, -5.0], [6.0], [7.0], [8.0], [9.0], [10.0], [11.0])
    expected_weights = ([4.0, 4.0, 4.0], 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    for term, target, weight in zip(terms, expected_targets, expected_weights, strict=True):
        np.testing.assert_array_equal(term.target, target)
        np.testing.assert_array_equal(term.weight, weight)

    assert retargeter.active_term_names() == (
        "orientation",
        "foot_anchor",
        "self_collision",
        "swing_clearance",
        "foot_route",
        "stance_height",
        "seat_contact",
        "joint_coupler",
    )


@requires_omni
def test_legacy_iteration_without_attached_costs_delegates_unchanged(probed_model, monkeypatch):
    model, _ = probed_model
    retargeter = _Stub(model)
    retargeter._solver_backend = "legacy"
    objective = retargeter._constraints.objective
    objective.orientation.site_ids = None
    objective.foot_anchor.contact = None
    objective.clearance.geoms = {}
    objective.route.targets = {}
    objective.stance_height.targets = {}
    objective.seat_contact.glute_geoms = ()
    objective.self_collision.pairs = []
    objective.coupler.pairs = []
    bound = SimpleNamespace(arguments={"frame_idx": 17})
    monkeypatch.setattr(retargeter, "_bind_solve_arguments", lambda _args, _kwargs: bound)
    calls = []
    monkeypatch.setattr(
        retargeter,
        "_solve_or_relax_nonpen",
        lambda *args, **kwargs: calls.append((args, kwargs)) or "base-result",
    )

    result = retargeter.solve_single_iteration("argument", option=True)

    assert result == "base-result"
    assert calls == [(("argument",), {"option": True})]
    assert retargeter._current_frame == 17


@requires_omni
def test_iteration_rejects_unknown_backend_before_objective_assembly(probed_model, monkeypatch):
    model, _ = probed_model
    retargeter = _Stub(model)
    retargeter._solver_backend = "mystery"
    bound = SimpleNamespace(arguments={"frame_idx": 23})
    monkeypatch.setattr(retargeter, "_bind_solve_arguments", lambda _args, _kwargs: bound)

    with pytest.raises(ValueError, match="solver_backend"):
        retargeter.solve_single_iteration()

    assert retargeter._current_frame == 23


def test_condensed_inequalities_preserve_native_row_and_cvxpy_constraint_order():
    import cvxpy as cp

    from terra._sqp import _InequalityConstraints

    constraints = _InequalityConstraints(n_dof=2)
    constraints.add_lower_bound(np.array([1.0, 2.0]), -1.0)
    constraints.add_two_sided(
        np.array([[3.0, 4.0], [5.0, 6.0]]),
        np.array([-2.0, -3.0]),
        np.array([2.0, 3.0]),
    )
    constraints.add_variable_bounds(np.array([-0.5, -0.6]), np.array([0.5, 0.6]))

    rows, rhs = constraints.native()
    np.testing.assert_array_equal(
        rows,
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [-3.0, -4.0],
                [-5.0, -6.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ),
    )
    np.testing.assert_array_equal(rhs, [-1.0, -2.0, -3.0, -2.0, -3.0, -0.5, -0.6, -0.5, -0.6])

    step = cp.Variable(2)
    step.value = np.zeros(2)
    cvxpy_constraints = constraints.cvxpy(step)
    assert len(cvxpy_constraints) == 5
    assert all(float(np.max(constraint.violation(), initial=0.0)) == 0.0 for constraint in cvxpy_constraints)

    with pytest.raises(ValueError, match="inequality shape"):
        constraints.add_lower_bound(np.ones(3), 0.0)


def test_laplacian_linearization_has_equivalent_native_and_cvxpy_forms():
    import cvxpy as cp

    from terra._sqp import _LaplacianLinearization

    jacobian = np.array([[1.0, -2.0], [0.5, 3.0]])
    current = np.array([0.2, -0.4])
    target = np.array([0.7, 0.1])
    row_scale = np.array([2.0, 0.25])
    linearization = _LaplacianLinearization(jacobian, current, target, row_scale)

    native_jacobian, native_target, native_weight = linearization.as_native_term()
    np.testing.assert_array_equal(native_jacobian, jacobian)
    np.testing.assert_array_equal(native_target, target - current)
    np.testing.assert_array_equal(native_weight, row_scale**2)

    step_value = np.array([0.3, -0.2])
    residual = jacobian @ step_value + current - target
    expected = float(np.sum((row_scale * residual) ** 2))
    step = cp.Variable(2)
    step.value = step_value
    assert linearization.cvxpy_expression(cp, step).value == pytest.approx(expected, abs=1e-14)


@requires_omni
@pytest.mark.parametrize(
    "backend,native_result,expected_calls",
    [
        ("native_clarabel", "native-result", ["native"]),
        ("native_clarabel", None, ["native", "cvxpy"]),
        ("condensed_cvxpy", None, ["cvxpy"]),
    ],
)
def test_condensed_iteration_routes_native_success_and_fallback(
    probed_model,
    monkeypatch,
    backend,
    native_result,
    expected_calls,
):
    model, _ = probed_model
    retargeter = _Stub(model)
    retargeter._solver_backend = backend
    q_locked = model.qpos0.copy()
    q_a_n_last = q_locked[retargeter.q_a_indices].copy()
    bound = SimpleNamespace(
        arguments={
            "q_locked": q_locked,
            "q_a_n_last": q_a_n_last,
            "q_t_last": q_locked.copy(),
            "obj_pts_local": np.zeros((1, 3)),
            "adj_list": [],
            "target_laplacian": np.zeros((1, 3)),
            "foot_sticking": {},
            "frame_idx": 4,
            "init_t": True,
        }
    )
    laplacian = object()
    constraints = object()
    calls = []
    monkeypatch.setattr(retargeter, "_condensed_laplacian_linearization", lambda *_args: laplacian)
    monkeypatch.setattr(retargeter, "_condensed_constraints", lambda *_args: constraints)
    monkeypatch.setattr(retargeter, "_trust_radius", lambda initial: calls.append(("trust", initial)) or 0.7)
    monkeypatch.setattr(
        retargeter,
        "_try_native_condensed_solve",
        lambda *_args: calls.append("native") or native_result,
    )
    monkeypatch.setattr(
        retargeter,
        "_solve_cvxpy_condensed",
        lambda *_args: calls.append("cvxpy") or "cvxpy-result",
    )

    result = retargeter._solve_condensed_iteration(bound, [])

    assert calls[0] == ("trust", True)
    assert calls[1:] == expected_calls
    assert result == (native_result or "cvxpy-result")


@requires_omni
def test_native_condensed_failure_records_auditable_fallback(probed_model, monkeypatch):
    import terra.retargeter as retargeter_module
    from terra._sqp import _InequalityConstraints

    model, _ = probed_model
    retargeter = _Stub(model)
    retargeter._native_fallback_count = 0
    retargeter._native_fallback_frames = frozenset()
    retargeter._native_fallback_reasons = ()
    q = model.qpos0.copy()
    q_a_n_last = q[retargeter.q_a_indices].copy()
    monkeypatch.setattr(retargeter, "_native_condensed_objective", lambda *_args: ([], []))
    monkeypatch.setattr(
        retargeter_module,
        "_native_clarabel_qp",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unstable normal equations")),
    )

    result = retargeter._try_native_condensed_solve(
        q,
        q_a_n_last,
        np.zeros_like(q_a_n_last),
        object(),
        _InequalityConstraints(retargeter.nq_a),
        [],
        {
            "w_nominal_tracking": 0.0,
            "q_a_nominal": None,
            "verbose": False,
            "frame_idx": 31,
        },
        trust_radius=0.2,
    )

    assert result is None
    assert retargeter._native_fallback_count == 1
    assert retargeter._native_fallback_frames == {31}
    assert retargeter._native_fallback_reasons == ("frame 31: unstable normal equations",)
