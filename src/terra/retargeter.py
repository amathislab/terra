"""Extend OmniRetarget with collision, contact, and orientation costs."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy import sparse as sp

from terra._interaction_mesh import _calculate_laplacian_coordinates, _get_adjacency_list
from terra._qp import _native_clarabel_qp
from terra._sqp import (
    _CvxpyShim,
    _InequalityConstraints,
    _LaplacianLinearization,
    _objective_with_quadratic_terms,
    _QuadraticTerm,
)
from terra.baselines.omniretarget import OMNIRETARGET_INSTALLED, InteractionMeshRetargeter
from terra.retarget_terms import RetargeterConstraintState
from terra.solver_backend import (
    cvxpy_condensed_objective,
    native_condensed_objective,
    updated_pose,
)

if TYPE_CHECKING:
    from holosoma_retargeting.config_types.retargeter import FootLockConfig, SelfCollisionConfig


class TerraRetargeter(InteractionMeshRetargeter if OMNIRETARGET_INSTALLED else object):
    """Add optional TERRA costs to the OmniRetarget interaction-mesh solver.

    The attach methods configure environment collision, self-collision, swing
    clearance, foot routing, orientation, joint couplers, and stance anchoring.
    Attached terms are linearized for each SQP iteration and included in the
    selected QP backend.
    """

    def __init__(
        self,
        task_constants: ModuleType,
        object_urdf_path: str | None,
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        foot_lock: FootLockConfig | None = None,
        self_collision: SelfCollisionConfig | None = None,
        visualize: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize OmniRetarget and TERRA's instance-owned extension state."""
        self._initialize_terra_state()
        super().__init__(
            task_constants=task_constants,
            object_urdf_path=object_urdf_path,
            q_a_init_idx=q_a_init_idx,
            activate_foot_sticking=activate_foot_sticking,
            activate_obj_non_penetration=activate_obj_non_penetration,
            activate_joint_limits=activate_joint_limits,
            step_size=step_size,
            collision_detection_threshold=collision_detection_threshold,
            penetration_tolerance=penetration_tolerance,
            foot_sticking_tolerance=foot_sticking_tolerance,
            foot_lock=foot_lock,
            self_collision=self_collision,
            visualize=visualize,
            debug=debug,
        )

    def _initialize_terra_state(self) -> None:
        """Create inert, independent state for every optional TERRA contribution."""
        self._constraints = RetargeterConstraintState()
        self._current_frame = 0
        # OmniRetarget reads this private cache inside its inherited object-collision
        # path. Explicit TERRA environment rows keep their cache in ``_constraints``.
        self._geom_names: list[str] = []
        self._geom_names_model_cache = None

        self._solver_backend = "legacy"
        self._native_fallback_count = 0
        self._native_fallback_frames: set[int] = set()
        self._native_fallback_reasons: tuple[str, ...] = ()

    def _bind_solve_arguments(self, args, kwargs):
        """Bind arguments to the OmniRetarget solve signature.

        Args:
            args: Positional arguments for ``solve_single_iteration``.
            kwargs: Keyword arguments for ``solve_single_iteration``.

        Returns:
            Bound arguments with defaults applied.
        """
        signature = getattr(self, "_solve_iteration_signature", None)
        if signature is None:
            signature = inspect.signature(super().solve_single_iteration)
            self._solve_iteration_signature = signature
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return bound

    def _ensure_forward(self, q: np.ndarray) -> None:
        """Ensure MuJoCo data is forwarded for the requested pose.

        Args:
            q: Full generalized-position vector.
        """
        q = np.asarray(q, dtype=float)
        cached = getattr(self, "_forward_q_cache", None)
        same_model = getattr(self, "_forward_model_cache", None) is self.robot_model
        same_data = getattr(self, "_forward_data_cache", None) is self.robot_data
        data_matches = self.robot_data.qpos.shape == q.shape and np.array_equal(self.robot_data.qpos, q)
        if same_model and same_data and cached is not None and np.array_equal(cached, q) and data_matches:
            return
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        self._forward_q_cache = np.array(q, dtype=float, copy=True)
        self._forward_model_cache = self.robot_model
        self._forward_data_cache = self.robot_data
        self._qdot_to_qvel_cache = None

    def _build_transform_qdot_to_qvel_fast(self):
        """Build or reuse the current position-to-velocity transform.

        Returns:
            OmniRetarget transform from generalized-position rates to velocities.
        """
        cached = getattr(self, "_qdot_to_qvel_cache", None)
        cached_q = getattr(self, "_qdot_to_qvel_q_cache", None)
        q = self.robot_data.qpos
        same_model = getattr(self, "_qdot_to_qvel_model_cache", None) is self.robot_model
        same_data = getattr(self, "_qdot_to_qvel_data_cache", None) is self.robot_data
        # The contact helpers set this flag only after `_ensure_forward(q)` has validated
        # qpos and invalidated the transform on change. Their dozens of point Jacobians can
        # therefore use the cached transform directly; direct callers retain the defensive
        # full-array comparison below.
        if getattr(self, "_contact_forward_is_current", False) and same_model and same_data and cached is not None:
            return cached
        if same_model and same_data and cached is not None and cached_q is not None and np.array_equal(cached_q, q):
            return cached
        transform = super()._build_transform_qdot_to_qvel_fast()
        self._qdot_to_qvel_cache = transform
        self._qdot_to_qvel_q_cache = np.array(q, dtype=float, copy=True)
        self._qdot_to_qvel_model_cache = self.robot_model
        self._qdot_to_qvel_data_cache = self.robot_data
        return transform

    def _interaction_laplacian_operators(self, vertices: np.ndarray, adj_list):
        """Build or reuse interaction-mesh Laplacian operators.

        Args:
            vertices: Interaction-mesh vertices with shape ``(N, 3)``.
            adj_list: Neighbor indices for each vertex.

        Returns:
            The scalar Laplacian and its three-dimensional Kronecker product.
        """
        from holosoma_retargeting.src import interaction_mesh_retargeter as _imr

        # One OmniRetarget ``iterate`` call holds one immutable adjacency object while its SQP
        # loop asks for these operators repeatedly. The enclosing scope below establishes
        # that identity guarantee. Direct callers and the first solve of every frame still
        # take the complete content-qualified path, including in-place mutation detection.
        if (
            getattr(self, "_laplacian_topology_scope_active", False)
            and getattr(self, "_laplacian_scoped_adjacency", None) is adj_list
            and getattr(self, "_laplacian_scoped_vertex_count", None) == len(vertices)
        ):
            return self._laplacian_operator_cache, self._laplacian_kron_cache

        topology = (
            len(vertices),
            tuple(tuple(int(neighbor) for neighbor in neighbors) for neighbors in adj_list),
        )
        if getattr(self, "_laplacian_topology_cache", None) != topology:
            laplacian = _imr.calculate_laplacian_matrix(vertices, adj_list)
            if not sp.issparse(laplacian):
                laplacian = sp.csr_matrix(laplacian)
            kron = sp.kron(laplacian, sp.eye(3, format="csr"), format="csr")
            self._laplacian_topology_cache = topology
            self._laplacian_operator_cache = laplacian
            self._laplacian_kron_cache = kron
        if getattr(self, "_laplacian_topology_scope_active", False):
            self._laplacian_scoped_adjacency = adj_list
            self._laplacian_scoped_vertex_count = len(vertices)
        return self._laplacian_operator_cache, self._laplacian_kron_cache

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str],
        obj_frame: bool = False,
        point_offsets: np.ndarray | None = None,
    ):
        """Calculate body-point Jacobians and positions in one forward pass.

        Args:
            q: Full generalized-position vector.
            links: Labels mapped to robot body names.
            obj_frame: Whether to express results in the object frame.
            point_offsets: Optional body-local point offset.

        Returns:
            Jacobians, positions, and optional object-pose data.
        """
        from scipy.spatial.transform import Rotation

        jacobians = {}
        positions = {}
        if obj_frame:
            if self.has_dynamic_object:
                obj_quat = q[-4:]
                obj_pos = q[-7:-4]
                obj_rot = Rotation.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]]).as_matrix()
                obj_rot_inv = obj_rot.T
            else:
                obj_rot = np.eye(3)
                obj_rot_inv = obj_rot
                obj_pos = np.zeros(3)

        self._ensure_forward(q)
        previous = getattr(self, "_contact_forward_is_current", False)
        self._contact_forward_is_current = True
        try:
            for name, link_name in links.items():
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
                point = np.zeros(3) if point_offsets is None else point_offsets
                jacobian = self._calc_contact_jacobian_from_point(body_id, point)
                position_world = self.robot_data.xpos[body_id]
                if obj_frame:
                    position = obj_rot_inv @ (position_world - obj_pos)
                    jacobian = obj_rot_inv @ jacobian
                else:
                    position = position_world
                jacobians[name] = np.array(jacobian[:, self.q_a_indices], dtype=float, copy=True)
                positions[name] = np.array(position, dtype=float, copy=True)
        finally:
            self._contact_forward_is_current = previous

        object_pose = {"position": obj_pos, "rotation": obj_rot} if obj_frame else None
        return jacobians, positions, object_pose

    def _compute_jacobian_for_contact_relative(self, *args, **kwargs):
        """Calculate a relative-contact Jacobian using the current forward pass.

        Args:
            *args: Positional arguments accepted by the OmniRetarget method.
            **kwargs: Keyword arguments accepted by the OmniRetarget method.

        Returns:
            The OmniRetarget relative-contact Jacobian result.
        """
        previous = getattr(self, "_contact_forward_is_current", False)
        self._contact_forward_is_current = True
        try:
            return super()._compute_jacobian_for_contact_relative(*args, **kwargs)
        finally:
            self._contact_forward_is_current = previous

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world: bool = False):
        """Calculate a body-point Jacobian using current MuJoCo data.

        Args:
            body_idx: MuJoCo body ID.
            p_body: Point in body-local or world coordinates.
            input_world: Whether ``p_body`` is already in world coordinates.

        Returns:
            Translational point Jacobian with respect to generalized positions.
        """
        if not getattr(self, "_contact_forward_is_current", False):
            return super()._calc_contact_jacobian_from_point(body_idx, p_body, input_world=input_world)

        p_body = np.asarray(p_body, dtype=float).reshape(3)
        rotation = self.robot_data.xmat[body_idx].reshape(3, 3)
        body_position = self.robot_data.xpos[body_idx]
        point_world = (p_body if input_world else body_position + rotation @ p_body).astype(np.float64).reshape(3, 1)
        expected_shape = (3, self.robot_model.nv)
        jac_pos = getattr(self, "_contact_jacobian_position_scratch", None)
        if jac_pos is None or jac_pos.shape != expected_shape:
            jac_pos = np.zeros(expected_shape, dtype=np.float64, order="C")
            self._contact_jacobian_position_scratch = jac_pos
        mujoco.mj_jac(
            self.robot_model,
            self.robot_data,
            jac_pos,
            None,
            point_world,
            int(body_idx),
        )
        transform = self._build_transform_qdot_to_qvel_fast()
        return jac_pos @ transform

    def iterate(self, *args, **kwargs):
        """Run one OmniRetarget SQP frame with isolated mesh topology state.

        Args:
            *args: Positional arguments accepted by the OmniRetarget method.
            **kwargs: Keyword arguments accepted by the OmniRetarget method.

        Returns:
            The OmniRetarget iteration result.
        """
        previous_scope = getattr(self, "_laplacian_topology_scope_active", False)
        previous_adjacency = getattr(self, "_laplacian_scoped_adjacency", None)
        previous_vertex_count = getattr(self, "_laplacian_scoped_vertex_count", None)
        self._laplacian_topology_scope_active = True
        self._laplacian_scoped_adjacency = None
        self._laplacian_scoped_vertex_count = None
        try:
            return super().iterate(*args, **kwargs)
        finally:
            self._laplacian_topology_scope_active = previous_scope
            self._laplacian_scoped_adjacency = previous_adjacency
            self._laplacian_scoped_vertex_count = previous_vertex_count

    def retarget_motion(self, *args, **kwargs):
        """Retarget a motion with optimized mesh-preparation helpers.

        Args:
            *args: Positional arguments accepted by the OmniRetarget method.
            **kwargs: Keyword arguments accepted by the OmniRetarget method.

        Returns:
            The OmniRetarget motion-retargeting result.
        """
        from holosoma_retargeting.src import interaction_mesh_retargeter as _imr

        omniretarget_adjacency = _imr.get_adjacency_list
        omniretarget_calculate = _imr.calculate_laplacian_coordinates
        _imr.get_adjacency_list = _get_adjacency_list
        _imr.calculate_laplacian_coordinates = _calculate_laplacian_coordinates
        try:
            return super().retarget_motion(*args, **kwargs)
        finally:
            if _imr.calculate_laplacian_coordinates is _calculate_laplacian_coordinates:
                _imr.calculate_laplacian_coordinates = omniretarget_calculate
            if _imr.get_adjacency_list is _get_adjacency_list:
                _imr.get_adjacency_list = omniretarget_adjacency

    def attach_environment_geoms(
        self, geom_ids, max_recovery_per_iter: float = 0.01, engage_from_frame: int = 0
    ) -> None:
        """Configure non-penetration against explicit environment geoms.

        Args:
            geom_ids: Static environment geom IDs.
            max_recovery_per_iter: Maximum penetration recovery per SQP step, in meters.
            engage_from_frame: First solver frame on which to enforce the constraint.
        """
        self._constraints.environment.attach(geom_ids, max_recovery_per_iter, engage_from_frame)

    def attach_self_collision(
        self,
        body_pairs: Sequence[tuple[str, str]],
        tolerance: float = 0.002,
        weight: float = 5000.0,
        max_recovery_per_iter: float | None = None,
        engage_from_frame: int = 0,
    ) -> int:
        """Configure a collision cost between selected robot body pairs.

        Args:
            body_pairs: Body-name pairs to keep separated.
            tolerance: Required geom separation in meters.
            weight: Quadratic weight on separation shortfall.
            max_recovery_per_iter: Optional shortfall cap per SQP step, in meters.
            engage_from_frame: First solver frame on which to apply the cost.

        Returns:
            Number of geom pairs monitored by the cost.

        Raises:
            ValueError: If a body has no collision geoms or the recovery cap is invalid.
        """
        return self._constraints.objective.self_collision.attach(
            self,
            body_pairs,
            tolerance,
            weight,
            max_recovery_per_iter,
            engage_from_frame,
        )

    def _self_collision_terms(self, q: np.ndarray) -> list:
        """Linearize active self-collision residuals at a pose.

        Args:
            q: Full generalized-position vector for the current SQP iterate.

        Returns:
            Weight, Jacobian row, and capped shortfall for each active geom pair.
        """
        return self._constraints.objective.self_collision.linearize(self, q)

    def attach_foot_clearance(
        self,
        geoms_by_side: dict[str, Sequence[int]],
        targets: dict[str, np.ndarray],
        weight: float = 500.0,
        max_recovery_per_iter: float = 0.01,
        activation_by_side: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Configure absolute sole-height costs for swing-foot clearance.

        Args:
            geoms_by_side: Sole geom IDs keyed by side.
            targets: Per-side absolute heights in meters. Inactive frames contain
                negative infinity.
            weight: Quadratic weight on clearance shortfall.
            max_recovery_per_iter: Maximum shortfall recovery per SQP step, in meters.
            activation_by_side: Optional per-side authority arrays in ``[0, 1]``.

        Raises:
            ValueError: If inputs are invalid or the model has no ``"floor"`` geom.
        """
        self._constraints.objective.clearance.attach(
            self,
            geoms_by_side,
            targets,
            weight,
            max_recovery_per_iter,
            activation_by_side,
        )

    def _clearance_terms(self, q: np.ndarray) -> list:
        """Linearize active sole-height shortfalls at a pose.

        Args:
            q: Full generalized-position vector for the current SQP iterate.

        Returns:
            Weight, vertical Jacobian row, and capped shortfall for each active sole geom.
        """
        return self._constraints.objective.clearance.linearize(self, q)

    def _clearance_target(self, geom_id: int) -> float:
        """Get a sole geom's target height for the current frame.

        Args:
            geom_id: Sole geom ID.

        Returns:
            Absolute target height, or negative infinity when inactive.
        """
        return self._constraints.objective.clearance.target(geom_id, self._current_frame)

    def attach_foot_route(
        self,
        targets_by_body: dict[str, np.ndarray],
        weight: float,
        activation_by_body: dict[str, np.ndarray] | None = None,
        max_recovery_per_iter: float | None = None,
    ) -> None:
        """Configure direct vertical targets for routed foot landmarks.

        Args:
            targets_by_body: Absolute height arrays keyed by robot body name.
            weight: Quadratic weight on vertical target residuals.
            activation_by_body: Optional per-body authority arrays in ``[0, 1]``.
            max_recovery_per_iter: Optional residual cap per SQP step, in meters.

        Raises:
            ValueError: If a body, recovery cap, or activation array is invalid.
        """
        self._constraints.objective.route.attach(
            self,
            targets_by_body,
            weight,
            activation_by_body,
            max_recovery_per_iter,
        )

    def attach_foot_stance_height(
        self,
        targets_by_body: dict[str, np.ndarray],
        activation_by_body: dict[str, np.ndarray],
        weight: float,
        max_recovery_per_iter: float,
    ) -> None:
        """Configure direct vertical foot-body targets during annotated stance."""
        self._constraints.objective.stance_height.attach(
            self,
            targets_by_body,
            activation_by_body,
            weight,
            max_recovery_per_iter,
        )

    def _foot_stance_height_terms(self, q: np.ndarray) -> list:
        """Linearize active vertical foot-body targets during stance."""
        return self._constraints.objective.stance_height.linearize(self, q)

    def attach_seat_contact(
        self,
        glute_geom_names: Sequence[str],
        activation_by_seat: dict[str, np.ndarray],
        *,
        weight: float,
        clearance: float,
        max_recovery_per_iter: float,
    ) -> None:
        """Configure exact glute-to-seat distance targets during seated rests.

        The target is expressed between collision geometry rather than through a fixed
        pelvis-origin offset.  It therefore calibrates the morphology used by the actual
        retargeted robot while leaving the motion-derived chair surface unchanged.
        """
        self._constraints.objective.seat_contact.attach(
            self,
            glute_geom_names,
            activation_by_seat,
            weight,
            clearance,
            max_recovery_per_iter,
        )

    def _seat_contact_terms(self, q: np.ndarray) -> list:
        """Linearize the nearest active glute/seat signed-distance target."""
        return self._constraints.objective.seat_contact.linearize(self, q)

    def _foot_route_terms(self, q: np.ndarray) -> list:
        """Linearize active foot-route targets at a pose.

        Args:
            q: Full generalized-position vector for the current SQP iterate.

        Returns:
            Weight, vertical Jacobian row, and capped residual for each active body.
        """
        return self._constraints.objective.route.linearize(self, q)

    def _prefilter_pairs_with_mj_collision(self, threshold: float):
        """Find geom pairs within a collision-detection threshold.

        Args:
            threshold: Temporary MuJoCo collision margin in meters.

        Returns:
            Candidate geom-ID pairs in ascending order within each pair.
        """
        return self._constraints.environment.prefilter_pairs(self, threshold)

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        """Build environment non-penetration Jacobians and signed distances.

        Args:
            q: Full generalized-position vector for the current SQP iterate.

        Returns:
            Contact Jacobians and capped signed distances keyed by geom pair.
        """
        # Suppression must precede collision-backend dispatch.  The clean OmniRetarget
        # profile binds fitted terrain through the inherited ``object_name`` path and
        # therefore leaves explicit environment geometry IDs unset. Checking only the
        # TERRA-owned path made its infeasibility retry assemble the identical QP twice.
        environment = self._constraints.environment
        if environment.suppressed:
            return {}, {}
        if environment.geom_ids is None:
            if self._geom_names_model_cache is not self.robot_model or len(self._geom_names) != self.robot_model.ngeom:
                self._geom_names = [
                    mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                    for geom_id in range(self.robot_model.ngeom)
                ]
                self._geom_names_model_cache = self.robot_model
            return super()._update_jacobians_and_phis_from_q(q)
        return environment.linearize(self, q)

    def attach_orientation_targets(
        self,
        site_ids: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        names: list[str],
    ) -> None:
        """Configure world-space orientation costs for mimic sites.

        Args:
            site_ids: Site IDs with shape ``(K,)``.
            targets: World rotations with shape ``(T, K, 3, 3)``.
            weights: Per-site weights with shape ``(K,)``.
            names: Ordered site names.

        Raises:
            ValueError: If targets and site IDs contain different site counts.
        """
        self._constraints.objective.orientation.attach(site_ids, targets, weights, names)

    def _orientation_linearisation(self, q: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Linearize site-orientation residuals at a pose.

        Args:
            q: Full generalized-position vector for the current SQP iterate.
            frame_idx: Solver frame used to select targets.

        Returns:
            Rotation-vector residuals with shape ``(3K,)`` and their Jacobian
            with shape ``(3K, nq_a)``.
        """
        return self._constraints.objective.orientation.linearize(self, q, frame_idx)

    def attach_joint_couplers(self, couplers: list, weight: float) -> None:
        """Configure polynomial joint-coupler consistency costs.

        Args:
            couplers: Dependent index, independent index, and polynomial tuples.
            weight: Quadratic weight on coupler residuals.
        """
        self._constraints.objective.coupler.attach(self, couplers, weight)

    def _coupler_linearisation(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearize polynomial joint-coupler residuals at a pose.

        Args:
            q: Full generalized-position vector for the current SQP iterate.

        Returns:
            Coupler Jacobian and residual arrays.
        """
        return self._constraints.objective.coupler.linearize(self, q)

    def attach_foot_anchoring(
        self,
        contact: dict,
        weight: float,
        ramp_frames: int,
        preserve_clipped_boundaries: bool = False,
        velocity_contact: dict | None = None,
        velocity_sites: dict | None = None,
        velocity_weight: float = 0.0,
        velocity_tracking_weight: float = 0.0,
        velocity_tolerance_m: float = 0.0,
    ) -> None:
        """Configure stance-foot anchoring and horizontal velocity costs.

        Args:
            contact: Per-foot Boolean contact arrays.
            weight: Quadratic weight on absolute anchor residuals.
            ramp_frames: Frames used to taper anchor engagement.
            preserve_clipped_boundaries: Do not taper a stance at a recording
                boundary where no touchdown or liftoff was observed.
            velocity_contact: Optional per-foot sticking arrays for velocity control.
            velocity_sites: Optional foot labels mapped to speed-control sites.
            velocity_weight: Weight on displacement beyond the tolerance.
            velocity_tracking_weight: Weight on total frame-relative displacement.
            velocity_tolerance_m: Unpenalized per-frame displacement in meters.

        Raises:
            ValueError: If a weight, tolerance, body, site, or label mapping is invalid.
        """
        self._constraints.objective.foot_anchor.attach(
            self,
            contact,
            weight,
            ramp_frames,
            preserve_clipped_boundaries,
            velocity_contact,
            velocity_sites,
            velocity_weight,
            velocity_tracking_weight,
            velocity_tolerance_m,
        )

    def _foot_kinematics(self, q: np.ndarray) -> tuple[dict, dict, dict, dict]:
        """Calculate foot-anchor and speed-site kinematics at a pose.

        Args:
            q: Full generalized-position vector.

        Returns:
            Anchor Jacobians, anchor positions, velocity Jacobians, and velocity
            positions keyed by foot label.
        """
        return self._constraints.objective.foot_anchor.kinematics(self, q)

    def _foot_velocity_kinematics(self, q: np.ndarray) -> tuple[dict, dict]:
        """Calculate horizontal kinematics at foot speed-control points.

        Args:
            q: Full generalized-position vector.

        Returns:
            Horizontal Jacobians and positions keyed by foot label.
        """
        _, _, jacobians, positions = self._foot_kinematics(q)
        return jacobians, positions

    def _foot_anchor_terms(self, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int) -> list:
        """Linearize foot-anchor and velocity residuals at a pose.

        Args:
            q: Full generalized-position vector for the current SQP iterate.
            q_t_last: Converged generalized positions from the previous frame.
            frame_idx: Solver frame.

        Returns:
            Weight, horizontal Jacobian, and residual tuples for active foot costs.
        """
        return self._constraints.objective.foot_anchor.linearize(self, q, q_t_last, frame_idx)

    def _has_attached_quadratic_terms(self) -> bool:
        """Return whether any optional TERRA objective term is configured."""
        return bool(self._constraints.objective.active_names())

    def active_term_names(self) -> tuple[str, ...]:
        """Return configured optional objective terms in assembly order."""
        return self._constraints.objective.active_names()

    @property
    def nonpenetration_relaxed_frames(self) -> frozenset[int]:
        """Frames whose infeasible QP required one non-penetration-free retry."""
        return frozenset(self._constraints.environment.relaxed_frames)

    @property
    def route_recovery_cap(self) -> float:
        """Maximum signed foot-route correction requested in one SQP step."""
        return self._constraints.objective.route.max_recovery_per_iter

    def _attached_quadratic_terms(
        self,
        q: np.ndarray,
        q_t_last: np.ndarray,
        frame_idx: int,
    ) -> list[_QuadraticTerm]:
        """Linearize every configured TERRA objective at one SQP iterate."""
        return self._constraints.objective.quadratic_terms(self, q, q_t_last, frame_idx)

    def _solve_legacy_with_quadratic_terms(
        self,
        terms: Sequence[_QuadraticTerm],
        *args,
        **kwargs,
    ):
        """Inject attached terms into OmniRetarget's legacy CVXPY formulation."""
        import cvxpy as cp
        from holosoma_retargeting.src import interaction_mesh_retargeter as _imr

        def append_terms(objective):
            return _objective_with_quadratic_terms(cp, objective, terms)

        original = _imr.cp
        _imr.cp = _CvxpyShim(cp, append_terms)
        try:
            return self._solve_or_relax_nonpen(*args, **kwargs)
        finally:
            _imr.cp = original

    def solve_single_iteration(self, *args, **kwargs):
        """Solve one SQP iteration with all attached TERRA costs.

        Args:
            *args: Positional arguments accepted by the OmniRetarget solve method.
            **kwargs: Keyword arguments accepted by the OmniRetarget solve method.

        Returns:
            The OmniRetarget-compatible solve result.

        Raises:
            RuntimeError: If the OmniRetarget objective cannot be intercepted safely.
            ValueError: If the selected solver backend is invalid.
        """
        # Record the frame before any early return: `_update_jacobians_and_phis_from_q` is
        # called from inside OmniRetarget's solve and has no other way to know which frame it is on.
        bound = self._bind_solve_arguments(args, kwargs)
        self._current_frame = int(bound.arguments["frame_idx"])

        if self._solver_backend not in {"legacy", "condensed_cvxpy", "native_clarabel"}:
            raise ValueError(
                "solver_backend must be 'legacy', 'condensed_cvxpy', or 'native_clarabel', "
                f"got {self._solver_backend!r}"
            )
        if self._solver_backend == "legacy" and not self._has_attached_quadratic_terms():
            return self._solve_or_relax_nonpen(*args, **kwargs)

        q = np.copy(bound.arguments["q_locked"])
        q[self.q_a_indices] = bound.arguments["q_a_n_last"]
        frame_idx = int(bound.arguments["frame_idx"])
        terms = self._attached_quadratic_terms(q, bound.arguments["q_t_last"], frame_idx)

        if self._solver_backend in {"condensed_cvxpy", "native_clarabel"}:
            return self._solve_condensed_or_relax_nonpen(bound, terms)
        return self._solve_legacy_with_quadratic_terms(terms, *args, **kwargs)

    def _condensed_laplacian_linearization(
        self,
        q: np.ndarray,
        object_points: np.ndarray,
        adjacency,
        target_laplacian: np.ndarray,
    ) -> _LaplacianLinearization:
        """Linearize interaction-mesh Laplacian matching at the current pose."""
        jacobians, positions, _ = self._calc_manipulator_jacobians(
            q,
            links=self.laplacian_match_links,
            obj_frame=(self.object_name != "ground"),
        )
        robot_links = list(self.laplacian_match_links)
        n_vertices = len(robot_links) + len(object_points)
        vertex_jacobian = np.zeros((3 * n_vertices, self.nq_a))
        for index, link in enumerate(robot_links):
            vertex_jacobian[3 * index : 3 * (index + 1), :] = jacobians[link]

        robot_points = np.array([positions[link] for link in robot_links])
        vertices = np.vstack([robot_points, object_points])
        laplacian, repeated_laplacian = self._interaction_laplacian_operators(vertices, adjacency)
        current = (laplacian @ vertices).reshape(-1)
        weights = (self.laplacian_weights * np.ones(n_vertices)).astype(float)
        return _LaplacianLinearization(
            jacobian=(repeated_laplacian @ vertex_jacobian)[:, self.q_a_indices],
            current=current,
            target=target_laplacian.reshape(-1),
            row_scale=np.sqrt(np.repeat(weights, 3)),
        )

    @staticmethod
    def _foot_sticking_side_keys(foot_sticking: dict) -> tuple[str, str]:
        """Resolve the left and right keys expected by OmniRetarget contact flags."""
        left_key = right_key = None
        for key in foot_sticking:
            if key.lower().startswith("l"):
                left_key = key
            elif key.lower().startswith("r"):
                right_key = key
        if left_key is None or right_key is None:
            raise ValueError("foot_sticking must include one left* and one right* key")
        return left_key, right_key

    def _add_condensed_foot_constraints(
        self,
        constraints: _InequalityConstraints,
        q: np.ndarray,
        q_t_last: np.ndarray,
        foot_sticking: dict,
        frame_idx: int,
    ) -> None:
        """Add optional horizontal sticking and vertical foot-lock constraints."""
        apply_sticking = (self.q_a_init_idx < 12) and self.activate_foot_sticking
        apply_lock = (self.q_a_init_idx < 12) and self.foot_lock.enable
        if not (apply_sticking or apply_lock):
            return

        foot_jacobians, foot_positions, _ = self._calc_manipulator_jacobians(
            q,
            links=self.foot_links,
            obj_frame=False,
        )
        if apply_sticking:
            _, previous_positions, _ = self._calc_manipulator_jacobians(
                q_t_last,
                links=self.foot_links,
                obj_frame=False,
            )
            left_key, right_key = self._foot_sticking_side_keys(foot_sticking)
            for key, jacobian in foot_jacobians.items():
                active = (("left" in key) and foot_sticking[left_key]) or (
                    ("right" in key) and foot_sticking[right_key]
                )
                if active:
                    lower = previous_positions[key] - foot_positions[key] - self.foot_sticking_tolerance
                    upper = lower + 2 * self.foot_sticking_tolerance
                    constraints.add_two_sided(jacobian[:2, self.q_a_indices], lower[:2], upper[:2])

        if apply_lock:
            for key, jacobian in foot_jacobians.items():
                anchor = self._is_foot_locked_in_window(key, frame_idx)
                if anchor is None:
                    continue
                delta = anchor - foot_positions[key][2]
                tolerance = self.foot_lock.tolerance
                constraints.add_two_sided(
                    jacobian[2, self.q_a_indices],
                    delta - tolerance,
                    delta + tolerance,
                )

    def _condensed_constraints(
        self,
        q: np.ndarray,
        q_t_last: np.ndarray,
        q_a_n_last: np.ndarray,
        foot_sticking: dict,
        frame_idx: int,
    ) -> _InequalityConstraints:
        """Assemble ordered hard constraints for both condensed backends."""
        constraints = _InequalityConstraints(self.nq_a)
        self._add_condensed_foot_constraints(constraints, q, q_t_last, foot_sticking, frame_idx)

        jacobians, distances = self._update_jacobians_and_phis_from_q(q)
        for key, distance in distances.items():
            jacobian = jacobians[key][self.q_a_indices]
            constraints.add_lower_bound(jacobian, -distance - self.penetration_tolerance)

        self_jacobians, self_distances = self._compute_self_collision_constraints(frame_idx)
        for key, distance in self_distances.items():
            jacobian = self_jacobians[key][self.q_a_indices]
            constraints.add_lower_bound(jacobian, self._self_collision_tolerance - distance)

        if self.activate_joint_limits:
            constraints.add_variable_bounds(
                self.q_a_lb - q_a_n_last,
                self.q_a_ub - q_a_n_last,
            )
        return constraints

    def _trust_radius(self, initial_iteration: bool) -> float:
        """Return the initial-pose or ordinary SQP trust radius."""
        return float(getattr(self, "_initial_step_size", self.step_size)) if initial_iteration else self.step_size

    def _native_condensed_objective(
        self,
        laplacian: _LaplacianLinearization,
        q_a_n_last: np.ndarray,
        dqa_smooth: np.ndarray,
        nominal_weight: float,
        q_a_nominal: np.ndarray | None,
        attached_terms: Sequence[_QuadraticTerm],
    ) -> tuple[list[tuple], list[tuple]]:
        """Assemble native least-squares and centered-quadratic terms."""
        return native_condensed_objective(
            nq_a=self.nq_a,
            track_nominal_indices=self.track_nominal_indices,
            q_diag=self.Q_diag,
            smooth_weight=self.smooth_weight,
            laplacian=laplacian,
            q_a_n_last=q_a_n_last,
            dqa_smooth=dqa_smooth,
            nominal_weight=nominal_weight,
            q_a_nominal=q_a_nominal,
            attached_terms=attached_terms,
        )

    def _updated_pose(self, q: np.ndarray, q_a_n_last: np.ndarray, step: np.ndarray) -> np.ndarray:
        """Apply an optimizer step and renormalize the floating-root quaternion."""
        return updated_pose(q, self.q_a_indices, q_a_n_last, step)

    def _record_native_fallback(self, frame_idx: int, error: RuntimeError | None) -> None:
        """Record one native failure before using the CVXPY oracle."""
        self._native_fallback_count = int(self._native_fallback_count) + 1
        self._native_fallback_frames = set(self._native_fallback_frames) | {frame_idx}
        reason = f"frame {frame_idx}: {error}"
        self._native_fallback_reasons = (*self._native_fallback_reasons, reason)

    def _try_native_condensed_solve(
        self,
        q: np.ndarray,
        q_a_n_last: np.ndarray,
        dqa_smooth: np.ndarray,
        laplacian: _LaplacianLinearization,
        constraints: _InequalityConstraints,
        attached_terms: Sequence[_QuadraticTerm],
        arguments: dict,
        trust_radius: float,
    ):
        """Run native Clarabel, returning ``None`` after a recorded failure."""
        least_squares, centered_quadratics = self._native_condensed_objective(
            laplacian,
            q_a_n_last,
            dqa_smooth,
            arguments["w_nominal_tracking"],
            arguments["q_a_nominal"],
            attached_terms,
        )
        native_result = None
        native_error = None
        try:
            native_result = _native_clarabel_qp(
                self.nq_a,
                least_squares,
                constraints.native(),
                trust_radius,
                centered_quadratics=centered_quadratics,
                verbose=arguments["verbose"],
            )
        except RuntimeError as exc:
            native_error = exc
        if native_result is not None:
            step, cost, _ = native_result
            return self._updated_pose(q, q_a_n_last, step), cost

        self._record_native_fallback(int(arguments["frame_idx"]), native_error)
        return None

    def _cvxpy_condensed_objective(
        self,
        cp,
        step,
        laplacian: _LaplacianLinearization,
        q_a_n_last: np.ndarray,
        dqa_smooth: np.ndarray,
        nominal_weight: float,
        q_a_nominal: np.ndarray | None,
        attached_terms: Sequence[_QuadraticTerm],
    ) -> list:
        """Build the condensed CVXPY objective in its established term order."""
        return cvxpy_condensed_objective(
            cp,
            step,
            track_nominal_indices=self.track_nominal_indices,
            q_diag=self.Q_diag,
            smooth_weight=self.smooth_weight,
            laplacian=laplacian,
            q_a_n_last=q_a_n_last,
            dqa_smooth=dqa_smooth,
            nominal_weight=nominal_weight,
            q_a_nominal=q_a_nominal,
            attached_terms=attached_terms,
        )

    def _solve_cvxpy_condensed(
        self,
        q: np.ndarray,
        q_a_n_last: np.ndarray,
        dqa_smooth: np.ndarray,
        laplacian: _LaplacianLinearization,
        constraints: _InequalityConstraints,
        attached_terms: Sequence[_QuadraticTerm],
        arguments: dict,
        trust_radius: float,
    ):
        """Solve the condensed CVXPY formulation and validate its optimizer step."""
        import cvxpy as cp

        step_variable = cp.Variable(len(self.q_a_indices), name="dqa")
        cvxpy_constraints = constraints.cvxpy(step_variable)
        cvxpy_constraints.append(cp.SOC(trust_radius, step_variable))
        objective_terms = self._cvxpy_condensed_objective(
            cp,
            step_variable,
            laplacian,
            q_a_n_last,
            dqa_smooth,
            arguments["w_nominal_tracking"],
            arguments["q_a_nominal"],
            attached_terms,
        )
        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), cvxpy_constraints)
        problem.solve(solver=cp.CLARABEL, verbose=arguments["verbose"])
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"CVXPY solve failed: {problem.status}")

        step = np.asarray(step_variable.value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(step)):
            raise RuntimeError("CVXPY solve returned a non-finite optimizer step")
        step_norm = float(np.linalg.norm(step))
        if step_norm > trust_radius + 1e-6:
            raise RuntimeError(f"CVXPY solve escaped the SQP trust region: {step_norm:.6g} > {trust_radius:.6g}")
        return self._updated_pose(q, q_a_n_last, step), problem.value

    def _solve_condensed_iteration(self, bound, terms: Sequence[_QuadraticTerm]):
        """Solve an SQP subproblem with the Laplacian variable eliminated.

        Args:
            bound: Bound OmniRetarget solve arguments with defaults applied.
            terms: Attached least-squares terms shared by both solver backends.

        Returns:
            Updated generalized positions and the subproblem objective value.

        Raises:
            RuntimeError: If neither the native nor CVXPY formulation solves.
            ValueError: If an assembled constraint has an invalid shape.
        """
        arguments = bound.arguments
        q_a_n_last = arguments["q_a_n_last"]
        assert len(q_a_n_last) == self.nq_a
        q = np.copy(arguments["q_locked"])
        q[self.q_a_indices] = q_a_n_last

        laplacian = self._condensed_laplacian_linearization(
            q,
            arguments["obj_pts_local"],
            arguments["adj_list"],
            arguments["target_laplacian"],
        )
        constraints = self._condensed_constraints(
            q,
            arguments["q_t_last"],
            q_a_n_last,
            arguments["foot_sticking"],
            int(arguments["frame_idx"]),
        )
        dqa_smooth = arguments["q_t_last"][self.q_a_indices] - q_a_n_last
        trust_radius = self._trust_radius(bool(arguments["init_t"]))
        if self._solver_backend == "native_clarabel":
            native_result = self._try_native_condensed_solve(
                q,
                q_a_n_last,
                dqa_smooth,
                laplacian,
                constraints,
                terms,
                arguments,
                trust_radius,
            )
            if native_result is not None:
                return native_result

        # The CVXPY expression tree is built only for the condensed backend or after a
        # certified native failure; successful native iterations remain CVXPY-free.
        return self._solve_cvxpy_condensed(
            q,
            q_a_n_last,
            dqa_smooth,
            laplacian,
            constraints,
            terms,
            arguments,
            trust_radius,
        )

    def _solve_condensed_or_relax_nonpen(self, bound, terms: Sequence[_QuadraticTerm]):
        """Run the condensed formulation with non-penetration recovery.

        Args:
            bound: Bound OmniRetarget solve arguments.
            terms: Attached least-squares terms shared by both solver backends.

        Returns:
            Updated generalized positions and objective value.
        """
        return self._run_or_relax_nonpen(lambda: self._solve_condensed_iteration(bound, terms))

    def _solve_or_relax_nonpen(self, *args, **kwargs):
        """Run an OmniRetarget SQP iteration with non-penetration recovery.

        Args:
            *args: Positional arguments accepted by the OmniRetarget solve method.
            **kwargs: Keyword arguments accepted by the OmniRetarget solve method.

        Returns:
            The OmniRetarget solve result.

        Raises:
            Exception: If failure is unrelated to non-penetration or the retry fails.
        """
        base_solve = super().solve_single_iteration
        return self._run_or_relax_nonpen(lambda: base_solve(*args, **kwargs))

    @staticmethod
    def _is_recoverable_solver_failure(exc: Exception) -> bool:
        """Return whether a solver exception represents an infeasible QP."""
        # Keep cvxpy optional at module import time, just like the rest of this class.
        import cvxpy as cp

        return (isinstance(exc, RuntimeError) and "infeasible" in str(exc).lower()) or isinstance(
            exc, cp.error.SolverError
        )

    def _run_or_relax_nonpen(self, solve_once):
        """Retry one infeasible SQP iteration without object non-penetration.

        Args:
            solve_once: Zero-argument callback that runs one QP formulation.

        Returns:
            Result from the initial solve or recovery attempt.

        Raises:
            Exception: If failure is ineligible or remains without non-penetration.
        """
        try:
            return solve_once()
        except Exception as exc:
            environment = self._constraints.environment
            if not self._is_recoverable_solver_failure(exc) or environment.suppressed:
                raise
            environment.suppressed = True
            try:
                out = solve_once()
            finally:
                environment.suppressed = False
            environment.relaxed_frames = set(environment.relaxed_frames) | {self._current_frame}
            return out
