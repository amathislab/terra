"""Typed state and linearizations for TERRA's optional retargeting terms.

The upstream OmniRetarget class owns the SQP loop and MuJoCo model.  This module
owns the configuration and mutable state of each TERRA contribution.  Keeping
those responsibilities separate makes the objective's active terms and their
assembly order explicit without changing the numerical formulas.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import mujoco
import numpy as np

from terra._qp import _QuadraticTerm, _row_quadratic_terms
from terra.contacts import contact_ramp

RowTerm = tuple[float, np.ndarray, float | np.ndarray]


class TermHost(Protocol):
    """Kinematic operations supplied by :class:`TerraRetargeter`."""

    robot_model: object
    robot_data: object
    q_a_indices: np.ndarray
    nq_a: int
    collision_detection_threshold: float
    penetration_tolerance: float
    foot_links: dict[str, str]
    _current_frame: int

    def _ensure_forward(self, q: np.ndarray) -> None: ...

    def _build_transform_qdot_to_qvel_fast(self): ...

    def _compute_jacobian_for_contact_relative(self, *args, **kwargs): ...

    def _orientation_linearisation(self, q: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]: ...

    def _foot_anchor_terms(self, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int) -> list[RowTerm]: ...

    def _self_collision_terms(self, q: np.ndarray) -> list[RowTerm]: ...

    def _clearance_terms(self, q: np.ndarray) -> list[RowTerm]: ...

    def _foot_route_terms(self, q: np.ndarray) -> list[RowTerm]: ...

    def _foot_stance_height_terms(self, q: np.ndarray) -> list[RowTerm]: ...

    def _seat_contact_terms(self, q: np.ndarray) -> list[RowTerm]: ...

    def _coupler_linearisation(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class QuadraticComponent(Protocol):
    """Common interface used to assemble every optional objective term."""

    name: ClassVar[str]

    @property
    def configured(self) -> bool: ...

    def quadratic_terms(
        self,
        host: TermHost,
        q: np.ndarray,
        q_t_last: np.ndarray,
        frame_idx: int,
    ) -> list[_QuadraticTerm]: ...


def _validate_activation(targets: dict, activation: dict | None, label: str) -> dict:
    """Validate per-frame authority arrays against their target arrays."""
    if activation is None:
        return {key: np.ones(len(target), dtype=float) for key, target in targets.items()}
    if set(activation) != set(targets):
        raise ValueError(f"{label} activation keys must match target keys: {sorted(activation)} != {sorted(targets)}")
    out = {}
    for key, target in targets.items():
        value = np.asarray(activation[key], dtype=float)
        if value.ndim != 1 or len(value) != len(target):
            raise ValueError(f"{label} activation for {key!r} must have shape ({len(target)},), got {value.shape}")
        if not np.all(np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{label} activation for {key!r} must be finite and in [0, 1]")
        out[key] = value
    return out


def _frame_activation(activation: np.ndarray | None, frame_idx: int) -> float:
    """Select one frame's authority, clamping to the available timebase."""
    if activation is None or len(activation) == 0:
        return 1.0
    index = min(max(frame_idx, 0), len(activation) - 1)
    return float(activation[index])


@dataclass(slots=True)
class EnvironmentConstraintState:
    """State for hard robot/environment non-penetration rows."""

    geom_ids: set[int] | None = None
    max_recovery_per_iter: float = 0.01
    engage_from_frame: int = 0
    suppressed: bool = False
    relaxed_frames: set[int] = field(default_factory=set)
    geom_names: list[str] = field(default_factory=list)
    geom_names_model: object | None = field(default=None, repr=False)
    saved_margins: np.ndarray | None = field(default=None, repr=False)

    def attach(
        self,
        geom_ids: Sequence[int],
        max_recovery_per_iter: float,
        engage_from_frame: int,
    ) -> None:
        """Bind the static environment geometry IDs."""
        self.geom_ids = {int(geom_id) for geom_id in geom_ids}
        self.max_recovery_per_iter = float(max_recovery_per_iter)
        self.engage_from_frame = int(engage_from_frame)

    def prefilter_pairs(self, host: TermHost, threshold: float) -> set[tuple[int, int]]:
        """Find geom pairs inside a temporary MuJoCo collision margin."""
        model, data = host.robot_model, host.robot_data
        geom_count = model.ngeom
        if self.geom_names_model is not model or len(self.geom_names) != geom_count:
            self.geom_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "" for geom_id in range(geom_count)
            ]
            self.geom_names_model = model

        if self.saved_margins is None or len(self.saved_margins) != geom_count:
            self.saved_margins = np.empty_like(model.geom_margin)
        self.saved_margins[:] = model.geom_margin
        model.geom_margin[:] = threshold
        mujoco.mj_collision(model, data)

        candidates = set()
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 < 0 or geom2 < 0:
                continue
            candidates.add((min(geom1, geom2), max(geom1, geom2)))

        model.geom_margin[:] = self.saved_margins
        return candidates

    def linearize(self, host: TermHost, q: np.ndarray) -> tuple[dict, dict]:
        """Build explicit environment Jacobians and capped signed distances."""
        if self.suppressed or host._current_frame < self.engage_from_frame:
            return {}, {}
        if self.geom_ids is None:
            raise RuntimeError("explicit environment linearization requires attached geometry IDs")

        host._ensure_forward(q)
        model, data = host.robot_model, host.robot_data
        threshold = float(host.collision_detection_threshold)
        candidates = self.prefilter_pairs(host, threshold)

        jacobians, phis = {}, {}
        fromto = np.zeros(6, dtype=float)
        contype, conaff = model.geom_contype, model.geom_conaffinity
        for geom1, geom2 in candidates:
            if (geom1 in self.geom_ids) == (geom2 in self.geom_ids):
                continue
            if contype[geom1] == 0 and conaff[geom1] == 0:
                continue
            if contype[geom2] == 0 and conaff[geom2] == 0:
                continue

            fromto[:] = 0.0
            distance = mujoco.mj_geomDistance(model, data, geom1, geom2, threshold, fromto)
            if distance <= threshold:
                jacobians[(geom1, geom2)] = host._compute_jacobian_for_contact_relative(
                    model.geom(geom1),
                    model.geom(geom2),
                    self.geom_names[geom1],
                    self.geom_names[geom2],
                    fromto,
                    distance,
                )
                floor_distance = -(host.penetration_tolerance + self.max_recovery_per_iter)
                phis[(geom1, geom2)] = float(max(distance, floor_distance))
        return jacobians, phis


@dataclass(slots=True)
class SelfCollisionTerm:
    """Soft separation term for selected robot geometry pairs."""

    name: ClassVar[str] = "self_collision"
    pairs: list[tuple[int, int]] = field(default_factory=list)
    tolerance: float = 0.0
    weight: float = 0.0
    max_recovery_per_iter: float = np.inf
    engage_from_frame: int = 0
    geom_names: list[str] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        return bool(self.pairs)

    def attach(
        self,
        host: TermHost,
        body_pairs: Sequence[tuple[str, str]],
        tolerance: float,
        weight: float,
        max_recovery_per_iter: float | None,
        engage_from_frame: int,
    ) -> int:
        """Resolve body pairs to all collidable geometry pairs."""
        model = host.robot_model
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "" for geom_id in range(model.ngeom)]
        by_body: dict[str, list[int]] = {}
        for geom_id in range(model.ngeom):
            if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
                continue
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
            by_body.setdefault(body, []).append(geom_id)

        pairs = []
        for body_a, body_b in body_pairs:
            geoms_a, geoms_b = by_body.get(body_a, []), by_body.get(body_b, [])
            missing = [name for name, geoms in ((body_a, geoms_a), (body_b, geoms_b)) if not geoms]
            if missing:
                raise ValueError(
                    f"Self-collision pair ({body_a!r}, {body_b!r}): no collision geoms on "
                    f"{', '.join(repr(name) for name in missing)}"
                )
            pairs.extend((geom_a, geom_b) for geom_a in geoms_a for geom_b in geoms_b)

        self.pairs = pairs
        self.tolerance = float(tolerance)
        self.weight = float(weight)
        if max_recovery_per_iter is None:
            self.max_recovery_per_iter = np.inf
        else:
            recovery = float(max_recovery_per_iter)
            if not np.isfinite(recovery) or recovery <= 0.0:
                raise ValueError(
                    f"self-collision max_recovery_per_iter must be positive and finite, got {max_recovery_per_iter!r}"
                )
            self.max_recovery_per_iter = recovery
        self.engage_from_frame = int(engage_from_frame)
        self.geom_names = names
        return len(pairs)

    def linearize(self, host: TermHost, q: np.ndarray) -> list[RowTerm]:
        """Linearize active separation shortfalls."""
        if not self.pairs or host._current_frame < self.engage_from_frame:
            return []
        host._ensure_forward(q)
        model, data = host.robot_model, host.robot_data
        threshold = float(host.collision_detection_threshold)
        fromto = np.zeros(6, dtype=float)
        terms = []
        for geom_a, geom_b in self.pairs:
            fromto[:] = 0.0
            distance = mujoco.mj_geomDistance(model, data, geom_a, geom_b, threshold, fromto)
            if distance >= self.tolerance:
                continue
            jacobian = host._compute_jacobian_for_contact_relative(
                model.geom(geom_a),
                model.geom(geom_b),
                self.geom_names[geom_a],
                self.geom_names[geom_b],
                fromto,
                distance,
            )
            shortfall = self.tolerance - distance
            terms.append(
                (
                    self.weight,
                    jacobian[host.q_a_indices],
                    min(shortfall, self.max_recovery_per_iter),
                )
            )
        return terms

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last, frame_idx
        return _row_quadratic_terms(host._self_collision_terms(q))


@dataclass(slots=True)
class ClearanceTerm:
    """One-sided absolute sole-height targets for swing clearance."""

    name: ClassVar[str] = "swing_clearance"
    geoms: dict[int, str] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)
    activation: dict[str, np.ndarray] = field(default_factory=dict)
    floor_id: int | None = None
    weight: float = 0.0
    max_recovery_per_iter: float = 0.01

    @property
    def configured(self) -> bool:
        return bool(self.geoms)

    def attach(
        self,
        host: TermHost,
        geoms_by_side: dict[str, Sequence[int]],
        targets: dict[str, np.ndarray],
        weight: float,
        max_recovery_per_iter: float,
        activation_by_side: dict[str, np.ndarray] | None,
    ) -> None:
        """Bind sole geometry and target arrays."""
        recovery = float(max_recovery_per_iter)
        if not np.isfinite(recovery) or recovery <= 0.0:
            raise ValueError(
                f"clearance max_recovery_per_iter must be positive and finite, got {max_recovery_per_iter!r}"
            )
        self.geoms = {int(geom_id): side for side, geom_ids in geoms_by_side.items() for geom_id in geom_ids}
        self.targets = {key: np.asarray(value, dtype=float) for key, value in targets.items()}
        self.activation = _validate_activation(self.targets, activation_by_side, "clearance")
        self.weight = float(weight)
        self.max_recovery_per_iter = recovery
        floor = mujoco.mj_name2id(host.robot_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor < 0:
            raise ValueError("Foot clearance needs a geom named 'floor' to measure height from")
        self.floor_id = int(floor)

    def target(self, geom_id: int, frame_idx: int) -> float:
        """Return one sole geometry's clamped-frame target."""
        side = self.geoms.get(geom_id)
        if side is None:
            return -np.inf
        targets = self.targets.get(side)
        if targets is None or len(targets) == 0:
            return -np.inf
        return float(targets[min(max(frame_idx, 0), len(targets) - 1)])

    def linearize(self, host: TermHost, q: np.ndarray) -> list[RowTerm]:
        """Linearize active sole-height shortfalls."""
        if not self.geoms or self.floor_id is None:
            return []
        host._ensure_forward(q)
        model, data = host.robot_model, host.robot_data
        transform = host._build_transform_qdot_to_qvel_fast()
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64, order="C")
        candidates = []
        for geom_id, side in self.geoms.items():
            required = self.target(geom_id, host._current_frame)
            if not np.isfinite(required):
                continue
            authority = _frame_activation(self.activation.get(side), host._current_frame)
            if authority <= 0.0:
                continue
            bottom = mujoco.mj_geomDistance(model, data, geom_id, self.floor_id, 5.0, None)
            shortfall = required - bottom
            if shortfall > 0.0:
                candidates.append((geom_id, self.weight * authority, shortfall))

        terms = []
        for geom_id, weight, shortfall in candidates:
            centre = data.geom_xpos[geom_id]
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                None,
                centre,
                int(model.geom_bodyid[geom_id]),
            )
            row = (jacobian_position @ transform)[2, host.q_a_indices]
            terms.append((weight, row, min(shortfall, self.max_recovery_per_iter)))
        return terms

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last, frame_idx
        return _row_quadratic_terms(host._clearance_terms(q))


@dataclass(slots=True)
class RouteTerm:
    """Direct vertical targets for routed foot bodies."""

    name: ClassVar[str] = "foot_route"
    targets: dict[int, np.ndarray] = field(default_factory=dict)
    activation: dict[int, np.ndarray] = field(default_factory=dict)
    weight: float = 0.0
    max_recovery_per_iter: float = np.inf

    @property
    def configured(self) -> bool:
        return bool(self.targets)

    def attach(
        self,
        host: TermHost,
        targets_by_body: dict[str, np.ndarray],
        weight: float,
        activation_by_body: dict[str, np.ndarray] | None,
        max_recovery_per_iter: float | None,
    ) -> None:
        """Resolve body names and bind routed height targets."""
        named_targets = {name: np.asarray(target, dtype=float) for name, target in targets_by_body.items()}
        named_activation = _validate_activation(named_targets, activation_by_body, "route")
        if max_recovery_per_iter is None:
            recovery = np.inf
        else:
            recovery = float(max_recovery_per_iter)
            if not np.isfinite(recovery) or recovery <= 0.0:
                raise ValueError(
                    f"route max_recovery_per_iter must be positive and finite, got {max_recovery_per_iter!r}"
                )
        self.targets = {}
        self.activation = {}
        for name, target in named_targets.items():
            body_id = mujoco.mj_name2id(host.robot_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Whole-foot route body {name!r} is not in the robot model")
            self.targets[int(body_id)] = target
            self.activation[int(body_id)] = named_activation[name]
        self.weight = float(weight)
        self.max_recovery_per_iter = recovery

    def linearize(self, host: TermHost, q: np.ndarray) -> list[RowTerm]:
        """Linearize active routed vertical targets."""
        if not self.targets:
            return []
        host._ensure_forward(q)
        model, data = host.robot_model, host.robot_data
        transform = host._build_transform_qdot_to_qvel_fast()
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64, order="C")
        terms = []
        for body_id, targets in self.targets.items():
            index = min(max(host._current_frame, 0), len(targets) - 1)
            target = float(targets[index])
            if not np.isfinite(target):
                continue
            authority = _frame_activation(self.activation.get(body_id), host._current_frame)
            if authority <= 0.0:
                continue
            point = data.xpos[body_id]
            residual = float(np.clip(target - point[2], -self.max_recovery_per_iter, self.max_recovery_per_iter))
            mujoco.mj_jac(model, data, jacobian_position, None, point, body_id)
            row = (jacobian_position @ transform)[2, host.q_a_indices]
            terms.append((self.weight * authority, row, residual))
        return terms

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last, frame_idx
        return _row_quadratic_terms(host._foot_route_terms(q))


@dataclass(slots=True)
class StanceHeightTerm:
    """Direct vertical body targets during annotated stance."""

    name: ClassVar[str] = "stance_height"
    targets: dict[int, np.ndarray] = field(default_factory=dict)
    activation: dict[int, np.ndarray] = field(default_factory=dict)
    weight: float = 0.0
    max_recovery_per_iter: float = np.inf

    @property
    def configured(self) -> bool:
        return bool(self.targets)

    def attach(
        self,
        host: TermHost,
        targets_by_body: dict[str, np.ndarray],
        activation_by_body: dict[str, np.ndarray],
        weight: float,
        max_recovery_per_iter: float,
    ) -> None:
        """Validate and bind stance targets."""
        weight = float(weight)
        recovery = float(max_recovery_per_iter)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("stance-height weight must be finite and positive")
        if not np.isfinite(recovery) or recovery <= 0.0:
            raise ValueError("stance-height recovery cap must be finite and positive")
        named_targets = {name: np.asarray(value, dtype=float) for name, value in targets_by_body.items()}
        named_activation = _validate_activation(named_targets, activation_by_body, "stance height")
        self.targets = {}
        self.activation = {}
        for name, target in named_targets.items():
            body_id = mujoco.mj_name2id(host.robot_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"stance-height body {name!r} is not in the robot model")
            self.targets[int(body_id)] = target
            self.activation[int(body_id)] = named_activation[name]
        self.weight = weight
        self.max_recovery_per_iter = recovery

    def linearize(self, host: TermHost, q: np.ndarray) -> list[RowTerm]:
        """Linearize active stance-height targets."""
        if not self.targets:
            return []
        host._ensure_forward(q)
        model, data = host.robot_model, host.robot_data
        transform = host._build_transform_qdot_to_qvel_fast()
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64, order="C")
        terms = []
        for body_id, targets in self.targets.items():
            index = min(max(host._current_frame, 0), len(targets) - 1)
            authority = _frame_activation(self.activation.get(body_id), host._current_frame)
            if authority <= 0.0:
                continue
            residual = float(
                np.clip(
                    targets[index] - data.xpos[body_id, 2],
                    -self.max_recovery_per_iter,
                    self.max_recovery_per_iter,
                )
            )
            point = data.xpos[body_id]
            mujoco.mj_jac(model, data, jacobian_position, None, point, int(body_id))
            row = (jacobian_position @ transform)[2, host.q_a_indices]
            terms.append((self.weight * authority, row, residual))
        return terms

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last, frame_idx
        return _row_quadratic_terms(host._foot_stance_height_terms(q))


@dataclass(slots=True)
class SeatContactTerm:
    """Exact glute-to-seat signed-distance target during seated rests."""

    name: ClassVar[str] = "seat_contact"
    glute_geoms: tuple[int, ...] = ()
    activation: dict[int, np.ndarray] = field(default_factory=dict)
    weight: float = 0.0
    clearance: float = 0.0
    max_recovery_per_iter: float = np.inf

    @property
    def configured(self) -> bool:
        return bool(self.glute_geoms)

    def attach(
        self,
        host: TermHost,
        glute_geom_names: Sequence[str],
        activation_by_seat: dict[str, np.ndarray],
        weight: float,
        clearance: float,
        max_recovery_per_iter: float,
    ) -> None:
        """Resolve geometry names and bind seat activation arrays."""
        weight = float(weight)
        clearance = float(clearance)
        recovery = float(max_recovery_per_iter)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError("seat-contact weight must be finite and positive")
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("seat-contact clearance must be finite and non-negative")
        if not np.isfinite(recovery) or recovery <= 0.0:
            raise ValueError("seat-contact recovery cap must be finite and positive")
        if not activation_by_seat:
            raise ValueError("seat-contact requires at least one seat activation array")

        def geom_id(name: str, label: str) -> int:
            value = mujoco.mj_name2id(host.robot_model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if value < 0:
                raise ValueError(f"seat-contact {label} geom {name!r} is not in the robot model")
            return int(value)

        glutes = tuple(geom_id(name, "glute") for name in glute_geom_names)
        if not glutes:
            raise ValueError("seat-contact requires at least one glute geom")
        activations: dict[int, np.ndarray] = {}
        lengths = set()
        for name, values in activation_by_seat.items():
            value = np.asarray(values, dtype=float)
            if value.ndim != 1 or not len(value):
                raise ValueError(f"seat-contact activation for {name!r} must be a non-empty 1D array")
            if not np.all(np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
                raise ValueError(f"seat-contact activation for {name!r} must be finite and in [0, 1]")
            activations[geom_id(name, "seat")] = value
            lengths.add(len(value))
        if len(lengths) != 1:
            raise ValueError("seat-contact activation arrays must share one timebase")
        self.glute_geoms = glutes
        self.activation = activations
        self.weight = weight
        self.clearance = clearance
        self.max_recovery_per_iter = recovery

    def linearize(self, host: TermHost, q: np.ndarray) -> list[RowTerm]:
        """Linearize the nearest active glute/seat signed distance."""
        if not self.glute_geoms or not self.activation:
            return []
        active = [
            (seat, _frame_activation(activation, host._current_frame))
            for seat, activation in self.activation.items()
            if _frame_activation(activation, host._current_frame) > 0.0
        ]
        if not active:
            return []

        host._ensure_forward(q)
        model, data = host.robot_model, host.robot_data
        nearest = None
        for glute in self.glute_geoms:
            for seat, authority in active:
                fromto = np.zeros(6, dtype=float)
                distance = float(mujoco.mj_geomDistance(model, data, glute, seat, 1.0, fromto))
                candidate = (distance, glute, seat, authority, fromto)
                if nearest is None or candidate[0] < nearest[0]:
                    nearest = candidate
        if nearest is None:
            return []

        distance, glute, _seat, authority, fromto = nearest
        direction = fromto[:3] - fromto[3:]
        norm = float(np.linalg.norm(direction))
        normal = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(
            model,
            data,
            jacobian_position,
            None,
            fromto[:3],
            int(model.geom_bodyid[glute]),
        )
        row = normal @ (jacobian_position @ host._build_transform_qdot_to_qvel_fast())[:, host.q_a_indices]
        residual = float(np.clip(self.clearance - distance, -self.max_recovery_per_iter, self.max_recovery_per_iter))
        return [(self.weight * authority, row, residual)]

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last, frame_idx
        return _row_quadratic_terms(host._seat_contact_terms(q))


@dataclass(slots=True)
class OrientationTerm:
    """World-space mimic-site orientation targets."""

    name: ClassVar[str] = "orientation"
    site_ids: np.ndarray | None = None
    targets: np.ndarray | None = None
    weights: np.ndarray | None = None
    names: list[str] = field(default_factory=list)

    @property
    def configured(self) -> bool:
        return self.site_ids is not None

    def attach(
        self,
        site_ids: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        names: list[str],
    ) -> None:
        """Bind world rotations and per-site weights."""
        self.site_ids = np.asarray(site_ids, dtype=int)
        self.targets = np.asarray(targets, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.names = list(names)
        if self.targets.shape[1] != len(self.site_ids):
            raise ValueError("targets and site_ids disagree on the number of sites")

    def linearize(self, host: TermHost, q: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Linearize world-space rotation-vector residuals."""
        from scipy.spatial.transform import Rotation

        if self.site_ids is None or self.targets is None:
            raise RuntimeError("orientation targets are not attached")
        host._ensure_forward(q)
        transform = host._build_transform_qdot_to_qvel_fast()
        site_count = len(self.site_ids)
        current = host.robot_data.site_xmat[self.site_ids].reshape(site_count, 3, 3)
        relative = self.targets[frame_idx] @ np.swapaxes(current, 1, 2)
        residual = Rotation.from_matrix(relative).as_rotvec().reshape(-1)
        jacobian = np.zeros((3 * site_count, host.nq_a))
        rotational = np.zeros((3, host.robot_model.nv), dtype=np.float64, order="C")
        for index, site_id in enumerate(self.site_ids):
            mujoco.mj_jacSite(host.robot_model, host.robot_data, None, rotational, int(site_id))
            jacobian[3 * index : 3 * index + 3, :] = (rotational @ transform)[:, host.q_a_indices]
        return residual, jacobian

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last
        target, jacobian = host._orientation_linearisation(q, frame_idx)
        row_scale = np.sqrt(np.repeat(self.weights, 3))
        return [_QuadraticTerm(jacobian, target, row_scale**2, cvxpy_row_scale=row_scale)]


@dataclass(slots=True)
class CouplerTerm:
    """Polynomial dependent/independent joint consistency term."""

    name: ClassVar[str] = "joint_coupler"
    pairs: list[tuple[int, int, np.ndarray]] = field(default_factory=list)
    dependent: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    independent: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    polynomials: np.ndarray = field(default_factory=lambda: np.empty((0, 5), dtype=float))
    rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    weight: float = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.pairs)

    def attach(self, host: TermHost, couplers: list, weight: float) -> None:
        """Resolve full-position indices to actuated-coordinate indices."""
        self.pairs = [
            (
                int(np.flatnonzero(host.q_a_indices == dependent)[0]),
                int(np.flatnonzero(host.q_a_indices == independent)[0]),
                polynomial,
            )
            for dependent, independent, polynomial in couplers
            if dependent in host.q_a_indices and independent in host.q_a_indices
        ]
        self.dependent = np.fromiter(
            (dependent for dependent, _, _ in self.pairs),
            dtype=int,
            count=len(self.pairs),
        )
        self.independent = np.fromiter(
            (independent for _, independent, _ in self.pairs),
            dtype=int,
            count=len(self.pairs),
        )
        self.polynomials = (
            np.stack([polynomial for _, _, polynomial in self.pairs]) if self.pairs else np.empty((0, 5), dtype=float)
        )
        self.rows = np.arange(len(self.pairs))
        self.weight = float(weight)

    def linearize(self, host: TermHost, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearize polynomial coupler residuals bit-for-bit."""
        row_count = len(self.rows)
        if row_count == 0:
            return np.zeros((0, host.nq_a)), np.zeros(0)
        x = q[host.q_a_indices[self.independent]]
        y = q[host.q_a_indices[self.dependent]]
        powers = np.array([[1.0, value, value**2, value**3, value**4] for value in x])
        derivatives = np.array([[0.0, 1.0, 2 * value, 3 * value**2, 4 * value**3] for value in x])
        values = (self.polynomials[:, None, :] @ powers[:, :, None])[:, 0, 0]
        slopes = (self.polynomials[:, None, :] @ derivatives[:, :, None])[:, 0, 0]
        jacobian = np.zeros((row_count, host.nq_a))
        jacobian[self.rows, self.dependent] = 1.0
        jacobian[self.rows, self.independent] = -slopes
        residual = np.array([float(values[row]) - y[row] for row in self.rows])
        return jacobian, residual

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        del q_t_last, frame_idx
        jacobian, target = host._coupler_linearisation(q)
        return [_QuadraticTerm(jacobian, target, self.weight)]


@dataclass(slots=True)
class FootAnchorTerm:
    """Touchdown anchoring and frame-relative horizontal velocity terms."""

    name: ClassVar[str] = "foot_anchor"
    contact: dict[str, np.ndarray] | None = None
    weight: float = 0.0
    anchor: dict[str, np.ndarray] = field(default_factory=dict)
    body_ids: dict[str, int] = field(default_factory=dict)
    ramp: dict[str, np.ndarray] = field(default_factory=dict)
    frame: int = -1
    velocity_contact: dict[str, np.ndarray] = field(default_factory=dict)
    velocity_sites: dict[str, str] = field(default_factory=dict)
    velocity_site_ids: dict[str, int] = field(default_factory=dict)
    velocity_previous: dict[str, np.ndarray] = field(default_factory=dict)
    velocity_weight: float = 0.0
    velocity_tracking_weight: float = 0.0
    velocity_tolerance_m: float = 0.0

    @property
    def configured(self) -> bool:
        return self.contact is not None

    def attach(
        self,
        host: TermHost,
        contact: dict,
        weight: float,
        ramp_frames: int,
        preserve_clipped_boundaries: bool,
        velocity_contact: dict | None,
        velocity_sites: dict | None,
        velocity_weight: float,
        velocity_tracking_weight: float,
        velocity_tolerance_m: float,
    ) -> None:
        """Bind contact schedules, body IDs, and optional speed-control sites."""
        if not np.isfinite(velocity_weight) or velocity_weight < 0.0:
            raise ValueError("velocity_weight must be finite and non-negative")
        if not np.isfinite(velocity_tracking_weight) or velocity_tracking_weight < 0.0:
            raise ValueError("velocity_tracking_weight must be finite and non-negative")
        if not np.isfinite(velocity_tolerance_m) or velocity_tolerance_m < 0.0:
            raise ValueError("velocity_tolerance_m must be finite and non-negative")
        self.contact = {key: np.asarray(value, dtype=bool) for key, value in contact.items()}
        self.weight = float(weight)
        self.ramp = {
            key: contact_ramp(
                value,
                ramp_frames,
                preserve_clipped_boundaries=preserve_clipped_boundaries,
            )
            for key, value in self.contact.items()
        }
        self.anchor = {}
        self.velocity_contact = {key: np.asarray(value, dtype=bool) for key, value in (velocity_contact or {}).items()}
        self.body_ids = {}
        for label, body_name in host.foot_links.items():
            body_id = mujoco.mj_name2id(host.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"foot anchor body {body_name!r} is absent from the model")
            self.body_ids[label] = body_id
        self.velocity_sites = dict(velocity_sites or {})
        missing_sites = set(self.velocity_sites) - set(host.foot_links)
        if missing_sites:
            raise ValueError(f"velocity site labels are not foot links: {sorted(missing_sites)}")
        missing_labels = set(self.velocity_contact) - set(self.velocity_sites)
        if self.velocity_sites and missing_labels:
            raise ValueError(f"velocity contact labels have no speed-control site: {sorted(missing_labels)}")
        self.velocity_site_ids = {}
        for label, site_name in self.velocity_sites.items():
            site_id = mujoco.mj_name2id(host.robot_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise ValueError(f"foot velocity site {site_name!r} is absent from the model")
            self.velocity_site_ids[label] = site_id
        self.velocity_weight = float(velocity_weight)
        self.velocity_tracking_weight = float(velocity_tracking_weight)
        self.velocity_tolerance_m = float(velocity_tolerance_m)
        self.velocity_previous = {}
        self.frame = -1

    def kinematics(self, host: TermHost, q: np.ndarray) -> tuple[dict, dict, dict, dict]:
        """Calculate anchor and speed-control kinematics in one forward pass."""
        host.robot_data.qpos[:] = q
        mujoco.mj_forward(host.robot_model, host.robot_data)
        transform = host._build_transform_qdot_to_qvel_fast()

        anchor_jacobians, anchor_positions = {}, {}
        jacobian_position = np.zeros((3, host.robot_model.nv), dtype=np.float64, order="C")
        for label, body_id in self.body_ids.items():
            jacobian_position[:] = 0.0
            point = np.asarray(host.robot_data.xpos[body_id], dtype=np.float64).reshape(3, 1)
            mujoco.mj_jac(
                host.robot_model,
                host.robot_data,
                jacobian_position,
                None,
                point,
                int(body_id),
            )
            anchor_jacobians[label] = np.array(
                (jacobian_position @ transform)[:, host.q_a_indices],
                dtype=float,
                copy=True,
            )
            anchor_positions[label] = np.array(host.robot_data.xpos[body_id], dtype=float, copy=True)

        if not self.velocity_sites:
            velocity_jacobians = {label: value[:2, :] for label, value in anchor_jacobians.items()}
            velocity_positions = {label: value[:2] for label, value in anchor_positions.items()}
            return anchor_jacobians, anchor_positions, velocity_jacobians, velocity_positions

        velocity_jacobians, velocity_positions = {}, {}
        for label, site_id in self.velocity_site_ids.items():
            jacobian_position[:] = 0.0
            mujoco.mj_jacSite(host.robot_model, host.robot_data, jacobian_position, None, site_id)
            velocity_jacobians[label] = np.array(
                (jacobian_position @ transform)[:2, host.q_a_indices],
                dtype=float,
                copy=True,
            )
            velocity_positions[label] = np.array(host.robot_data.site_xpos[site_id, :2], dtype=float, copy=True)
        return anchor_jacobians, anchor_positions, velocity_jacobians, velocity_positions

    def linearize(
        self,
        host: TermHost,
        q: np.ndarray,
        q_t_last: np.ndarray,
        frame_idx: int,
    ) -> list[RowTerm]:
        """Linearize touchdown anchors and frame-relative speed residuals."""
        jacobians, positions, velocity_jacobians, velocity_positions = self.kinematics(host, q)
        new_frame = frame_idx != self.frame
        self.frame = frame_idx
        captured = None
        if new_frame:
            _, captured, _, self.velocity_previous = self.kinematics(host, q_t_last)

        terms = []
        for label in host.foot_links:
            ramp = self.ramp.get(label)
            if ramp is not None and frame_idx < len(ramp):
                if not self.contact[label][frame_idx]:
                    self.anchor.pop(label, None)
                else:
                    if new_frame and label not in self.anchor:
                        self.anchor[label] = np.array(captured[label][:2], dtype=float)
                    anchor = self.anchor.get(label)
                    if anchor is not None and ramp[frame_idx] > 0.0:
                        terms.append(
                            (
                                self.weight * float(ramp[frame_idx]),
                                jacobians[label][:2, :],
                                positions[label][:2] - anchor,
                            )
                        )

            velocity = self.velocity_contact.get(label)
            previous = self.velocity_previous.get(label)
            if (
                (self.velocity_weight > 0.0 or self.velocity_tracking_weight > 0.0)
                and velocity is not None
                and frame_idx < len(velocity)
                and velocity[frame_idx]
                and previous is not None
            ):
                displacement = velocity_positions[label] - previous
                distance = float(np.linalg.norm(displacement))
                if self.velocity_tracking_weight > 0.0:
                    terms.append((self.velocity_tracking_weight, velocity_jacobians[label], displacement))
                if self.velocity_weight > 0.0 and distance > self.velocity_tolerance_m:
                    excess = displacement * (1.0 - self.velocity_tolerance_m / distance)
                    terms.append((self.velocity_weight, velocity_jacobians[label], excess))
        return terms

    def quadratic_terms(
        self, host: TermHost, q: np.ndarray, q_t_last: np.ndarray, frame_idx: int
    ) -> list[_QuadraticTerm]:
        return [
            _QuadraticTerm(jacobian, -offset, weight)
            for weight, jacobian, offset in host._foot_anchor_terms(q, q_t_last, frame_idx)
        ]


@dataclass(slots=True)
class AttachedTermState:
    """All optional terms in their stable scientific assembly order."""

    orientation: OrientationTerm = field(default_factory=OrientationTerm)
    foot_anchor: FootAnchorTerm = field(default_factory=FootAnchorTerm)
    self_collision: SelfCollisionTerm = field(default_factory=SelfCollisionTerm)
    clearance: ClearanceTerm = field(default_factory=ClearanceTerm)
    route: RouteTerm = field(default_factory=RouteTerm)
    stance_height: StanceHeightTerm = field(default_factory=StanceHeightTerm)
    seat_contact: SeatContactTerm = field(default_factory=SeatContactTerm)
    coupler: CouplerTerm = field(default_factory=CouplerTerm)

    def components(self) -> tuple[QuadraticComponent, ...]:
        """Return components in the objective's intentional order."""
        return (
            self.orientation,
            self.foot_anchor,
            self.self_collision,
            self.clearance,
            self.route,
            self.stance_height,
            self.seat_contact,
            self.coupler,
        )

    def active_names(self) -> tuple[str, ...]:
        """Describe configured contributions without inspecting solver internals."""
        return tuple(component.name for component in self.components() if component.configured)

    def quadratic_terms(
        self,
        host: TermHost,
        q: np.ndarray,
        q_t_last: np.ndarray,
        frame_idx: int,
    ) -> list[_QuadraticTerm]:
        """Linearize configured components in stable order."""
        return [
            term
            for component in self.components()
            if component.configured
            for term in component.quadratic_terms(host, q, q_t_last, frame_idx)
        ]


@dataclass(slots=True)
class RetargeterConstraintState:
    """Complete instance-owned state of TERRA's solver extensions."""

    environment: EnvironmentConstraintState = field(default_factory=EnvironmentConstraintState)
    objective: AttachedTermState = field(default_factory=AttachedTermState)
