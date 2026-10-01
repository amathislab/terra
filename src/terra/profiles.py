"""Resolve validated TERRA and OmniRetarget solver configurations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Literal

from terra.constants import (
    MYOFULLBODY_FOOT_LINKS,
    MYOFULLBODY_POSTHOC_SELF_COLLISION_PAIRS,
    MYOFULLBODY_SELF_COLLISION_PAIRS,
)
from terra.defaults import (
    DEFAULT_ANCHOR_GATE,
    DEFAULT_AXIAL_SMOOTH_WEIGHT,
    DEFAULT_CLEARANCE_CAP,
    DEFAULT_CLEARANCE_FRACTION,
    DEFAULT_CLEARANCE_LOOKAHEAD,
    DEFAULT_CLEARANCE_MAX_RECOVERY_PER_ITER,
    DEFAULT_CLEARANCE_RAMP_FRAMES,
    DEFAULT_CLEARANCE_SURFACE_RAMP_SECONDS,
    DEFAULT_CLEARANCE_WEIGHT,
    DEFAULT_CONTACT_HEIGHT,
    DEFAULT_CONTACT_SPEED,
    DEFAULT_COUPLER_WEIGHT,
    DEFAULT_FLAT_FOOT_ORIENT_WEIGHT,
    DEFAULT_FLAT_SELF_COLLISION_WEIGHT,
    DEFAULT_FOOT_ANCHOR_WEIGHT,
    DEFAULT_FOOT_MODE,
    DEFAULT_FOOT_RAMP_FRAMES,
    DEFAULT_FOOT_VELOCITY_LIMIT,
    DEFAULT_FOOT_VELOCITY_TRACKING_WEIGHT,
    DEFAULT_FOOT_VELOCITY_WEIGHT,
    DEFAULT_INITIAL_STEP_SIZE,
    DEFAULT_MAX_ROOT_SOURCE_DEVIATION,
    DEFAULT_MAX_ROOT_STEP,
    DEFAULT_MIN_SWING_CLEARANCE,
    DEFAULT_MTP_SMOOTH_WEIGHT,
    DEFAULT_ORIENT_WEIGHT,
    DEFAULT_POSTHOC_MAX_RUN,
    DEFAULT_POSTHOC_PEN_THRESHOLD,
    DEFAULT_POSTHOC_TENDON_MAX_FRAMES,
    DEFAULT_POSTHOC_TENDON_MAX_LOCAL_FRAMES,
    DEFAULT_POSTHOC_TENDON_THRESHOLD,
    DEFAULT_POSTHOC_TRACKING_MAX_REGRESSION,
    DEFAULT_POSTHOC_TRACKING_MEAN_REGRESSION,
    DEFAULT_SEAT_CONTACT_CLEARANCE,
    DEFAULT_SEAT_CONTACT_MAX_RECOVERY_PER_ITER,
    DEFAULT_SEAT_CONTACT_RAMP_FRAMES,
    DEFAULT_SEAT_CONTACT_WEIGHT,
    DEFAULT_SELF_COLLISION_MAX_RECOVERY_PER_ITER,
    DEFAULT_SELF_COLLISION_TOLERANCE,
    DEFAULT_SELF_COLLISION_WEIGHT,
    DEFAULT_SOLVER_BACKEND,
    DEFAULT_STANCE_HEIGHT_MAX_RECOVERY_PER_ITER,
    DEFAULT_STANCE_HEIGHT_WEIGHT,
    DEFAULT_SWING_TARGET_LIFT,
    DEFAULT_SWING_TARGET_MAX_RECOVERY_PER_ITER,
    DEFAULT_SWING_TARGET_RAMP_FRAMES,
    DEFAULT_SWING_TARGET_WEIGHT,
    DEFAULT_TERRAIN_PENETRATION_TOLERANCE,
    DEFAULT_TRUNK_Q_DIAG,
    DEFAULT_TRUNK_SMOOTH_WEIGHT,
    DEFAULT_WARMUP_FRAMES,
)

MethodProfile = Literal["terra", "omniretarget"]
SolverBackend = Literal["legacy", "condensed_cvxpy", "native_clarabel"]


@dataclass(frozen=True, slots=True)
class SolverConfig(Mapping[str, object]):
    """Complete, validated input record for one retargeting solve.

    The public API still accepts a flat mapping so existing JSON configuration files
    remain valid. Once resolved, every accepted key has a named field and unknown
    keys fail closed. Conditional defaults are materialized by :meth:`for_scene`
    before a :class:`terra.assembly.SolveContext` is created.
    """

    method_profile: MethodProfile = "terra"
    # Source preparation and reconstruction.
    use_fitted_shape: bool = True
    calibrate_sites: bool | None = None
    site_calibration_version: str | None = None
    source_sole_offsets: Mapping[str, float] | None = None
    extra_landmarks: Mapping[str, str] = field(default_factory=dict)
    contact_speed: float = DEFAULT_CONTACT_SPEED
    contact_height: float = DEFAULT_CONTACT_HEIGHT
    terrain: object | None = None
    terrain_fit: Mapping[str, object] = field(default_factory=dict)

    # Interaction scene.
    terrain_point_spacing: float = 0.20
    ground_range: tuple[float, float] | None = None
    ground_size: int | None = None

    # Base SQP and smoothing.
    solver_backend: SolverBackend = DEFAULT_SOLVER_BACKEND
    q_a_init_idx: int = -7
    activate_joint_limits: bool = True
    activate_obj_non_penetration: bool = True
    penetration_tolerance: float | None = None
    foot_sticking_tolerance: float = 1e-3
    step_size: float = 0.2
    initial_step_size: float = DEFAULT_INITIAL_STEP_SIZE
    debug: bool = False
    foot_links: Mapping[str, str] = field(default_factory=lambda: dict(MYOFULLBODY_FOOT_LINKS))
    smooth_weight: float = 0.2
    trunk_smooth_weight: float = DEFAULT_TRUNK_SMOOTH_WEIGHT
    trunk_q_diag: float = DEFAULT_TRUNK_Q_DIAG
    axial_smooth_weight: float = DEFAULT_AXIAL_SMOOTH_WEIGHT
    mtp_smooth_weight: float | None = None
    max_root_step: float = DEFAULT_MAX_ROOT_STEP
    max_root_source_deviation: float = DEFAULT_MAX_ROOT_SOURCE_DEVIATION

    # Orientation and articulated-manifold costs.
    orient_weight: float = DEFAULT_ORIENT_WEIGHT
    foot_orient_mode: str = "on"
    foot_orient_weight: float | None = None
    upper_orient_weight: float | None = None
    torso_frame_mode: str = "off"
    torso_orient_weight: float = DEFAULT_ORIENT_WEIGHT
    coupler_weight: float = DEFAULT_COUPLER_WEIGHT

    # Environment and self-collision constraints.
    nonpen_mode: str = "scene"
    nonpen_engage_frame: int | None = None
    nonpen_max_recovery: float = 0.01
    selfpen_mode: str = "legs"
    selfpen_engage_frame: int | None = None
    selfpen_pairs: tuple[tuple[str, str], ...] = MYOFULLBODY_SELF_COLLISION_PAIRS
    selfpen_tolerance: float = DEFAULT_SELF_COLLISION_TOLERANCE
    selfpen_weight: float | None = None
    selfpen_max_recovery_per_iter: float = DEFAULT_SELF_COLLISION_MAX_RECOVERY_PER_ITER

    # Contact, stance, and foot motion.
    warmup_frames: int = DEFAULT_WARMUP_FRAMES
    foot_mode: str = DEFAULT_FOOT_MODE
    foot_planted_speed: float | None = None
    sole_offset_mode: str | None = None
    anchor_gate: str = DEFAULT_ANCHOR_GATE
    foot_anchor_weight: float = DEFAULT_FOOT_ANCHOR_WEIGHT
    foot_velocity_weight: float = DEFAULT_FOOT_VELOCITY_WEIGHT
    foot_velocity_tracking_weight: float = DEFAULT_FOOT_VELOCITY_TRACKING_WEIGHT
    foot_velocity_limit: float = DEFAULT_FOOT_VELOCITY_LIMIT
    foot_ramp_frames: int = DEFAULT_FOOT_RAMP_FRAMES
    stance_height_weight: float = DEFAULT_STANCE_HEIGHT_WEIGHT
    stance_height_probe_mode: str = "all"
    stance_height_probe_tolerance_m: float = 0.015
    stance_height_ramp_frames: int = DEFAULT_FOOT_RAMP_FRAMES
    stance_height_release_ramp_frames: int | None = None
    stance_height_max_recovery_per_iter: float = DEFAULT_STANCE_HEIGHT_MAX_RECOVERY_PER_ITER
    stance_landmark_clearance_cap_m: float | None = None
    stance_landmark_midstance_both_cap_m: float | None = None
    stance_landmark_midstance_fraction: float = 0.5
    stance_landmark_probe_tolerance_m: float = 0.015

    # Swing clearance and route targets.
    clearance_mode: str | None = None
    clearance_fraction: float = DEFAULT_CLEARANCE_FRACTION
    clearance_cap: float = DEFAULT_CLEARANCE_CAP
    clearance_lookahead: float = DEFAULT_CLEARANCE_LOOKAHEAD
    min_swing_clearance: float = DEFAULT_MIN_SWING_CLEARANCE
    clearance_ramp_frames: int = DEFAULT_CLEARANCE_RAMP_FRAMES
    clearance_surface_ramp_seconds: float = DEFAULT_CLEARANCE_SURFACE_RAMP_SECONDS
    clearance_weight: float = DEFAULT_CLEARANCE_WEIGHT
    clearance_max_recovery_per_iter: float = DEFAULT_CLEARANCE_MAX_RECOVERY_PER_ITER
    swing_target_lift: float = DEFAULT_SWING_TARGET_LIFT
    swing_target_ramp_frames: int = DEFAULT_SWING_TARGET_RAMP_FRAMES
    swing_target_weight: float = DEFAULT_SWING_TARGET_WEIGHT
    swing_target_max_recovery_per_iter: float = DEFAULT_SWING_TARGET_MAX_RECOVERY_PER_ITER

    # Chair contact.
    seat_contact_mode: str = "off"
    seat_contact_ramp_frames: int = DEFAULT_SEAT_CONTACT_RAMP_FRAMES
    seat_contact_weight: float = DEFAULT_SEAT_CONTACT_WEIGHT
    seat_contact_clearance_m: float = DEFAULT_SEAT_CONTACT_CLEARANCE
    seat_contact_max_recovery_per_iter: float = DEFAULT_SEAT_CONTACT_MAX_RECOVERY_PER_ITER

    # Bounded postprocessing.
    posthoc_repair: bool = True
    posthoc_pen_threshold: float = DEFAULT_POSTHOC_PEN_THRESHOLD
    posthoc_max_run: int = DEFAULT_POSTHOC_MAX_RUN
    posthoc_selfpen_pairs: tuple[tuple[str, str], ...] = MYOFULLBODY_POSTHOC_SELF_COLLISION_PAIRS
    posthoc_tendon_repair: bool = True
    posthoc_tendon_threshold: float = DEFAULT_POSTHOC_TENDON_THRESHOLD
    posthoc_tendon_max_frames: int = DEFAULT_POSTHOC_TENDON_MAX_FRAMES
    posthoc_tendon_max_local_frames: int = DEFAULT_POSTHOC_TENDON_MAX_LOCAL_FRAMES
    posthoc_tracking_mean_regression: float = DEFAULT_POSTHOC_TRACKING_MEAN_REGRESSION
    posthoc_tracking_max_regression: float = DEFAULT_POSTHOC_TRACKING_MAX_REGRESSION

    _explicit_keys: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate controlling modes and freeze mapping-valued fields."""
        modes = {
            "method_profile": (self.method_profile, {"terra", "omniretarget"}),
            "solver_backend": (self.solver_backend, {"legacy", "condensed_cvxpy", "native_clarabel"}),
            "foot_mode": (self.foot_mode, {"anchored", "omniretarget", "off"}),
            "nonpen_mode": (self.nonpen_mode, {"scene", "off"}),
            "selfpen_mode": (self.selfpen_mode, {"legs", "off"}),
            "foot_orient_mode": (self.foot_orient_mode, {"on", "off"}),
            "torso_frame_mode": (self.torso_frame_mode, {"positions", "off"}),
            "seat_contact_mode": (self.seat_contact_mode, {"glute_distance", "off"}),
            "stance_height_probe_mode": (self.stance_height_probe_mode, {"all", "source_support"}),
        }
        optional_modes = {
            "sole_offset_mode": (self.sole_offset_mode, {"on", "off"}),
            "clearance_mode": (self.clearance_mode, {"source", "off"}),
        }
        for name, (value, accepted) in modes.items():
            if value not in accepted:
                raise ValueError(f"{name} must be one of {sorted(accepted)}, got {value!r}")
        for name, (value, accepted) in optional_modes.items():
            if name in self._explicit_keys and value is None:
                raise ValueError(f"{name} may not be null when explicitly configured")
            if value is not None and value not in accepted:
                raise ValueError(f"{name} must be one of {sorted(accepted)}, got {value!r}")
        for name in (
            "calibrate_sites",
            "penetration_tolerance",
            "foot_orient_weight",
            "upper_orient_weight",
            "selfpen_weight",
            "foot_planted_speed",
            "ground_range",
            "ground_size",
            "stance_height_release_ramp_frames",
        ):
            if name in self._explicit_keys and getattr(self, name) is None:
                raise ValueError(f"{name} may not be null when explicitly configured")
        for name in (
            "source_sole_offsets",
            "extra_landmarks",
            "terrain_fit",
            "foot_links",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, MappingProxyType):
                object.__setattr__(self, name, MappingProxyType(dict(value)))

    @classmethod
    def field_names(cls) -> frozenset[str]:
        """Return all accepted flat configuration keys."""
        return frozenset(item.name for item in fields(cls) if not item.name.startswith("_"))

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None = None) -> SolverConfig:
        """Resolve a flat public mapping into a validated typed record."""
        raw = dict(config or {})
        profile = raw.get("method_profile")
        if profile is not None and not isinstance(profile, str):
            raise ValueError("method_profile must be 'terra' or 'omniretarget'")
        return resolve_solver_config(profile, raw)

    @classmethod
    def _from_resolved(
        cls,
        resolved: Mapping[str, object],
        *,
        explicit_keys: frozenset[str],
    ) -> SolverConfig:
        unknown = sorted(set(resolved) - cls.field_names())
        if unknown:
            names = ", ".join(repr(name) for name in unknown)
            raise ValueError(f"unknown TERRA solver configuration key(s): {names}")
        return cls(**dict(resolved), _explicit_keys=explicit_keys)  # type: ignore[arg-type]

    def for_source_calibration(self, *, calibrate_sites: bool, version: str) -> SolverConfig:
        """Materialize source-landmark calibration choices."""
        return replace(self, calibrate_sites=bool(calibrate_sites), site_calibration_version=str(version))

    def for_scene(self, *, on_terrain: bool) -> SolverConfig:
        """Materialize every terrain-dependent default before solver assembly."""
        explicit = self._explicit_keys

        def selected(name: str, current: Any, default: Any) -> Any:
            return current if name in explicit else default

        orient_default = (
            0.0
            if self.foot_orient_mode == "off"
            else self.orient_weight
            if on_terrain
            else DEFAULT_FLAT_FOOT_ORIENT_WEIGHT
        )
        ground_range_default = (-3.0, 3.0) if on_terrain else (-10.0, 10.0)
        ground_size_default = 8 if on_terrain else 10
        return replace(
            self,
            penetration_tolerance=selected(
                "penetration_tolerance",
                self.penetration_tolerance,
                DEFAULT_TERRAIN_PENETRATION_TOLERANCE if on_terrain else 1e-3,
            ),
            mtp_smooth_weight=selected(
                "mtp_smooth_weight",
                self.mtp_smooth_weight,
                DEFAULT_MTP_SMOOTH_WEIGHT if on_terrain else None,
            ),
            foot_orient_weight=selected("foot_orient_weight", self.foot_orient_weight, orient_default),
            selfpen_weight=selected(
                "selfpen_weight",
                self.selfpen_weight,
                DEFAULT_SELF_COLLISION_WEIGHT if on_terrain else DEFAULT_FLAT_SELF_COLLISION_WEIGHT,
            ),
            sole_offset_mode=selected("sole_offset_mode", self.sole_offset_mode, "on" if on_terrain else "off"),
            clearance_mode=selected("clearance_mode", self.clearance_mode, "source" if on_terrain else "off"),
            ground_range=selected("ground_range", self.ground_range, ground_range_default),
            ground_size=selected("ground_size", self.ground_size, ground_size_default),
            stance_height_release_ramp_frames=selected(
                "stance_height_release_ramp_frames",
                self.stance_height_release_ramp_frames,
                self.stance_height_ramp_frames,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete flat configuration."""
        resolved = {}
        for name in self.field_names():
            value = getattr(self, name)
            resolved[name] = dict(value) if isinstance(value, Mapping) else value
        return resolved

    def __getitem__(self, key: str) -> object:
        if key not in self.field_names():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.field_names()))

    def __len__(self) -> int:
        return len(self.field_names())


# Definitive global profile selected by the v2 residual-weight search.
# Scene resolution
# still substitutes the documented flat-scene values for fields whose meaning is
# terrain-dependent (for example clearance mode and self-collision weight).
FROZEN_TERRA_PROFILE_NAME = "v2-p02"
FROZEN_TERRA_PROFILE_SHA256 = "1e734d48345686dbc548e7c379cbab321f621cf6d4642494dfbceb273ae06a04"
FROZEN_TERRA_TERRAIN_PROFILE: Mapping[str, object] = MappingProxyType(
    {
        "smooth_weight": 0.2,
        "trunk_smooth_weight": 2.0,
        "trunk_q_diag": 0.001,
        "axial_smooth_weight": 10.0,
        "mtp_smooth_weight": 10.0,
        "orient_weight": 0.5,
        "foot_orient_mode": "off",
        "foot_orient_weight": 0.0,
        "upper_orient_weight": 0.0,
        "torso_frame_mode": "off",
        "torso_orient_weight": 0.0,
        "coupler_weight": 0.0,
        "nonpen_mode": "scene",
        "penetration_tolerance": 0.0009,
        "nonpen_max_recovery": 0.01,
        "selfpen_mode": "legs",
        "selfpen_tolerance": 0.002,
        "selfpen_weight": 20000.0,
        "selfpen_max_recovery_per_iter": 0.002,
        "foot_mode": "anchored",
        "sole_offset_mode": "on",
        "foot_anchor_weight": 100.0,
        "foot_velocity_weight": 150.0,
        "foot_velocity_tracking_weight": 200.0,
        "foot_velocity_limit": 0.25,
        "stance_height_weight": 400.0,
        "stance_height_probe_mode": "all",
        "stance_height_probe_tolerance_m": 0.015,
        "stance_height_max_recovery_per_iter": 0.004,
        "clearance_mode": "source",
        "clearance_fraction": 0.7,
        "clearance_cap": 0.15,
        "clearance_lookahead": 0.12,
        "min_swing_clearance": 0.0,
        "clearance_weight": 1500.0,
        "clearance_max_recovery_per_iter": 0.01,
        "swing_target_lift": 0.06,
        "swing_target_weight": 1500.0,
        "swing_target_max_recovery_per_iter": 0.004,
        "seat_contact_mode": "glute_distance",
        "seat_contact_weight": 2000.0,
    }
)

_SCENE_CONDITIONAL_PROFILE_KEYS = frozenset(
    {
        "clearance_mode",
        "foot_orient_weight",
        "mtp_smooth_weight",
        "penetration_tolerance",
        "selfpen_weight",
        "sole_offset_mode",
    }
)
TERRA_PROFILE_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {key: value for key, value in FROZEN_TERRA_TERRAIN_PROFILE.items() if key not in _SCENE_CONDITIONAL_PROFILE_KEYS}
)

OMNIRETARGET_PROFILE = {
    "solver_backend": "legacy",
    "use_fitted_shape": True,
    "calibrate_sites": True,
    "activate_obj_non_penetration": True,
    "activate_joint_limits": True,
    "orient_weight": 0.0,
    "coupler_weight": 0.0,
    "foot_mode": "omniretarget",
    "foot_velocity_weight": 0.0,
    "foot_velocity_tracking_weight": 0.0,
    "warmup_frames": 0,
    "nonpen_mode": "off",
    "selfpen_mode": "off",
    "clearance_mode": "off",
    "sole_offset_mode": "off",
    "foot_orient_mode": "off",
    "mtp_smooth_weight": 0.0,
    "min_swing_clearance": 0.0,
    "swing_target_lift": 0.0,
    "seat_contact_mode": "off",
    "posthoc_repair": False,
    "posthoc_tendon_repair": False,
    "trunk_q_diag": 1e-3,
    "trunk_smooth_weight": 0.2,
    "axial_smooth_weight": 0.2,
}


# OmniRetarget's objective inside the otherwise unchanged TERRA pipeline.  In
# particular, this keeps the resolved TERRA solver backend, warm-up,
# initialization, smoothing, temporal rate, and post-hoc repair settings.  It
# changes only objective/input corrections introduced by TERRA, reverting foot
# sticking and object non-penetration to the inherited OmniRetarget terms.
MATCHED_TERRA_CORE_OVERRIDES = {
    "activate_obj_non_penetration": True,
    "orient_weight": 0.0,
    "foot_orient_weight": 0.0,
    "upper_orient_weight": 0.0,
    "torso_frame_mode": "off",
    "torso_orient_weight": 0.0,
    "foot_orient_mode": "off",
    "coupler_weight": 0.0,
    "foot_mode": "omniretarget",
    "foot_anchor_weight": 0.0,
    "foot_velocity_weight": 0.0,
    "foot_velocity_tracking_weight": 0.0,
    "stance_height_weight": 0.0,
    "nonpen_mode": "off",
    "selfpen_mode": "off",
    "clearance_mode": "off",
    "sole_offset_mode": "off",
    "min_swing_clearance": 0.0,
    "swing_target_lift": 0.0,
    "seat_contact_mode": "off",
}

TERRA_SPECIFIC_CONTROLS = (
    "orient_weight",
    "foot_orient_weight",
    "upper_orient_weight",
    "torso_frame_mode",
    "torso_orient_weight",
    "foot_velocity_weight",
    "foot_velocity_tracking_weight",
    "foot_velocity_limit",
    "stance_height_weight",
    "coupler_weight",
    "warmup_frames",
    "nonpen_mode",
    "selfpen_mode",
    "clearance_mode",
    "sole_offset_mode",
    "foot_orient_mode",
    "mtp_smooth_weight",
    "min_swing_clearance",
    "swing_target_lift",
    "seat_contact_mode",
    "posthoc_repair",
    "posthoc_tendon_repair",
)


def resolve_method_profile(profile: str | None, config: Mapping[str, object] | None = None) -> dict[str, object]:
    """Resolve a named method profile while preserving the flat public format."""
    profile = profile or "terra"
    if profile not in {"terra", "omniretarget"}:
        raise ValueError(f"method profile must be 'terra' or 'omniretarget', got {profile!r}")
    resolved = deepcopy(dict(config or {}))
    unknown = sorted(set(resolved) - SolverConfig.field_names())
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"unknown TERRA solver configuration key(s): {names}")
    if profile == "omniretarget":
        resolved.update(OMNIRETARGET_PROFILE)
    else:
        for key, value in TERRA_PROFILE_DEFAULTS.items():
            resolved.setdefault(key, value)
    resolved["method_profile"] = profile
    return resolved


def resolve_solver_config(profile: str | None, config: Mapping[str, object] | None = None) -> SolverConfig:
    """Resolve and validate a public mapping as a typed solver record."""
    raw = dict(config or {})
    resolved = resolve_method_profile(profile, raw)
    explicit = frozenset(raw) | ({"method_profile"} if profile is not None else set())
    if (profile or "terra") == "omniretarget":
        explicit |= frozenset(OMNIRETARGET_PROFILE)
    return SolverConfig._from_resolved(resolved, explicit_keys=frozenset(explicit))


def _value(config: Mapping[str, object] | SolverConfig, key: str, default: object) -> object:
    """Read a profile-inspection value from either public or typed configuration."""
    if isinstance(config, SolverConfig):
        return getattr(config, key, default)
    return config.get(key, default)


def active_qp_terms(config: Mapping[str, object] | SolverConfig, *, on_terrain: bool = True) -> list[str]:
    """List active TERRA-specific solver terms."""
    active = []
    if float(_value(config, "orient_weight", 1.0)) > 0:
        active.append("orientation")
    if (
        _value(config, "torso_frame_mode", "off") == "positions"
        and float(_value(config, "torso_orient_weight", 1.0)) > 0
    ):
        active.append("position_derived_torso_orientation")
    if float(_value(config, "coupler_weight", 200.0)) > 0:
        active.append("joint_couplers")
    if _value(config, "foot_mode", "anchored") == "anchored":
        active.append("anchored_feet")
        velocity_weight = float(_value(config, "foot_velocity_weight", DEFAULT_FOOT_VELOCITY_WEIGHT))
        tracking_weight = float(_value(config, "foot_velocity_tracking_weight", DEFAULT_FOOT_VELOCITY_TRACKING_WEIGHT))
        if velocity_weight > 0 or tracking_weight > 0:
            active.append("stance_foot_velocity")
        if float(_value(config, "stance_height_weight", 0.0)) > 0:
            active.append("annotated_stance_height")
    if _value(config, "nonpen_mode", "scene") != "off":
        active.append("direct_environment_nonpenetration")
    if _value(config, "selfpen_mode", "legs") != "off":
        active.append("direct_self_collision")
    if _value(config, "clearance_mode", "source" if on_terrain else "off") != "off":
        active.append("swing_clearance")
    if _value(config, "sole_offset_mode", "on" if on_terrain else "off") != "off":
        active.append("sole_offsets")
    if _value(config, "foot_orient_mode", "on") != "off":
        active.append("foot_orientation")
    if float(_value(config, "mtp_smooth_weight", 10.0 if on_terrain else 0.0) or 0.0) > 0:
        active.append("mtp_smoothing")
    if float(_value(config, "min_swing_clearance", 0.0)) > 0:
        active.append("minimum_swing_clearance")
    if on_terrain and float(_value(config, "swing_target_lift", 0.06)) > 0:
        active.append("swing_route")
    if bool(_value(config, "seat_contact_active", False)):
        active.append("chair_glute_contact")
    return active


def all_active_qp_terms(config: Mapping[str, object] | SolverConfig, *, on_terrain: bool = True) -> list[str]:
    """List every active base and TERRA-specific solver term."""
    active = ["interaction_mesh"]
    if bool(_value(config, "activate_joint_limits", True)):
        active.append("joint_limits")
    if bool(_value(config, "activate_obj_non_penetration", True)):
        active.append("omniretarget_object_nonpenetration")
    if _value(config, "foot_mode", "anchored") == "omniretarget":
        active.append("omniretarget_foot_sticking")
    if float(_value(config, "trunk_q_diag", 1e-3)) > 0:
        active.append("trunk_q_regularizer")
    return active + active_qp_terms(config, on_terrain=on_terrain)


__all__ = [
    "FROZEN_TERRA_PROFILE_NAME",
    "FROZEN_TERRA_PROFILE_SHA256",
    "FROZEN_TERRA_TERRAIN_PROFILE",
    "MATCHED_TERRA_CORE_OVERRIDES",
    "OMNIRETARGET_PROFILE",
    "TERRA_PROFILE_DEFAULTS",
    "TERRA_SPECIFIC_CONTROLS",
    "MethodProfile",
    "SolverBackend",
    "SolverConfig",
    "active_qp_terms",
    "all_active_qp_terms",
    "resolve_method_profile",
    "resolve_solver_config",
]
