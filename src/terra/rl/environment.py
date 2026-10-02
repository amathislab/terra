"""Full-body environments with TERRA's physical observation layout."""

from __future__ import annotations

from collections.abc import Sequence

from mujoco import MjSpec

from loco_mujoco.core import ObservationType
from musclemimic.environments.base import LocoEnv
from musclemimic.environments.humanoids.myofullbody import (
    MjxMyoFullBody as MuscleMimicMjxMyoFullBody,
)
from musclemimic.environments.humanoids.myofullbody import (
    MyoFullBody as MuscleMimicMyoFullBody,
)
from terra.rl.egocentric import HeadingFrameFreeJointVelocity


def _contact_pair_key(pair: Sequence[str]) -> tuple[str, str]:
    if isinstance(pair, (str, bytes)) or len(pair) != 2:
        raise ValueError("each disabled contact pair must contain exactly two geom names")
    first, second = pair
    if not isinstance(first, str) or not first or not isinstance(second, str) or not second:
        raise ValueError("disabled contact-pair geom names must be non-empty strings")
    if first == second:
        raise ValueError("a disabled contact pair must name two different geoms")
    return tuple(sorted((first, second)))


def _normalize_disabled_contact_pairs(
    pairs: Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, str], ...]:
    if pairs is None:
        return ()
    if isinstance(pairs, (str, bytes)):
        raise ValueError("disabled_contact_pairs must be a list of two-geom pairs")
    normalized = tuple(_contact_pair_key(pair) for pair in pairs)
    if len(normalized) != len(set(normalized)):
        raise ValueError("disabled_contact_pairs contains a duplicate pair")
    return normalized


def _delete_explicit_contact_pairs(
    spec: MjSpec,
    pairs: Sequence[tuple[str, str]],
) -> MjSpec:
    """Delete configured explicit pairs before either physics backend compiles."""
    targets = set(pairs)
    if not targets:
        return spec
    removed: set[tuple[str, str]] = set()
    for pair in list(spec.pairs):
        key = _contact_pair_key((pair.geomname1, pair.geomname2))
        if key in targets:
            spec.delete(pair)
            removed.add(key)
    missing = targets - removed
    if missing:
        formatted = ", ".join(f"{first} | {second}" for first, second in sorted(missing))
        raise ValueError(f"configured explicit contact pair does not exist: {formatted}")
    return spec


class _TerraObservationLayout:
    """Build the proprioceptive and terrain observations in a fixed order."""

    def __init__(
        self,
        *args,
        use_egocentric_root_observations: bool = False,
        disabled_contact_pairs: Sequence[Sequence[str]] | None = None,
        **kwargs,
    ) -> None:
        self._use_egocentric_root_observations = bool(use_egocentric_root_observations)
        self._disabled_contact_pairs = _normalize_disabled_contact_pairs(disabled_contact_pairs)
        super().__init__(*args, **kwargs)

    def _apply_spec_changes(self, spec: MjSpec) -> MjSpec:
        spec = super()._apply_spec_changes(spec)
        return _delete_explicit_contact_pairs(spec, self._disabled_contact_pairs)

    def _get_observation_specification(self, spec: MjSpec) -> list[ObservationType]:
        joint_names = [joint.name for joint in spec.joints if joint.name != self.root_free_joint_xml_name]
        observations = []

        if self._enable_joint_pos_observations:
            if self._use_egocentric_root_observations:
                observations.extend(
                    [
                        ObservationType.EntryFromFreeJointPos(
                            entry_index=2,
                            obs_name="q_root_height",
                            xml_name=self.root_free_joint_xml_name,
                        ),
                        ObservationType.ProjectedGravityVector(
                            "projected_gravity",
                            self.root_free_joint_xml_name,
                        ),
                    ]
                )
            else:
                root_position = (
                    ObservationType.FreeJointPos
                    if self._enable_global_root_position_observation
                    else ObservationType.FreeJointPosNoXY
                )
                observations.append(root_position("q_free_joint", self.root_free_joint_xml_name))
            observations.append(ObservationType.JointPosArray("q_all_pos", joint_names))

        if self._enable_joint_vel_observations:
            if self._use_egocentric_root_observations:
                observations.append(
                    HeadingFrameFreeJointVelocity(
                        "dq_free_joint_heading",
                        self.root_free_joint_xml_name,
                    )
                )
            else:
                observations.append(ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name))
            observations.append(ObservationType.JointVelArray("dq_all_vel", joint_names))

        actuator_observations = (
            (self._enable_muscle_length_observations, "muscle_length", ObservationType.ActuatorLength),
            (self._enable_muscle_velocity_observations, "muscle_velocity", ObservationType.ActuatorVelocity),
            (self._enable_muscle_force_observations, "muscle_force", ObservationType.ActuatorForce),
            (self._enable_muscle_excitation_observations, "muscle_excitation", ObservationType.ActuatorExcitation),
            (self._enable_muscle_activation_observations, "muscle_activation", ObservationType.ActuatorActivation),
        )
        for actuator in spec.actuators:
            for enabled, prefix, observation_type in actuator_observations:
                if enabled:
                    observations.append(observation_type(f"{prefix}_{actuator.name.lower()}", xml_name=actuator.name))

        if self._enable_touch_sensor_observations:
            observations.extend(
                ObservationType.TouchSensor(f"touch_{name}", xml_name=name)
                for name in ("r_foot", "r_toes", "l_foot", "l_toes")
            )

        if self._enable_heightmap_observations:
            observations.append(
                ObservationType.HeightMatrix(
                    "terrain_heightmap",
                    grid_rows=self._heightmap_grid_rows,
                    grid_cols=self._heightmap_grid_cols,
                    grid_resolution=self._heightmap_grid_resolution,
                    grid_forward_offset=self._heightmap_grid_forward_offset,
                    body_name=self._heightmap_body_name,
                    allow_randomization=False,
                )
            )

        return observations


class MyoFullBody(_TerraObservationLayout, MuscleMimicMyoFullBody):
    """CPU MuJoCo environment used for evaluation and rendering."""


class MjxMyoFullBody(_TerraObservationLayout, MuscleMimicMjxMyoFullBody):
    """MJX environment used for policy training and backend parity checks."""

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """Preserve TERRA's body--terrain contacts on the MJX-JAX backend.

        Upstream MyoFullBody disables automatic contacts for every non-ground
        geometry when using MJX-JAX. That removes the body side of TERRA's
        automatic body--terrain contact pairs and is therefore not a valid
        implementation of this environment. Warp already preserves the full
        contact model and still needs upstream's static contact budgets.
        """
        if self.mjx_backend == "jax":
            return spec
        return super()._modify_spec_for_mjx(spec)


def register_environments() -> None:
    """Install the local environment classes under the canonical cache names."""

    LocoEnv.registered_envs["MyoFullBody"] = MyoFullBody
    LocoEnv.registered_envs["MjxMyoFullBody"] = MjxMyoFullBody


__all__ = ["MjxMyoFullBody", "MyoFullBody", "register_environments"]
