"""TERRA policy rewards."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
from flax import struct

from musclemimic.core.reward.trajectory_based import MimicReward, MimicRewardState
from terra.rl.termination import DEFAULT_CORE_UPPER_BODY_SITES


@struct.dataclass
class TerraRewardState(MimicRewardState):
    """Reserved arrays preserving the compiled checkpoint rollout state layout."""

    emg_count: Any = None
    emg_prediction_mean: Any = None
    emg_target_mean: Any = None
    emg_prediction_m2: Any = None
    emg_target_m2: Any = None
    emg_co_moment: Any = None


class TerraReward(MimicReward):
    """Track global locomotion while measuring articulated pose without the free root."""

    def __init__(self, env: Any, **parameters: Any):
        self._use_dynamic_velocity_reward_weights = bool(parameters.pop("use_dynamic_velocity_reward_weights", False))
        self._activation_floor = float(parameters.pop("activation_floor", 0.0))
        self._activation_floor_coeff = float(parameters.pop("activation_floor_coeff", 0.0))
        self._core_upper_body_position_reward_weight = float(
            parameters.pop("core_upper_body_position_reward_weight", 1.0)
        )
        self._core_upper_body_position_reward_exp = float(parameters.pop("core_upper_body_position_reward_exp", 40.0))
        self._terminal_quality_bonus_weight = float(parameters.pop("terminal_quality_bonus_weight", 25.0))
        core_sites = tuple(parameters.pop("core_upper_body_sites", DEFAULT_CORE_UPPER_BODY_SITES))
        parameters.setdefault("global_root_tracking", True)
        parameters.setdefault("joint_only_qpos_qvel", True)
        parameters.setdefault("root_velocity_frame", "global")
        super().__init__(env, **parameters)

        site_names = list(self._info_props["sites_for_mimic"])
        required_sites = ("pelvis_mimic", *core_sites)
        missing_sites = [name for name in required_sites if name not in site_names]
        if missing_sites:
            raise ValueError("upper-body reward sites are missing from sites_for_mimic: " + ", ".join(missing_sites))
        self._pelvis_site_index = site_names.index("pelvis_mimic")
        self._core_upper_body_site_indices = np.asarray(
            [site_names.index(name) for name in core_sites],
            dtype=int,
        )

        for name, value in (
            ("activation_floor_coeff", self._activation_floor_coeff),
            ("core_upper_body_position_reward_weight", self._core_upper_body_position_reward_weight),
            ("core_upper_body_position_reward_exp", self._core_upper_body_position_reward_exp),
            ("terminal_quality_bonus_weight", self._terminal_quality_bonus_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self._activation_floor <= 1.0:
            raise ValueError("activation_floor must lie in [0, 1]")

    def init_state(
        self,
        env: Any,
        key: Any,
        model: Any,
        data: Any,
        backend: ModuleType,
    ) -> TerraRewardState:
        """Initialize checkpoint-compatible zero state."""

        base = super().init_state(env, key, model, data, backend)
        zeros = backend.zeros(12, dtype=backend.float32)
        return TerraRewardState(
            last_qvel=base.last_qvel,
            last_action=base.last_action,
            imitation_error_total=base.imitation_error_total,
            emg_count=zeros,
            emg_prediction_mean=zeros,
            emg_target_mean=zeros,
            emg_prediction_m2=zeros,
            emg_target_m2=zeros,
            emg_co_moment=zeros,
        )

    def _core_upper_body_tracking(self, env: Any, data: Any, carry: Any, backend: ModuleType):
        """Return pelvis-relative upper-body quality and MPJPE."""

        reference = env.th.get_current_traj_data(carry, backend)
        current_sites = data.site_xpos[self._rel_site_ids]
        if self._site_mapper.requires_mapping:
            trajectory_indices = self._site_mapper.model_ids_to_traj_indices(self._rel_site_ids)
            target_sites = reference.site_xpos[trajectory_indices]
        else:
            target_sites = reference.site_xpos[self._rel_site_ids]

        pelvis_index = self._pelvis_site_index
        core_indices = self._core_upper_body_site_indices
        current_relative = current_sites[core_indices] - current_sites[pelvis_index]
        target_relative = target_sites[core_indices] - target_sites[pelvis_index]
        error = current_relative - target_relative
        squared_distance = backend.sum(backend.square(error), axis=-1)
        mean_squared_distance = backend.mean(squared_distance)
        mpjpe = backend.mean(backend.linalg.norm(error, axis=-1))
        quality = backend.exp(-self._core_upper_body_position_reward_exp * mean_squared_distance)
        return quality, mpjpe

    @staticmethod
    def _sanitize_reward_outputs(
        reward: Any,
        reward_info: dict[str, Any],
        backend: ModuleType,
    ) -> tuple[Any, dict[str, Any]]:
        """Keep one invalid simulator transition from poisoning training state.

        ``MimicReward`` sanitizes its scalar total, but TERRA adds upper-body
        and terminal terms afterwards.  A rare nonfinite MJX state can therefore
        reintroduce NaN here.  Preserve an explicit marker for the environment
        boundary, then make every floating reward output safe for CPU and MJX
        callers alike.
        """

        nonfinite = backend.logical_not(backend.all(backend.isfinite(backend.asarray(reward))))
        sanitized: dict[str, Any] = {}
        for name, value in reward_info.items():
            array = backend.asarray(value)
            if np.issubdtype(np.dtype(array.dtype), np.inexact):
                nonfinite = backend.logical_or(
                    nonfinite,
                    backend.logical_not(backend.all(backend.isfinite(array))),
                )
                value = backend.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
            sanitized[name] = value

        sanitized["numerics_nonfinite_reward"] = nonfinite
        safe_reward = backend.nan_to_num(
            backend.asarray(reward),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return safe_reward, sanitized

    def __call__(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        absorbing: Any,
        info: dict[str, Any],
        env: Any,
        model: Any,
        data: Any,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Track motion with fixed velocity weights, effort costs, and terminal quality.

        Dynamic velocity weights can instead be supplied by the reward curriculum.
        """

        if not self._use_dynamic_velocity_reward_weights:
            carry = carry.replace(
                qvel_w_sum=backend.asarray(self._qvel_w_sum, dtype=carry.qvel_w_sum.dtype),
                root_vel_w_sum=backend.asarray(self._root_vel_w_sum, dtype=carry.root_vel_w_sum.dtype),
            )

        reward, carry, reward_info = super().__call__(
            state,
            action,
            next_state,
            absorbing,
            info,
            env,
            model,
            data,
            carry,
            backend,
        )
        activation = data.act
        zero = backend.asarray(0.0)
        raw_activation_energy = backend.mean(backend.square(activation)) if activation.size else zero
        if activation.size and self._activation_floor > 0.0:
            floor = backend.asarray(self._activation_floor, dtype=activation.dtype)
            normalized_shortfall = backend.maximum(floor - activation, 0.0) / floor
            raw_activation_floor_violation = backend.mean(backend.square(normalized_shortfall))
            below_floor = backend.asarray(activation < floor, dtype=activation.dtype)
            raw_activation_below_floor_fraction = backend.mean(below_floor)
        else:
            raw_activation_floor_violation = zero
            raw_activation_below_floor_fraction = zero
        activation_floor_penalty = -self._activation_floor_coeff * raw_activation_floor_violation
        core_quality, core_mpjpe = self._core_upper_body_tracking(env, data, carry, backend)
        core_reward = self._core_upper_body_position_reward_weight * core_quality
        # Keep the zero-valued fields and operation order used by saved checkpoints.
        emg_reward = backend.asarray(0.0, dtype=backend.float32)
        emg_info = {
            "emg_correlation_raw": emg_reward,
            "emg_correlation_defined": emg_reward,
            "emg_supervision_active": emg_reward,
            "emg_target_channel_count": emg_reward,
            "emg_prediction_std_raw": emg_reward,
            "emg_tracking_gate": emg_reward,
        }
        reached_trajectory_end = env.th.reached_trajectory_end(carry.traj_state, backend)
        successful_trajectory_end = backend.logical_and(
            reached_trajectory_end,
            backend.logical_not(absorbing),
        )
        terminal_quality_bonus = self._terminal_quality_bonus_weight * core_quality * successful_trajectory_end
        reward = backend.maximum(
            reward + core_reward + terminal_quality_bonus + activation_floor_penalty + emg_reward,
            0.0,
        )
        base_activation_penalty = reward_info.get("penalty_activation_energy", zero)
        reward_info["penalty_total"] = reward_info.get("penalty_total", zero) + activation_floor_penalty
        reward_info["reward_total"] = reward
        reward_info["activation_energy_raw"] = raw_activation_energy
        reward_info["activation_floor_violation_raw"] = raw_activation_floor_violation
        reward_info["activation_below_floor_fraction_raw"] = raw_activation_below_floor_fraction
        reward_info["penalty_activation_floor"] = activation_floor_penalty
        reward_info["penalty_activity_regularization"] = base_activation_penalty + activation_floor_penalty
        reward_info["reward_core_upper_body"] = core_reward
        reward_info["err_core_upper_body_relative"] = core_mpjpe
        reward_info["reward_terminal_quality_bonus"] = terminal_quality_bonus
        reward_info["reward_emg_correlation"] = emg_reward
        reward_info.update(emg_info)
        reward, reward_info = self._sanitize_reward_outputs(reward, reward_info, backend)
        return reward, carry, reward_info


__all__ = ["TerraReward"]
