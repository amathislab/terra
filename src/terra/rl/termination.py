"""Termination rules for faithful global motion tracking."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import jax.numpy as jnp
import numpy as np

from loco_mujoco.core.utils.mujoco import mj_jntname2qposid
from musclemimic.core.terminal_state_handler.enhanced_fullbody import (
    MeanSiteDeviationTerminalStateHandler,
)

DEFAULT_CORE_UPPER_BODY_SITES = (
    "upper_body_mimic",
    "head_mimic",
    "left_shoulder_mimic",
    "right_shoulder_mimic",
)


class TerraGlobalMPJPETerminalStateHandler(MeanSiteDeviationTerminalStateHandler):
    """Terminate on global/core keypoint MPJPE or excessive root rotation.

    The position criterion is the mean Euclidean distance between the current
    and reference mimic sites in world coordinates.  Unlike the generic
    MuscleMimic implementation, reference XY is shifted to an episode-local
    origin only when the environment does not preserve trajectory root XY.

    A second mean over the lumbar, head, and shoulder sites prevents accurate
    legs from hiding a collapsed torso in the global mean.
    """

    def __init__(
        self,
        env: Any,
        mean_site_deviation_threshold: float = 0.15,
        core_upper_body_mean_site_deviation_threshold: float | None = 0.15,
        core_upper_body_uses_global_threshold: bool = False,
        core_upper_body_curriculum_initial_threshold: float | None = None,
        curriculum_initial_global_threshold: float | None = None,
        core_upper_body_sites: list[str] | None = None,
        root_orientation_threshold: float | None = 1.0,
        enable_site_check: bool = True,
        exclude_sites: list[str] | None = None,
        **handler_config: dict[str, Any],
    ):
        super().__init__(
            env,
            mean_site_deviation_threshold=mean_site_deviation_threshold,
            enable_site_check=enable_site_check,
            exclude_sites=exclude_sites,
            **handler_config,
        )
        self.core_upper_body_mean_site_deviation_threshold = (
            None
            if core_upper_body_mean_site_deviation_threshold is None
            else float(core_upper_body_mean_site_deviation_threshold)
        )
        self.core_upper_body_uses_global_threshold = bool(core_upper_body_uses_global_threshold)
        self.core_upper_body_curriculum_initial_threshold = (
            None
            if core_upper_body_curriculum_initial_threshold is None
            else float(core_upper_body_curriculum_initial_threshold)
        )
        self.curriculum_initial_global_threshold = (
            None
            if curriculum_initial_global_threshold is None
            else float(curriculum_initial_global_threshold)
        )
        if (
            self.core_upper_body_mean_site_deviation_threshold is not None
            and self.core_upper_body_mean_site_deviation_threshold <= 0.0
        ):
            raise ValueError("core_upper_body_mean_site_deviation_threshold must be positive or null")
        if self.core_upper_body_uses_global_threshold:
            if self.core_upper_body_curriculum_initial_threshold is None:
                raise ValueError(
                    "core_upper_body_curriculum_initial_threshold is required when "
                    "core upper-body termination uses the curriculum"
                )
            if self.curriculum_initial_global_threshold is None:
                raise ValueError(
                    "curriculum_initial_global_threshold is required when "
                    "core upper-body termination uses the curriculum"
                )
            if self.core_upper_body_mean_site_deviation_threshold is None:
                raise ValueError(
                    "core_upper_body_mean_site_deviation_threshold is required when "
                    "core upper-body termination uses the curriculum"
                )
            if (
                self.core_upper_body_curriculum_initial_threshold
                < self.core_upper_body_mean_site_deviation_threshold
            ):
                raise ValueError(
                    "core upper-body curriculum initial threshold must be at least its final threshold"
                )
            if (
                self.curriculum_initial_global_threshold
                < self.mean_site_deviation_threshold
            ):
                raise ValueError(
                    "global curriculum initial threshold must be at least its final threshold"
                )

        configured_core_sites = tuple(core_upper_body_sites or DEFAULT_CORE_UPPER_BODY_SITES)
        sites_for_mimic = []
        if hasattr(env, "_goal") and hasattr(env._goal, "_info_props"):
            sites_for_mimic = env._goal._info_props.get("sites_for_mimic", [])
        if not sites_for_mimic:
            sites_for_mimic = getattr(env, "sites_for_mimic", [])

        missing_core_sites = [name for name in configured_core_sites if name not in sites_for_mimic]
        if self.core_upper_body_mean_site_deviation_threshold is not None and missing_core_sites:
            raise ValueError(
                "core upper-body termination sites are missing from sites_for_mimic: " + ", ".join(missing_core_sites)
            )
        self.core_upper_body_sites = configured_core_sites
        self._core_upper_body_indices = np.asarray(
            [sites_for_mimic.index(name) for name in configured_core_sites if name in sites_for_mimic],
            dtype=int,
        )
        self.root_orientation_threshold = root_orientation_threshold
        self._root_qpos_ids_quat: np.ndarray | None = None

        root_joint_name = self._info_props.get("root_free_joint_xml_name")
        model = getattr(env, "_model", None)
        if root_joint_name and model is not None:
            try:
                root_qpos_ids = np.asarray(mj_jntname2qposid(root_joint_name, model), dtype=int)
                if root_qpos_ids.size >= 7:
                    self._root_qpos_ids_quat = root_qpos_ids[3:7]
            except (ValueError, TypeError, IndexError):
                self._root_qpos_ids_quat = None

    def _site_deviations(
        self,
        env: Any,
        state: Any,
        carry: Any,
        backend: ModuleType,
    ) -> Any | None:
        """Return global mimic-site errors in the environment's world frame."""
        if not self.enable_site_check or not hasattr(env, "th") or env.th is None:
            return None
        if not (hasattr(env, "_goal") and hasattr(env._goal, "_rel_site_ids")):
            return None
        if not hasattr(state, "site_xpos"):
            return None

        ref_data = env.th.get_current_traj_data(carry, backend)
        if not hasattr(ref_data, "site_xpos"):
            return None

        site_mapping = env._goal._rel_site_ids
        current_mapped_sites = state.site_xpos[site_mapping]
        if env._goal._site_mapper.requires_mapping:
            traj_indices = env._goal._site_mapper.model_ids_to_traj_indices(site_mapping)
            ref_mapped_sites = ref_data.site_xpos[traj_indices]
        else:
            ref_mapped_sites = ref_data.site_xpos

        if (
            not getattr(env, "preserve_trajectory_root_xy", False)
            and self._root_qpos_ids_xy is not None
            and hasattr(env.th, "get_init_traj_data")
        ):
            init_ref = env.th.get_init_traj_data(carry, backend)
            if hasattr(init_ref, "qpos"):
                root_xy = init_ref.qpos[self._root_qpos_ids_xy]
                offset = backend.concatenate((root_xy, backend.zeros(1, dtype=root_xy.dtype)))
                ref_mapped_sites = ref_mapped_sites - offset

        return backend.linalg.norm(current_mapped_sites - ref_mapped_sites, axis=-1)

    def _check_mean_site_deviation(
        self,
        env: Any,
        state: Any,
        carry: Any,
        backend: ModuleType,
    ) -> bool | jnp.ndarray:
        """Check mean global mimic-site error."""
        site_deviations = self._site_deviations(env, state, carry, backend)
        if site_deviations is None:
            return backend.asarray(False)
        return self._global_mean_site_deviation_exceeded(site_deviations, carry, backend)

    def _global_mean_site_deviation_exceeded(
        self,
        site_deviations: Any,
        carry: Any,
        backend: ModuleType,
    ) -> bool | jnp.ndarray:
        if self._has_exclusions:
            if self._n_included == 0:
                return backend.asarray(False)
            site_deviations = backend.take(site_deviations, self._include_indices, axis=0)

        mean_deviation = backend.mean(site_deviations)
        return backend.logical_or(
            backend.logical_not(backend.isfinite(mean_deviation)),
            backend.greater(mean_deviation, carry.termination_threshold),
        )

    def _check_core_upper_body_mean_site_deviation(
        self,
        env: Any,
        state: Any,
        carry: Any,
        backend: ModuleType,
    ) -> bool | jnp.ndarray:
        """Check the core upper-body mean without lower-body dilution."""
        threshold = getattr(self, "core_upper_body_mean_site_deviation_threshold", None)
        indices = getattr(self, "_core_upper_body_indices", np.asarray([], dtype=int))
        if threshold is None or indices.size == 0:
            return backend.asarray(False)
        site_deviations = self._site_deviations(env, state, carry, backend)
        if site_deviations is None:
            return backend.asarray(False)
        return self._core_upper_body_mean_site_deviation_exceeded(site_deviations, carry, backend)

    def _core_upper_body_mean_site_deviation_exceeded(
        self,
        site_deviations: Any,
        carry: Any,
        backend: ModuleType,
    ) -> bool | jnp.ndarray:
        if getattr(self, "core_upper_body_uses_global_threshold", False):
            global_final = self.mean_site_deviation_threshold
            global_initial = self.curriculum_initial_global_threshold
            core_final = self.core_upper_body_mean_site_deviation_threshold
            core_initial = self.core_upper_body_curriculum_initial_threshold
            progress = backend.clip(
                (carry.termination_threshold - global_final)
                / max(global_initial - global_final, 1e-8),
                0.0,
                1.0,
            )
            threshold = core_final + progress * (core_initial - core_final)
        else:
            threshold = self.core_upper_body_mean_site_deviation_threshold
        indices = self._core_upper_body_indices
        core_deviations = backend.take(site_deviations, indices, axis=0)
        mean_deviation = backend.mean(core_deviations)
        return backend.logical_or(
            backend.logical_not(backend.isfinite(mean_deviation)),
            backend.greater(mean_deviation, threshold),
        )

    def _check_root_orientation(
        self,
        env: Any,
        state: Any,
        carry: Any,
        backend: ModuleType,
    ) -> bool | jnp.ndarray:
        """Check root geodesic orientation error in radians."""
        if self.root_orientation_threshold is None or self._root_qpos_ids_quat is None:
            return backend.asarray(False)
        if not hasattr(env, "th") or env.th is None or not hasattr(state, "qpos"):
            return backend.asarray(False)

        ref_data = env.th.get_current_traj_data(carry, backend)
        if not hasattr(ref_data, "qpos"):
            return backend.asarray(False)

        current_quat = state.qpos[self._root_qpos_ids_quat]
        reference_quat = ref_data.qpos[self._root_qpos_ids_quat]
        current_quat = current_quat / backend.linalg.norm(current_quat)
        reference_quat = reference_quat / backend.linalg.norm(reference_quat)
        dot = backend.abs(backend.dot(current_quat, reference_quat))
        angular_distance = 2.0 * backend.arccos(backend.clip(dot, 0.0, 1.0))
        return backend.logical_or(
            backend.logical_not(backend.isfinite(angular_distance)),
            backend.greater(angular_distance, self.root_orientation_threshold),
        )

    def _is_absorbing_compat(
        self,
        env: Any,
        obs: np.ndarray | jnp.ndarray,
        info: dict[str, Any],
        data: Any,
        carry: Any,
        backend: ModuleType,
    ) -> tuple[bool | jnp.ndarray, Any]:
        site_deviations = self._site_deviations(env, data, carry, backend)
        if site_deviations is None:
            site_violation = backend.asarray(False)
            core_upper_body_violation = backend.asarray(False)
        else:
            site_violation = self._global_mean_site_deviation_exceeded(site_deviations, carry, backend)
            threshold = getattr(self, "core_upper_body_mean_site_deviation_threshold", None)
            indices = getattr(self, "_core_upper_body_indices", np.asarray([], dtype=int))
            core_upper_body_violation = (
                backend.asarray(False)
                if threshold is None or indices.size == 0
                else self._core_upper_body_mean_site_deviation_exceeded(site_deviations, carry, backend)
            )
        orientation_violation = self._check_root_orientation(env, data, carry, backend)
        position_violation = backend.logical_or(site_violation, core_upper_body_violation)
        return backend.logical_or(position_violation, orientation_violation), carry


__all__ = ["DEFAULT_CORE_UPPER_BODY_SITES", "TerraGlobalMPJPETerminalStateHandler"]
