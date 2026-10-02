"""Trajectory-indexed terrain for paired multi-motion imitation."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import MjData, MjModel, MjSpec
from mujoco.mjx import Data, Model

from loco_mujoco.core.terrain import Terrain, TerrainSpec
from loco_mujoco.core.utils.backend import assert_backend_is_supported
from terra.rl.egocentric import rotation_matrix_from_quaternion


class PairedBoxTerrain(Terrain):
    """Select one box terrain using the episode's trajectory index.

    The MuJoCo topology is fixed at the largest box count in the dataset. At
    reset/step time the selected trajectory's box positions, rotations, and
    sizes are written into those slots. Unused slots are moved far below the
    world. This makes each lane of the vectorized MJX environment collide with
    and observe the terrain paired with its own reference trajectory.
    """

    GEOM_PREFIX = "terrain_box"
    _INACTIVE_Z = -1000.0

    def __init__(
        self,
        env: Any,
        terrains: Sequence[dict],
        rgba=(0.55, 0.45, 0.35, 1.0),
        contact_margin: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(env, **kwargs)
        if not terrains:
            raise ValueError("PairedBoxTerrain requires at least one terrain")
        self.contact_margin = float(contact_margin)
        if not np.isfinite(self.contact_margin) or self.contact_margin < 0.0:
            raise ValueError("contact_margin must be finite and non-negative")
        self.rgba = tuple(float(value) for value in rgba)
        self.specs = tuple(TerrainSpec(boxes=tuple(value.get("boxes", ()))) for value in terrains)
        self.max_boxes = max(len(spec.boxes) for spec in self.specs)
        if self.max_boxes < 1:
            raise ValueError("PairedBoxTerrain requires at least one non-flat terrain box")

        n_terrains = len(self.specs)
        self._pos = np.zeros((n_terrains, self.max_boxes, 3), dtype=np.float32)
        self._pos[..., 2] = self._INACTIVE_Z
        self._size = np.full((n_terrains, self.max_boxes, 3), 1e-4, dtype=np.float32)
        self._quat = np.zeros((n_terrains, self.max_boxes, 4), dtype=np.float32)
        self._quat[..., 0] = 1.0
        self._yaw = np.zeros((n_terrains, self.max_boxes), dtype=np.float32)
        self._pitch = np.zeros((n_terrains, self.max_boxes), dtype=np.float32)
        self._active = np.zeros((n_terrains, self.max_boxes), dtype=bool)
        for terrain_index, spec in enumerate(self.specs):
            for box_index, box in enumerate(spec.boxes):
                self._pos[terrain_index, box_index] = box.pos
                self._size[terrain_index, box_index] = box.size
                self._quat[terrain_index, box_index] = box.quat
                self._yaw[terrain_index, box_index] = box.yaw
                self._pitch[terrain_index, box_index] = box.pitch
                self._active[terrain_index, box_index] = True
        self._geom_ids: np.ndarray | None = None

    def modify_spec(self, spec: MjSpec) -> MjSpec:
        worldbody = spec.worldbody
        for box_index in range(self.max_boxes):
            worldbody.add_geom(
                name=f"{self.GEOM_PREFIX}_{box_index}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=self._pos[0, box_index],
                quat=self._quat[0, box_index],
                size=self._size[0, box_index],
                rgba=self.rgba,
                friction=(1.0, 0.1, 0.1),
                condim=3,
                contype=1,
                conaffinity=1,
                margin=self.contact_margin,
                group=2,
            )
        return spec

    def _resolve_geom_ids(self, env: Any) -> np.ndarray:
        if self._geom_ids is None:
            ids = [
                mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.GEOM_PREFIX}_{index}")
                for index in range(self.max_boxes)
            ]
            if any(value < 0 for value in ids):
                raise RuntimeError("PairedBoxTerrain geometry slots are missing from the compiled model")
            self._geom_ids = np.asarray(ids, dtype=np.int32)
        return self._geom_ids

    def _terrain_index(self, carry: Any, backend: ModuleType):
        index = carry.traj_state.traj_no
        if backend == jnp:
            return backend.asarray(index, dtype=backend.int32)
        return int(index)

    def reset(self, env: Any, model: MjModel | Model, data: MjData | Data, carry: Any, backend: ModuleType):
        assert_backend_is_supported(backend)
        return data, carry

    def update(
        self,
        env: Any,
        model: MjModel | Model,
        data: MjData | Data,
        carry: Any,
        backend: ModuleType,
    ):
        assert_backend_is_supported(backend)
        geom_ids = self._resolve_geom_ids(env)
        terrain_index = self._terrain_index(carry, backend)
        positions = backend.asarray(self._pos)[terrain_index]
        sizes = backend.asarray(self._size)[terrain_index]
        quaternions = backend.asarray(self._quat)[terrain_index]
        rotation_matrices = rotation_matrix_from_quaternion(quaternions, backend)
        # MuJoCo and standard MJX expose ``geom_aabb`` as (ngeom, 6), while
        # MJX-Warp exposes the same center/half-size data as (ngeom, 2, 3).
        local_aabbs = (
            backend.stack((backend.zeros_like(sizes), sizes), axis=-2)
            if model.geom_aabb.ndim == 3
            else backend.concatenate((backend.zeros_like(sizes), sizes), axis=-1)
        )
        bounding_radii = backend.linalg.norm(sizes, axis=-1)

        if backend == jnp:
            ids = backend.asarray(geom_ids)
            model = model.tree_replace(
                {
                    "geom_pos": model.geom_pos.at[ids].set(positions),
                    "geom_size": model.geom_size.at[ids].set(sizes),
                    "geom_quat": model.geom_quat.at[ids].set(quaternions),
                    "geom_aabb": model.geom_aabb.at[ids].set(local_aabbs),
                    "geom_rbound": model.geom_rbound.at[ids].set(bounding_radii),
                }
            )
            if data is not None:
                # MJX-Warp deliberately treats world-body geoms as static and
                # does not recompute their world transforms in kinematics.
                # Paired terrain slots are world geoms whose model-space pose
                # changes by trajectory, so update the collision transforms
                # explicitly alongside the model.  Without this, Warp keeps
                # colliding against terrain zero while observations describe
                # the selected trajectory's terrain.
                geom_xmat = rotation_matrices
                if data.geom_xmat.ndim == 2:
                    geom_xmat = geom_xmat.reshape((geom_xmat.shape[0], -1))
                data = data.tree_replace(
                    {
                        "geom_xpos": data.geom_xpos.at[ids].set(positions),
                        "geom_xmat": data.geom_xmat.at[ids].set(geom_xmat),
                    }
                )
        else:
            model.geom_pos[geom_ids] = positions
            model.geom_size[geom_ids] = sizes
            model.geom_quat[geom_ids] = quaternions
            model.geom_aabb[geom_ids] = local_aabbs
            model.geom_rbound[geom_ids] = bounding_radii
            if data is not None:
                data.geom_xpos[geom_ids] = positions
                data.geom_xmat[geom_ids] = rotation_matrices.reshape((len(geom_ids), -1))
        return model, data, carry

    @property
    def is_dynamic(self) -> bool:
        # ``True`` is reserved by the current viewer for mutable heightfields.
        # The environment invokes update() for this terrain on every step.
        return False

    def sample_heights_at_points(
        self,
        x: np.ndarray | jnp.ndarray,
        y: np.ndarray | jnp.ndarray,
        model: MjModel | Model,
        carry: Any,
        backend: ModuleType,
    ):
        assert_backend_is_supported(backend)
        terrain_index = self._terrain_index(carry, backend)
        pos = backend.asarray(self._pos)[terrain_index]
        size = backend.asarray(self._size)[terrain_index]
        yaw = backend.asarray(self._yaw)[terrain_index]
        pitch = backend.asarray(self._pitch)[terrain_index]
        active = backend.asarray(self._active)[terrain_index]

        dx = x[..., None] - pos[:, 0]
        dy = y[..., None] - pos[:, 1]
        cy, sy = backend.cos(yaw), backend.sin(yaw)
        cp, sp = backend.cos(pitch), backend.sin(pitch)
        q = cy * dx + sy * dy
        lo = backend.full(dx.shape, -backend.inf)
        hi = backend.full(dx.shape, backend.inf)
        for axis_offset, direction, half_extent in (
            (cp * q + sp * pos[:, 2], -sp, size[:, 0]),
            (sp * q - cp * pos[:, 2], cp, size[:, 2]),
        ):
            safe_direction = backend.where(backend.abs(direction) < 1e-12, 1.0, direction)
            entry_0 = (-half_extent - axis_offset) / safe_direction
            entry_1 = (half_extent - axis_offset) / safe_direction
            flat = backend.abs(direction) < 1e-12
            miss = flat & (backend.abs(axis_offset) > half_extent)
            lo = backend.where(
                flat,
                backend.where(miss, backend.inf, lo),
                backend.maximum(lo, backend.minimum(entry_0, entry_1)),
            )
            hi = backend.where(
                flat,
                backend.where(miss, -backend.inf, hi),
                backend.minimum(hi, backend.maximum(entry_0, entry_1)),
            )
        lateral = backend.abs(-sy * dx + cy * dy) <= size[:, 1]
        heights = backend.where(active & lateral & (lo <= hi), hi, 0.0)
        return backend.max(heights, axis=-1)


__all__ = ["PairedBoxTerrain"]
