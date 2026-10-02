"""Imitation factory extensions for motion-specific paired terrains."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from loco_mujoco.smpl.retargeting import retargeting_cache_dir
from loco_mujoco.task_factories.dataset_confs import AMASSDatasetConf
from loco_mujoco.task_factories.imitation_factory import ImitationFactory
from terra._methods import validate_method
from terra._musclemimic import TerrainSpec
from terra.terrain.metadata import TerrainMetadata


@dataclasses.dataclass(frozen=True, slots=True)
class _PairedTerrain:
    """One retargeted motion and its matching terrain metadata."""

    motion_path: Path
    terrain_path: Path | None
    metadata: TerrainMetadata


class TerraImitationFactory(ImitationFactory):
    """Load trajectories and retain each motion's own paired terrain.

    Upstream ``ImitationFactory`` intentionally requires exactly one motion
    when ``load_paired_terrain`` is enabled. TERRA validates every requested
    terrain metadata file and constructs a trajectory-indexed terrain whose active geometry
    changes with the reference selected for each vectorized environment.
    """

    @staticmethod
    def _amass_config(value: AMASSDatasetConf | dict | DictConfig) -> AMASSDatasetConf:
        if isinstance(value, AMASSDatasetConf):
            return value
        raw = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else dict(value)
        valid_keys = {field.name for field in dataclasses.fields(AMASSDatasetConf)}
        values = {key: item for key, item in raw.items() if key in valid_keys}
        load_paired_terrain = bool(values.get("load_paired_terrain", False))
        if load_paired_terrain:
            # Upstream only implements its single-motion paired-terrain path for
            # TERRA artifacts. This factory owns the multi-motion implementation
            # below, which is method-independent and validates every sidecar.
            validate_method(str(values.get("retargeting_method", "")))
            values["load_paired_terrain"] = False
        config = AMASSDatasetConf(**values)
        config.load_paired_terrain = load_paired_terrain
        return config

    @staticmethod
    def _configure_reset_mixture(
        env: Any,
        frame_zero_reset_probability: float,
        first_quarter_reset_probability: float = 0.0,
    ) -> Any:
        probability = float(frame_zero_reset_probability)
        first_quarter_probability = float(first_quarter_reset_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("frame_zero_reset_probability must be in [0, 1]")
        if not 0.0 <= first_quarter_probability <= 1.0:
            raise ValueError("first_quarter_reset_probability must be in [0, 1]")
        if probability + first_quarter_probability > 1.0:
            raise ValueError("frame_zero_reset_probability and first_quarter_reset_probability must sum to at most 1")
        env.th.frame_zero_reset_probability = probability
        env.th.first_quarter_reset_probability = first_quarter_probability
        return env

    @staticmethod
    def _configure_motion_groups(env: Any, motions: list[str]) -> Any:
        """Attach stable movement-family ids for hierarchical PPO sampling."""
        known_families = {
            "stairascent": "stair_ascent",
            "stairdescent": "stair_descent",
            "slopeascent": "slope_ascent",
            "slopedescent": "slope_descent",
        }
        motion_families = []
        for motion in motions:
            family = next(
                (known_families[part.casefold()] for part in Path(motion).parts if part.casefold() in known_families),
                "other",
            )
            motion_families.append(family)
        family_names = tuple(sorted(set(motion_families)))
        family_to_id = {name: index for index, name in enumerate(family_names)}
        env.trajectory_group_names = family_names
        env.trajectory_group_ids = tuple(family_to_id[name] for name in motion_families)
        return env

    @classmethod
    def _paired_terrain(
        cls,
        env_name: str,
        config: AMASSDatasetConf,
        motion: str,
    ) -> _PairedTerrain:
        if config.cache_root is None:
            raise ValueError(
                "paired-terrain training requires an explicit amass_dataset_conf.cache_root; "
                "`terra train run` supplies it from TERRA_ARTIFACT_ROOT"
            )
        cache_root = Path(config.cache_root).expanduser()
        cache_dir = retargeting_cache_dir(
            cache_root,
            env_name.replace("Mjx", ""),
            config.retargeting_method,
            config.output_cache_subdir,
        )
        motion_path = cache_dir / f"{motion}.npz"
        terrain_path = motion_path.with_name(f"{motion_path.stem}_terrain.json")
        if not motion_path.is_file():
            raise FileNotFoundError(f"paired retargeted motion not found: {motion_path}")
        if not terrain_path.is_file() and config.require_nonflat_terrain:
            raise FileNotFoundError(f"paired terrain metadata not found: {terrain_path}")
        if terrain_path.is_file():
            metadata = TerrainMetadata.load(terrain_path)
            paired_terrain_path: Path | None = terrain_path
        else:
            metadata = TerrainMetadata.from_terrain(TerrainSpec())
            paired_terrain_path = None
        if config.require_nonflat_terrain and metadata.terrain.is_flat:
            raise ValueError(f"expected non-flat paired terrain, but {terrain_path} is flat")
        return _PairedTerrain(
            motion_path=motion_path,
            terrain_path=paired_terrain_path,
            metadata=metadata,
        )

    @classmethod
    def make(
        cls,
        env_name: str,
        amass_dataset_conf: AMASSDatasetConf | dict | DictConfig | None = None,
        frame_zero_reset_probability: float = 0.0,
        first_quarter_reset_probability: float = 0.0,
        terrain_collision_margin: float = 0.0,
        **kwargs: Any,
    ):
        terrain_collision_margin = float(terrain_collision_margin)
        if not math.isfinite(terrain_collision_margin) or terrain_collision_margin < 0.0:
            raise ValueError("terrain_collision_margin must be finite and non-negative")
        if amass_dataset_conf is None:
            env = super().make(env_name=env_name, amass_dataset_conf=None, **kwargs)
            return cls._configure_reset_mixture(
                env,
                frame_zero_reset_probability,
                first_quarter_reset_probability,
            )

        config = cls._amass_config(amass_dataset_conf)
        if not config.load_paired_terrain:
            env = super().make(env_name=env_name, amass_dataset_conf=config, **kwargs)
            motions = list(cls.get_amass_dataset_paths(config))
            env = cls._configure_motion_groups(env, motions)
            return cls._configure_reset_mixture(
                env,
                frame_zero_reset_probability,
                first_quarter_reset_probability,
            )

        dataset_paths = cls.get_amass_dataset_paths(config)
        if not dataset_paths:
            raise ValueError("load_paired_terrain requires at least one AMASS motion")
        paired_terrains = tuple(cls._paired_terrain(env_name, config, motion) for motion in dataset_paths)

        if "terrain_type" in kwargs or "terrain_params" in kwargs:
            raise ValueError("Paired AMASS terrain cannot be combined with explicit terrain_type/terrain_params.")

        unpaired_config = dataclasses.replace(config, load_paired_terrain=False)
        all_flat = all(paired.metadata.terrain.is_flat for paired in paired_terrains)
        if all_flat:
            if terrain_collision_margin:
                raise ValueError("terrain_collision_margin requires non-flat paired box terrain")
            terrain_type = "StaticTerrain"
            terrain_params = {}
        elif len(paired_terrains) == 1 and not terrain_collision_margin:
            terrain_type = "BoxTerrain"
            terrain_params = paired_terrains[0].metadata.terrain.to_env_params()
        else:
            terrain_type = "PairedBoxTerrain"
            terrain_params = {
                "terrains": [paired.metadata.terrain.to_env_params() for paired in paired_terrains],
                "contact_margin": terrain_collision_margin,
            }
        env = super().make(
            env_name=env_name,
            amass_dataset_conf=unpaired_config,
            terrain_type=terrain_type,
            terrain_params=terrain_params,
            **kwargs,
        )
        if len(paired_terrains) > 1 and int(env.th.n_trajectories) != len(paired_terrains):
            raise ValueError(
                "paired multi-motion training requires exactly one trajectory per motion file; "
                f"loaded {env.th.n_trajectories} trajectories from {len(paired_terrains)} files"
            )
        env.paired_motion_paths = tuple(paired.motion_path for paired in paired_terrains)
        env.paired_terrain_paths = tuple(paired.terrain_path for paired in paired_terrains)
        env.paired_terrain_metadata = tuple(paired.metadata for paired in paired_terrains)
        env.paired_motion_path = env.paired_motion_paths[0]
        env.paired_terrain_path = env.paired_terrain_paths[0]
        env.terrain_metadata = env.paired_terrain_metadata[0]
        motions = list(dataset_paths)
        env = cls._configure_motion_groups(env, motions)
        return cls._configure_reset_mixture(
            env,
            frame_zero_reset_probability,
            first_quarter_reset_probability,
        )


__all__ = ["TerraImitationFactory"]
