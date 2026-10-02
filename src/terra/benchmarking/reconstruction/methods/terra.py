"""Fit terrain from normalized SMPL-H motion landmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from terra.dataset_pipeline import (
    DatasetConfig,
    MotionRecord,
    ensure_robot_shape,
    fit_record_terrain,
    load_dataset_config,
    load_motion_records,
)
from terra.terrain import resolve_terra_reconstruction_profile

from ..core import PreparedMotion, ReconstructionResult

_PROFILE_DISPLAY_NAMES = {
    "no-physical-cues": "TERRA w/o physical cues",
    "full": "TERRA",
}


@dataclass(frozen=True)
class TerraMethod:
    dataset_config_path: Path
    profile_name: str = "full"
    cache_root: Path | None = None
    model_root: Path | None = None
    motion_paths: tuple[Path, ...] = ()
    _shape_ready: bool = field(default=False, init=False, repr=False, compare=False)
    _config: DatasetConfig = field(init=False, repr=False, compare=False)
    _records: dict[str, MotionRecord] = field(init=False, repr=False, compare=False)

    display_name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self) -> None:
        profile = resolve_terra_reconstruction_profile(self.profile_name)
        object.__setattr__(self, "profile_name", profile.cli_name)
        config = load_dataset_config(self.dataset_config_path)
        if self.cache_root is not None:
            config = replace(config, cache_root=self.cache_root.expanduser().resolve())
        if self.model_root is not None:
            config = replace(config, smpl_model_path=self.model_root.expanduser().resolve())
        if config.terrain_mode != "fit":
            raise ValueError(f"dataset {config.name} does not enable terrain fitting")
        object.__setattr__(self, "_config", config)
        if self.motion_paths:
            if config.manifest_path is not None:
                raise ValueError(
                    "--motion requires a dataset without a conversion manifest; use --motions for converted datasets"
                )
            records = []
            for source in self.motion_paths:
                source = source.expanduser().resolve()
                if source.suffix != ".npz" or not source.is_file():
                    raise ValueError(f"motion must be an existing SMPL-H .npz: {source}")
                relative = source.relative_to(config.input_root)
                records.append(MotionRecord(relative.with_suffix("").as_posix(), config.name, source))
        else:
            records = load_motion_records(config)
        object.__setattr__(self, "_records", {record.motion: record for record in records})
        object.__setattr__(self, "display_name", _PROFILE_DISPLAY_NAMES[self.profile_name])
        object.__setattr__(self, "description", profile.description)

    @property
    def name(self) -> str:
        return "terra" if self.profile_name == "full" else f"terra-{self.profile_name}"

    @property
    def options(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "dataset": self._config.name,
        }

    def prepare(self, motion: str) -> PreparedMotion:
        try:
            record = self._records[motion]
        except KeyError as error:
            raise FileNotFoundError("motion is absent from the converted dataset manifest") from error
        if not record.fit_passed:
            raise ValueError("converted motion is marked fit_passed=false")
        if not self._shape_ready:
            ensure_robot_shape(self._config)
            object.__setattr__(self, "_shape_ready", True)
        return PreparedMotion(record)

    def fit(
        self,
        motion: str,
        prepared: PreparedMotion,
    ) -> ReconstructionResult:
        record = prepared.state
        terrain, report, validation, _offsets = fit_record_terrain(
            self._config,
            record,
            reconstruction_profile=self.profile_name,
        )
        if terrain is None:
            raise RuntimeError(f"TERRA {self.profile_name} produced no TerrainSpec for {motion}")
        return ReconstructionResult(terrain.to_dict(), report, validation)

    def summarize(self, result: ReconstructionResult) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "n_frames": result.fit.get("n_frames"),
            "n_stance_events": result.fit.get("n_stance_events"),
            "n_boxes": len(result.terrain.get("boxes", [])),
            "validation_passed": result.validation.get("passed"),
        }
