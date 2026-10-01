"""The two registered TERRA reconstruction profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terra.dataset_pipeline import (
    DatasetConfig,
    MotionRecord,
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
    _config: DatasetConfig = field(init=False, repr=False, compare=False)
    _records: dict[str, MotionRecord] = field(init=False, repr=False, compare=False)

    display_name: str = field(init=False)
    description: str = field(init=False)

    def __post_init__(self) -> None:
        profile = resolve_terra_reconstruction_profile(self.profile_name)
        object.__setattr__(self, "profile_name", profile.cli_name)
        config = load_dataset_config(self.dataset_config_path)
        if config.terrain_mode != "fit":
            raise ValueError(f"dataset {config.name} does not enable terrain fitting")
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_records", {record.motion: record for record in load_motion_records(config)})
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
        config = self._config
        input_identity = {
            "dataset": {
                "name": config.name,
                "config_path": str(self.dataset_config_path.expanduser().resolve()),
                "contact_source": "kinematic",
                "contact_joints": list(config.contact_joints),
                "calibration_mode": config.calibration_mode,
                "calibrate_sites": config.calibrate_sites,
                "terrain_fit": config.terrain_fit,
            },
            "source_motion": {"identifier": record.motion, "path": str(record.source_path)},
        }
        return PreparedMotion(record, input_identity)

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
        profile_input = report.get("input")
        if not isinstance(profile_input, dict):
            profile_input = {}
        report["input"] = profile_input | {
            **prepared.input,
        }
        return ReconstructionResult(terrain.to_dict(), report, validation)

    def summarize(self, result: ReconstructionResult) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "n_frames": result.fit.get("n_frames"),
            "n_stance_events": result.fit.get("n_stance_events"),
            "n_boxes": len(result.terrain.get("boxes", [])),
            "validation_passed": result.validation.get("passed"),
        }
