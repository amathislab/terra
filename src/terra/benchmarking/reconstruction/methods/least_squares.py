"""Contact least-squares terrain reconstruction baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from terra.benchmarking.terrain import LeastSquaresPlaneConfig, fit_least_squares_contact_plane
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

from ..core import PreparedMotion, ReconstructionResult
from .common import MotionLandmarks, prepare_motion_landmarks


@dataclass(frozen=True)
class LeastSquaresMethod:
    config: LeastSquaresPlaneConfig = field(default_factory=LeastSquaresPlaneConfig)
    env_name: str = "MyoFullBody"
    source_paths: dict[str, Path] = field(default_factory=dict, repr=False, compare=False)
    smpl_model_path: Path | None = field(default=None, repr=False, compare=False)
    cache_root: Path | None = field(default=None, repr=False, compare=False)

    name: str = "contact-least-squares"
    display_name: str = "Contact least squares"
    description: str = (
        "One affine terrain plane fitted by least squares to kinematically inferred toe and ankle contacts."
    )

    @property
    def options(self) -> dict[str, Any]:
        return {
            "env_name": self.env_name,
            "contact_joints": list(DEFAULT_CONTACT_JOINTS),
            "config": asdict(self.config),
        }

    def prepare(self, motion: str) -> PreparedMotion:
        landmarks, identity = prepare_motion_landmarks(
            motion,
            self.env_name,
            source_path=self.source_paths.get(motion),
            smpl_model_path=self.smpl_model_path,
            cache_root=self.cache_root,
        )
        return PreparedMotion(landmarks, identity)

    def fit(
        self,
        motion: str,
        prepared: PreparedMotion,
    ) -> ReconstructionResult:
        landmarks = prepared.state
        if not isinstance(landmarks, MotionLandmarks):
            raise TypeError("contact-least-squares prepared state is not MotionLandmarks")
        terrain, report = fit_least_squares_contact_plane(
            landmarks.joints,
            landmarks.joint_names,
            landmarks.fps,
            config=self.config,
            input_stage=prepared.input["input_stage"],
        )
        input_record = dict(prepared.input)
        report["input"] = input_record
        return ReconstructionResult(
            terrain=terrain.to_dict(),
            fit=report,
            validation={
                "passed": None,
                "role": "not_run_for_simple_reconstruction_baseline",
                "influenced_reconstruction": False,
            },
        )

    def summarize(self, result: ReconstructionResult) -> dict[str, Any]:
        return {
            "n_frames": result.fit.get("n_frames"),
            "n_contact_events": result.fit.get("n_contact_events"),
            "n_boxes": result.fit.get("n_boxes"),
        }
