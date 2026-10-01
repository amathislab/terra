"""Benchmark adapter for Voronoi terrain reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from terra.benchmarking.terrain import (
    VoronoiConfig,
    fit_voronoi_terrain_from_motion,
)
from terra.source import (
    SEATED_SURFACE_MIN_SKIN_WEIGHT,
    SEATED_SURFACE_QUANTILE,
    posed_seat_support_heights,
)
from terra.terrain import SeatRest, validate_terrain
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

from ..core import PreparedMotion, ReconstructionResult
from .common import MotionLandmarks, prepare_motion_landmarks


@dataclass(frozen=True)
class VoronoiState:
    landmarks: MotionLandmarks
    offsets: dict[str, float]


@dataclass(frozen=True)
class VoronoiMethod:
    motions: tuple[str, ...]
    config: VoronoiConfig = field(default_factory=VoronoiConfig)
    env_name: str = "MyoFullBody"
    terrain_links: tuple[str, ...] = ("L_Toe", "R_Toe", "Pelvis")
    contact_joints: tuple[str, ...] = DEFAULT_CONTACT_JOINTS
    pelvis_link: str | None = "Pelvis"
    seat_height_source: str = "shared_posed_posterior_body_surface"
    posed_seat_frame: str = "normalized"
    link_offsets: dict[str, float] = field(default_factory=dict)
    source_paths: dict[str, Path] = field(default_factory=dict, repr=False, compare=False)
    smpl_model_path: Path | None = field(default=None, repr=False, compare=False)
    cache_root: Path | None = field(default=None, repr=False, compare=False)

    name: str = "voronoi"
    display_name: str = "Voronoi"
    description: str = "Discrete-height Voronoi terrain reconstruction based on TIP and SceneBot."

    def __post_init__(self) -> None:
        if self.env_name.replace("Mjx", "") != "MyoFullBody":
            raise ValueError("Voronoi is defined only for MyoFullBody landmarks")
        if not self.terrain_links:
            raise ValueError("Voronoi terrain_links must not be empty")
        if self.contact_joints != DEFAULT_CONTACT_JOINTS:
            raise ValueError("Voronoi contact_joints must be L/R toe and L/R ankle")
        if self.seat_height_source not in {
            "shared_posed_posterior_body_surface",
            "fixed_link_offset",
        }:
            raise ValueError(
                "Voronoi seat_height_source must be shared_posed_posterior_body_surface or fixed_link_offset"
            )
        if self.posed_seat_frame not in {"normalized", "apparatus"}:
            raise ValueError("posed_seat_frame must be normalized or apparatus")
        unknown = sorted(set(self.link_offsets) - set(self.terrain_links))
        if unknown:
            raise ValueError(f"Voronoi offsets contain non-terrain links: {unknown}")
        if any(not np.isfinite(value) for value in self.link_offsets.values()):
            raise ValueError("Voronoi link offsets must be finite")
        if (
            self.seat_height_source == "shared_posed_posterior_body_surface"
            and self.pelvis_link is not None
            and self.link_offsets.get(self.pelvis_link, 0.0) != 0.0
        ):
            raise ValueError("shared posed seat support cannot also use a fixed pelvis link offset")

    @property
    def options(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "env_name": self.env_name,
            "terrain_links": list(self.terrain_links),
            "contact_joints": list(self.contact_joints),
            "pelvis_link": self.pelvis_link,
            "seat_height_source": self.seat_height_source,
            "link_offsets_m": dict(sorted(self.link_offsets.items())),
        }

    def prepare(self, motion: str) -> PreparedMotion:
        landmarks, input_identity = prepare_motion_landmarks(
            motion,
            self.env_name,
            source_path=self.source_paths.get(motion),
            smpl_model_path=self.smpl_model_path,
            cache_root=self.cache_root,
        )
        offsets = {link: self.link_offsets.get(link, 0.0) for link in self.terrain_links}
        reconstruction_contacts = tuple(link for link in self.contact_joints if link in self.terrain_links)
        input_identity = dict(input_identity) | {
            "link_surface_offsets_m": dict(sorted(offsets.items())),
            "seat_height_source": self.seat_height_source,
            "posed_seat_frame": self.posed_seat_frame,
            "seat_height_heuristic": (
                {
                    "surface": "posterior_pelvis_and_proximal_hips",
                    "minimum_combined_skinning_weight": SEATED_SURFACE_MIN_SKIN_WEIGHT,
                    "height_quantile": SEATED_SURFACE_QUANTILE,
                    "summary_over_frames": "median_of_start_midpoint_end",
                }
                if self.seat_height_source == "shared_posed_posterior_body_surface"
                else None
            ),
            "contact_intervals": {
                "mode": "kinematic",
                "joints": list(reconstruction_contacts),
                "benchmark_queries": list(self.contact_joints),
            },
        }
        return PreparedMotion(
            VoronoiState(landmarks, offsets),
            input_identity,
        )

    def fit(
        self,
        motion: str,
        prepared: PreparedMotion,
    ) -> ReconstructionResult:
        state = prepared.state
        if not isinstance(state, VoronoiState):
            raise TypeError("Voronoi prepared state is not VoronoiState")
        landmarks = state.landmarks
        joints, fps, names = landmarks.joints, landmarks.fps, list(landmarks.joint_names)

        def resolve_pelvis_support(intervals: tuple[tuple[int, int], ...]) -> np.ndarray:
            if not intervals:
                return np.empty(0, dtype=float)
            if self.pelvis_link != "Pelvis":
                raise ValueError("shared posed seat support requires pelvis_link='Pelvis'")
            source_path = self.source_paths.get(motion)
            if source_path is None or self.smpl_model_path is None or self.cache_root is None:
                raise ValueError("shared posed seat support requires dataset motion, SMPL-H, and fitted-shape paths")
            pelvis = names.index("Pelvis")
            foot_indices = [names.index(link) for link in self.contact_joints if link in names]
            if not foot_indices:
                raise ValueError("shared posed seat support requires foot landmarks")
            rests = []
            for start, end in intervals:
                mid = (start + end) // 2
                pelvis_xy = joints[start:end, pelvis, :2].copy()
                here = np.median(pelvis_xy, axis=0)
                towards = np.mean(joints[mid, foot_indices, :2], axis=0) - here
                rests.append(
                    SeatRest(
                        start=start,
                        end=end,
                        z=float(np.median(joints[start:end, pelvis, 2])),
                        xy=pelvis_xy,
                        yaw=float(np.arctan2(towards[1], towards[0])),
                    )
                )
            from terra.runtime import shape_cache_path
            from terra.smplh import load_smplh_motion

            return posed_seat_support_heights(
                load_smplh_motion(source_path),
                str(self.smpl_model_path),
                str(shape_cache_path(self.env_name, self.cache_root)),
                joints,
                names,
                rests,
                calibrate_sites=True,
            ) - (
                float(prepared.input["normalization"]["source_to_normalized_translation_m"][2])
                if self.posed_seat_frame == "apparatus"
                else 0.0
            )

        shared_height = self.seat_height_source == "shared_posed_posterior_body_surface"
        terrain, report = fit_voronoi_terrain_from_motion(
            joints,
            names,
            fps,
            terrain_links=self.terrain_links,
            pelvis_link=self.pelvis_link,
            link_surface_offsets=state.offsets,
            pelvis_support_height_resolver=resolve_pelvis_support if shared_height else None,
            pelvis_support_height_source=self.seat_height_source,
            config=self.config,
            input_stage=prepared.input["input_stage"],
        )
        input_record = dict(prepared.input) | {
            "collision_points": {"source": "joints", "joint_order": names},
        }
        report["input"] = input_record

        foot_links = tuple(link for link in self.contact_joints if link in names and link in self.terrain_links)
        kinematic_intervals = {link: [] for link in foot_links}
        for interval in report["support_intervals"]:
            if interval.get("evidence_stage") != "accepted_before_edge_pruning":
                raise ValueError("fit support interval has an unexpected evidence stage")
            link = interval["link"]
            if link in kinematic_intervals and interval.get("kind") == "foot":
                kinematic_intervals[link].append([float(interval["start"]) / fps, float(interval["end"]) / fps])
        pelvis_index = names.index(self.pelvis_link) if self.pelvis_link in names else None
        seat_intervals = [
            interval
            for interval in report["support_intervals"]
            if interval["link"] == self.pelvis_link and interval.get("kind") == "pelvis"
        ]
        seat_rests = [
            SeatRest(
                start=int(interval["start"]),
                end=int(interval["end"]),
                z=float(np.median(joints[int(interval["start"]) : int(interval["end"]), pelvis_index, 2])),
                xy=joints[int(interval["start"]) : int(interval["end"]), pelvis_index, :2].copy(),
            )
            for interval in seat_intervals
            if pelvis_index is not None
        ]
        seat_support_heights = (
            [float(interval["surface_height_m"]) for interval in seat_intervals] if shared_height else None
        )
        validation_offsets = {link: state.offsets.get(link, 0.0) for link in foot_links}
        contact_tol = 0.05
        pelvis_offset = state.offsets.get(self.pelvis_link, 0.0) if self.pelvis_link is not None else 0.0
        validation = validate_terrain(
            joints,
            names,
            terrain,
            fps,
            contact_joints=foot_links,
            stance_speed=self.config.velocity_threshold_m_s,
            contact_tol=contact_tol,
            offsets=validation_offsets,
            seat_rests=seat_rests,
            pelvis_seat_offset=pelvis_offset,
            seat_support_heights=seat_support_heights,
            compensate_sloped_offsets=False,
            _kinematic_intervals_s=kinematic_intervals,
        )
        validation.update(
            role="in_sample_diagnostic_only",
            influenced_reconstruction=False,
            settings={
                "contact_tol_m": contact_tol,
                "contact_joints": list(foot_links),
                "contact_interval_source": "fit.support_intervals[accepted_before_edge_pruning]",
                "kinematic_contact_intervals_s": kinematic_intervals,
                "link_surface_offsets_m": validation_offsets,
                "stance_speed_m_s": self.config.velocity_threshold_m_s,
                "seat_rest_intervals": [{"start": rest.start, "end": rest.end} for rest in seat_rests],
                "pelvis_link": self.pelvis_link,
                "pelvis_seat_offset_m": pelvis_offset,
                "seat_height_source": self.seat_height_source,
                "seated_surface_min_skin_weight": (SEATED_SURFACE_MIN_SKIN_WEIGHT if shared_height else None),
                "seated_surface_quantile": SEATED_SURFACE_QUANTILE if shared_height else None,
                "compensate_sloped_offsets": False,
            },
        )
        report["model"] = "voronoi"
        report["method"] = "Voronoi terrain reconstruction based on TIP and SceneBot"
        return ReconstructionResult(terrain.to_dict(), report, validation)

    def summarize(self, result: ReconstructionResult) -> dict[str, Any]:
        return {
            "n_frames": result.fit.get("n_frames"),
            "n_contact_edges": result.fit.get("n_contact_edges"),
            "n_boxes": result.fit.get("n_boxes"),
            "validation_passed": result.validation.get("passed"),
        }
