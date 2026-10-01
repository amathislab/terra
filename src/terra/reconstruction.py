"""Typed production service for TERRA terrain reconstruction.

Callers prepare motion landmarks and any dataset-specific calibration evidence,
then submit one :class:`ReconstructionRequest`.  The service is the sole production
owner of profile resolution, terrain fitting, internal validation, and their shared
report structure.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from terra._musclemimic import TerrainSpec
from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
from terra.source import posed_seat_support_heights
from terra.terrain import (
    DEFAULT_CONTACT_JOINTS,
    TerraReconstructionProfile,
    detect_seat_rests,
)


@dataclass(frozen=True)
class CalibrationEvidence:
    """Motion-derived orientation and sole-height calibration supplied to fitting."""

    neutral_foot_pitch: Mapping[str, float] | None = None
    joint_offsets: Mapping[str, float] | None = None
    source: str = "motion"
    path: str | None = None


@dataclass(frozen=True)
class ValidationPolicy:
    """Validation behavior that differs only by calibrated landmark convention."""

    compensate_sloped_offsets: bool | None = None


@dataclass(frozen=True)
class ReconstructionRequest:
    """Input to one terrain fit and validation pass.

    Use ``mode="fit"`` with ``joints`` shaped (frames, landmarks, 3),
    matching ``joint_names``, and positive ``fps``. Contact intervals are
    inferred from motion; callers cannot supply them. ``fit_options`` and an
    optional named ``profile`` select the fitting rules, while ``calibration``
    supplies measured foot pitch and offsets. Use ``mode="flat"`` with no
    landmarks or frame rate for a flat-ground result.
    """

    mode: Literal["fit", "flat"]
    joints: np.ndarray | None
    joint_names: tuple[str, ...]
    fps: float | None
    fit_options: Mapping[str, Any] = field(default_factory=dict)
    calibration: CalibrationEvidence = field(default_factory=CalibrationEvidence)
    profile: TerraReconstructionProfile | None = None
    record_profile: bool = False
    report_fields: Mapping[str, Any] = field(default_factory=dict)
    input_metadata: Mapping[str, Any] | None = None
    seat_ground_datum_correction_m: float | None = None
    validation: ValidationPolicy = field(default_factory=ValidationPolicy)


@dataclass(frozen=True)
class TerrainFitResult:
    """One fitted terrain and the evidence/configuration that produced it."""

    terrain: TerrainSpec | None
    report: dict[str, Any]
    validation: dict[str, Any]
    effective_fit_options: dict[str, Any]
    calibration: CalibrationEvidence


def add_posed_seat_support(
    fit_options: Mapping[str, Any],
    joints: np.ndarray,
    fps: float,
    motion_data: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    smpl_model_path: str | os.PathLike[str],
    fitted_shape_path: str | os.PathLike[str],
    *,
    use_fitted_shape: bool,
    calibrate_sites: bool,
    ground_datum_correction_m: float = 0.0,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Add posed posterior body-surface heights in the terrain fitter's source datum.

    Foot-supported surfaces are already expressed in the source/apparatus frame: both
    stance-joint heights and their calibrated surface offsets are measured after source
    normalization, so the common vertical translation cancels.  The posed-seat estimator
    instead subtracts a translation-invariant pelvis-to-surface distance from an absolute
    normalized pelvis height, leaving that translation in its result.  Callers therefore
    pass the negative source-to-normalized vertical translation exactly once so seats and
    foot-supported boxes share the source/apparatus frame.
    """

    options = dict(fit_options)
    if options.get("seat", "auto") != "auto" or "seat_support_heights" in options:
        return options
    if not use_fitted_shape:
        if logger is not None:
            logger.warning(
                "Terrain fit: posed seat support requires the robot-fitted body shape; "
                "using the fixed pelvis offset for subject-shaped landmarks"
            )
        return options
    names = list(SMPLH_DEMO_JOINTS)
    rests = detect_seat_rests(
        joints,
        names,
        fps,
        allow_boundary_truncation=bool(options.get("allow_boundary_truncated_support", True)),
    )
    if not rests:
        return options
    loaded_motion = motion_data() if callable(motion_data) else motion_data
    support = posed_seat_support_heights(
        dict(loaded_motion),
        str(smpl_model_path),
        str(fitted_shape_path),
        joints,
        names,
        rests,
        calibrate_sites=calibrate_sites,
    )
    if not np.isfinite(ground_datum_correction_m):
        raise ValueError("ground_datum_correction_m must be finite")
    options["seat_support_heights"] = support + float(ground_datum_correction_m)
    if logger is not None:
        logger.info("Terrain fit: %d seated rest(s) use the posed posterior body surface", len(support))
    return options


def _effective_options(request: ReconstructionRequest) -> dict[str, Any]:
    options = dict(request.fit_options)
    if "contact_intervals_s" in options:
        raise ValueError("contact_intervals_s is not supported; contact timing is inferred from motion")
    configured_joints = tuple(options.pop("contact_joints", DEFAULT_CONTACT_JOINTS))
    if configured_joints != DEFAULT_CONTACT_JOINTS:
        raise ValueError(f"contact_joints must be {list(DEFAULT_CONTACT_JOINTS)!r}")
    if request.profile is None:
        return options
    return request.profile.fit_options(
        options,
        neutral_foot_pitch=request.calibration.neutral_foot_pitch,
        calibrated_joint_offsets=request.calibration.joint_offsets,
    )


def _annotate_report(
    request: ReconstructionRequest,
    report: dict[str, Any],
    effective_options: Mapping[str, Any],
) -> None:
    report.update(request.report_fields)
    if report.get("seat_support_heights") is not None and request.seat_ground_datum_correction_m is not None:
        report["seat_ground_datum_correction_m"] = request.seat_ground_datum_correction_m
    if request.record_profile:
        if request.profile is None:
            raise ValueError("record_profile=True requires an explicit reconstruction profile")
        report["reconstruction_profile"] = request.profile.to_dict(effective_options)
        if request.input_metadata is not None:
            report["input"] = dict(request.input_metadata)
        report["unsupported_support_kinds"] = list(request.profile.unsupported_support_kinds)
        if request.profile.unsupported_support_kinds:
            report["evidence_completeness"] = "foot_only"
            report["evidence_limitation"] = (
                "This reduced profile disables seat reconstruction and therefore cannot "
                "recover pelvis-only support such as a chair seat."
            )


def reconstruct_terrain(request: ReconstructionRequest) -> TerrainFitResult:
    """Fit support geometry and validate it against motion landmarks.

    In fit mode, returns a terrain specification, a fit evidence report,
    validation metrics, and effective options. In flat mode, returns
    ``terrain=None`` and a passing flat validation record. A failed validation
    is reported in the result; invalid request structure raises ``ValueError``.
    The CLI cohort runner persists these fields per motion and records failures
    in ``status.csv``.
    """

    if request.mode == "flat":
        effective_options = dict(request.fit_options)
        if request.joints is not None or request.fps is not None:
            raise ValueError("flat reconstruction requests must not provide motion landmarks")
        report: dict[str, Any] = {"model": "flat"}
        _annotate_report(request, report, effective_options)
        return TerrainFitResult(
            terrain=None,
            report=report,
            validation={"passed": True},
            effective_fit_options=effective_options,
            calibration=request.calibration,
        )
    if request.mode != "fit":
        raise ValueError(f"unsupported reconstruction mode {request.mode!r}")
    effective_options = _effective_options(request)
    if request.joints is None or request.fps is None:
        raise ValueError("fit reconstruction requests require joints and fps")
    joints = np.asarray(request.joints)
    names = list(request.joint_names)
    from terra.terrain import PELVIS_SEAT_OFFSET, fit_terrain_from_motion, validate_terrain

    terrain, report = fit_terrain_from_motion(joints, names, float(request.fps), **effective_options)
    compensate = request.validation.compensate_sloped_offsets
    if compensate is None:
        # Motion-estimated offsets need slope compensation. Any explicitly supplied
        # calibration—including the zero-offset ablation—is a fixed validation baseline.
        compensate = report.get("joint_offsets_source") == "motion"
    validation = validate_terrain(
        joints,
        names,
        terrain,
        float(request.fps),
        offsets=report.get("joint_offsets"),
        pelvis_seat_offset=report.get("pelvis_seat_offset", PELVIS_SEAT_OFFSET),
        seat_support_heights=report.get("seat_support_heights"),
        allow_boundary_truncation=bool(effective_options.get("allow_boundary_truncated_support", True)),
        compensate_sloped_offsets=compensate,
    )
    _annotate_report(request, report, effective_options)
    return TerrainFitResult(
        terrain=terrain,
        report=report,
        validation=validation,
        effective_fit_options=effective_options,
        calibration=request.calibration,
    )


def implicit_calibration_evidence(options: Mapping[str, Any]) -> CalibrationEvidence:
    """Extract production calibration settings from an unprofiled fitter option map."""

    return CalibrationEvidence(
        neutral_foot_pitch=options.get("neutral_foot_pitch"),
        joint_offsets=options.get("calibrated_joint_offsets"),
    )


__all__ = [
    "CalibrationEvidence",
    "ReconstructionRequest",
    "TerrainFitResult",
    "ValidationPolicy",
    "add_posed_seat_support",
    "implicit_calibration_evidence",
    "reconstruct_terrain",
]
