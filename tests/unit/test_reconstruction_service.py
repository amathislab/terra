"""Parity fixtures for the shared production TERRA reconstruction service."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pytest

from terra.profiles import SolverConfig
from terra.reconstruction import (
    CalibrationEvidence,
    ReconstructionRequest,
    reconstruct_terrain,
)
from terra.terrain import (
    PELVIS_SEAT_OFFSET,
    detect_seat_rests,
    fit_terrain_from_motion,
    resolve_terra_reconstruction_profile,
    validate_terrain,
)
from tests.unit.test_terrain_reconstruction import (
    FPS,
    JOINT_OFFSET,
    JOINTS,
    RAMP_HEIGHT,
    SEAT_JOINTS,
    STAIR_HEIGHT,
    profile_walk,
    synth_sit,
    synth_walk,
)


def _normalized(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    return value


def _direct(joints, names, fps, options, *, compensate=None):
    terrain, report = fit_terrain_from_motion(joints, names, fps, **options)
    if compensate is None:
        compensate = report.get("joint_offsets_source") != "flat_reference"
    validation = validate_terrain(
        joints,
        names,
        terrain,
        fps,
        offsets=report.get("joint_offsets"),
        pelvis_seat_offset=report.get("pelvis_seat_offset", PELVIS_SEAT_OFFSET),
        seat_support_heights=report.get("seat_support_heights"),
        compensate_sloped_offsets=compensate,
    )
    return terrain, report, validation


def _assert_strictly_serializable(result) -> None:
    payload = {
        "terrain": None if result.terrain is None else result.terrain.to_dict(),
        "fit": result.report,
        "validation": result.validation,
    }
    serialized = json.dumps(_normalized(payload), allow_nan=False)
    assert "free_space_u" not in serialized
    assert "free_space_width_limit" not in serialized


@pytest.mark.parametrize(
    ("family", "motion"),
    (
        ("flat", synth_walk(beam_height=0.0)),
        ("ramp", profile_walk(RAMP_HEIGHT)),
        ("stairs", profile_walk(STAIR_HEIGHT)),
        ("seat", synth_sit()),
    ),
)
def test_shared_service_preserves_direct_numerical_fit_and_validation(family, motion):
    names = SEAT_JOINTS if family == "seat" else JOINTS
    options = {}
    direct_terrain, direct_report, direct_validation = _direct(motion, names, FPS, options)

    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=motion,
            joint_names=tuple(names),
            fps=FPS,
            fit_options=options,
        )
    )

    assert result.terrain is not None
    assert result.terrain.to_dict() == direct_terrain.to_dict()
    assert _normalized(result.report) == _normalized(direct_report)
    assert _normalized(result.validation) == _normalized(direct_validation)
    _assert_strictly_serializable(result)


@pytest.mark.parametrize(
    "profile_name",
    ("no-physical-cues", "full"),
)
def test_calibration_and_named_profiles_have_one_effective_configuration(profile_name):
    motion = synth_walk(beam_height=0.10)
    profile = resolve_terra_reconstruction_profile(profile_name)
    calibration = CalibrationEvidence(
        neutral_foot_pitch={"L": 0.0, "R": 0.0},
        joint_offsets=JOINT_OFFSET,
        source="manifest",
        path="flat-reference.npz",
    )
    effective = profile.fit_options(
        {},
        neutral_foot_pitch=calibration.neutral_foot_pitch,
        calibrated_joint_offsets=calibration.joint_offsets,
    )
    direct_terrain, direct_report, direct_validation = _direct(
        motion,
        JOINTS,
        FPS,
        effective,
        compensate=False,
    )

    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=motion,
            joint_names=tuple(JOINTS),
            fps=FPS,
            calibration=calibration,
            profile=profile,
            record_profile=True,
            report_fields={
                "contact_source": "kinematic",
                "calibration_path": "flat-reference.npz",
            },
            input_metadata={"adapter": "fixture"},
        )
    )

    assert result.terrain is not None
    assert result.terrain.to_dict() == direct_terrain.to_dict()
    for key, value in direct_report.items():
        assert _normalized(result.report[key]) == _normalized(value)
    assert _normalized(result.validation) == _normalized(direct_validation)
    assert _normalized(result.effective_fit_options) == _normalized(effective)
    assert result.report["reconstruction_profile"]["name"] == profile.name
    assert result.report["contact_source"] == "kinematic"
    assert result.calibration.path == "flat-reference.npz"
    _assert_strictly_serializable(result)


@pytest.mark.parametrize(
    "fit_options",
    (
        {"contact_intervals_s": {"L_Toe": [[0.0, 1.0]]}},
        {"contact_joints": ("L_Toe",)},
    ),
)
def test_shared_service_rejects_alternate_contact_inputs(fit_options):
    with pytest.raises(ValueError, match="contact_"):
        reconstruct_terrain(
            ReconstructionRequest(
                mode="fit",
                joints=synth_walk(beam_height=0.0),
                joint_names=tuple(JOINTS),
                fps=FPS,
                fit_options=fit_options,
            )
        )


def test_flat_dataset_request_preserves_flat_report_and_profile_settings():
    profile = resolve_terra_reconstruction_profile("full")
    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="flat",
            joints=None,
            joint_names=(),
            fps=None,
            profile=profile,
            record_profile=True,
            report_fields={"dataset_config": "study"},
        )
    )
    assert result.terrain is None
    assert result.validation == {"passed": True}
    assert result.report["model"] == "flat"
    assert result.report["dataset_config"] == "study"
    assert result.report["reconstruction_profile"] == profile.to_dict({})


def test_pipeline_auto_resolution_has_the_same_numerical_result_as_the_service():
    from terra.baselines.omniretarget import SMPLH_DEMO_JOINTS
    from terra.pipeline import _resolve_terrain

    compact = profile_walk(RAMP_HEIGHT)
    names = tuple(SMPLH_DEMO_JOINTS)
    motion = np.zeros((len(compact), len(names), 3), dtype=compact.dtype)
    for source_index, joint in enumerate(JOINTS):
        motion[:, names.index(joint)] = compact[:, source_index]
    options = {"seat": "off"}
    expected = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=motion,
            joint_names=names,
            fps=FPS,
            fit_options=options,
        )
    )

    actual = _resolve_terrain(
        "auto",
        motion,
        FPS,
        SolverConfig.from_mapping({"terrain_fit": options, "use_fitted_shape": False, "calibrate_sites": False}),
        logging.getLogger(__name__),
        motion_data={},
        smpl_model_path="unused-smpl-model",
        fitted_shape_path="unused-fitted-shape",
        normalization={"source_to_normalized_translation_m": [0.0, 0.0, 0.0]},
    )

    assert expected.terrain is not None
    assert actual.to_dict() == expected.terrain.to_dict()


def test_service_scores_rejected_floor_level_seat_without_a_length_mismatch():
    motion = synth_sit(pelvis_z=0.56)
    rests = detect_seat_rests(motion, SEAT_JOINTS, FPS)
    result = reconstruct_terrain(
        ReconstructionRequest(
            mode="fit",
            joints=motion,
            joint_names=tuple(SEAT_JOINTS),
            fps=FPS,
            fit_options={"seat_support_heights": [0.03] * len(rests)},
        )
    )

    assert result.report["n_seats"] == 0
    assert result.validation["n_seat_rests"] == len(rests)
