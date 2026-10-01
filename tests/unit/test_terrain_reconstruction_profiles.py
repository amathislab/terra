"""Tests for TERRA and its physical-cues reconstruction ablation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from terra.terrain.reconstruction_profiles import (
    TERRA_FULL_PROFILE,
    TERRA_NO_PHYSICAL_CUES_PROFILE,
    resolve_terra_reconstruction_profile,
)
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

OFFSETS = {
    "L_Ankle": 0.07,
    "L_Toe": 0.02,
    "R_Ankle": 0.07,
    "R_Toe": 0.02,
}
NEUTRAL = {"L": -20.0, "R": -21.0}


def _options(name: str, base=None):
    return resolve_terra_reconstruction_profile(name).fit_options(
        base,
        neutral_foot_pitch=NEUTRAL,
        calibrated_joint_offsets=OFFSETS,
    )


def test_profile_names_resolve_from_canonical_and_cli_spellings():
    assert resolve_terra_reconstruction_profile("full").name == TERRA_FULL_PROFILE
    assert resolve_terra_reconstruction_profile("no-physical-cues").name == TERRA_NO_PHYSICAL_CUES_PROFILE
    assert resolve_terra_reconstruction_profile(TERRA_FULL_PROFILE).cli_name == "full"
    with pytest.raises(ValueError, match="unsupported TERRA reconstruction profile"):
        resolve_terra_reconstruction_profile("bad")


def test_full_preserves_current_modes_and_enables_both_additional_cues():
    base = {"max_extension": (0.0, 0.0), "seat_support_heights": [0.31]}
    options = _options("full", base)

    assert options["max_extension"] == (0.0, 0.0)
    assert options["seat_support_heights"] == [0.31]
    assert options["neutral_foot_pitch"] == NEUTRAL
    assert "contact_intervals_s" not in options
    assert options["calibrated_joint_offsets"] == OFFSETS
    assert not any(key in options for key in ("ramp", "stair_flight", "seat"))
    assert base == {"max_extension": (0.0, 0.0), "seat_support_heights": [0.31]}


def test_profile_serialization_records_the_full_method_options():
    profile = resolve_terra_reconstruction_profile("full")
    options = _options("full")
    settings = profile.to_dict(options)

    assert settings["name"] == TERRA_FULL_PROFILE
    assert settings["effective_fit_options"] == options
    assert settings["forced_fit_options"] == {}
    assert settings["additional_evidence"] == {
        "calibrated_sole_offsets": True,
        "free_space_extent": True,
        "neutral_foot_pitch": True,
        "physical_family_cues": True,
        "posed_seat_surface": True,
    }
    assert "ramp_primitive" in settings["cumulative_stages"]
    assert "per_level_boxes" in settings["cumulative_stages"]


def test_no_physical_cues_changes_only_ramp_step_selection_evidence():
    full = _options("full")
    ablated = _options("no-physical-cues")

    assert ablated["calibrated_joint_offsets"] == full["calibrated_joint_offsets"]
    assert ablated["neutral_foot_pitch"] is None
    assert "contact_intervals_s" not in ablated
    assert ablated["use_free_space_evidence"] is True
    assert "exclude_claimed" not in ablated
    assert ablated["family_evidence_mode"] == "height_only"

    settings = resolve_terra_reconstruction_profile("no-physical-cues").to_dict(ablated)
    assert settings["additional_evidence"]["physical_family_cues"] is False
    assert settings["forced_fit_options"] == {"family_evidence_mode": "height_only"}


def test_posed_seat_surface_is_returned_to_the_motion_derived_ground_datum(monkeypatch):
    from terra import pipeline, reconstruction

    monkeypatch.setattr(reconstruction, "detect_seat_rests", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(
        reconstruction,
        "posed_seat_support_heights",
        lambda *_args, **_kwargs: np.array([0.31, 0.56]),
    )

    options = pipeline._add_posed_seat_support(
        {},
        np.zeros((2, len(pipeline.SMPLH_DEMO_JOINTS), 3)),
        100.0,
        {},
        "smpl",
        "shape.pkl",
        use_fitted_shape=True,
        calibrate_sites=True,
        ground_datum_correction_m=-0.06,
    )

    np.testing.assert_allclose(options["seat_support_heights"], [0.25, 0.50])


def test_posed_seat_surface_is_invariant_to_lowest_toe_translation(monkeypatch):
    from terra import reconstruction

    names = list(reconstruction.SMPLH_DEMO_JOINTS)
    pelvis = names.index("Pelvis")
    monkeypatch.setattr(reconstruction, "detect_seat_rests", lambda *_args, **_kwargs: [object()])

    def posed(_motion, _model, _shape, solver_joints, *_args, **_kwargs):
        return np.array([0.31 + solver_joints[0, pelvis, 2]])

    monkeypatch.setattr(reconstruction, "posed_seat_support_heights", posed)

    published = []
    for translation in (0.0, 0.08):
        joints = np.zeros((2, len(names), 3))
        joints[:, :, 2] += translation
        options = reconstruction.add_posed_seat_support(
            {},
            joints,
            100.0,
            {},
            "smpl",
            "shape.pkl",
            use_fitted_shape=True,
            calibrate_sites=True,
            ground_datum_correction_m=-translation,
        )
        published.append(options["seat_support_heights"])

    np.testing.assert_allclose(published[0], published[1])


def test_dataset_seat_surface_is_invariant_to_normalization_anchor(monkeypatch, tmp_path):
    from terra import reconstruction

    dataset_pipeline, config, record, calls = _mock_dataset_fit(monkeypatch, tmp_path)
    monkeypatch.setattr(dataset_pipeline, "load_smplh_motion", lambda _path: {})
    monkeypatch.setattr(reconstruction, "detect_seat_rests", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(
        reconstruction,
        "posed_seat_support_heights",
        lambda _motion, _model, _shape, solver_joints, *_args, **_kwargs: np.array([0.31 + solver_joints[0, 0, 2]]),
    )

    published = []
    for translation in (0.0, 0.08):
        joints = np.zeros((10, 2, 3), dtype=float)
        joints[:, :, 2] += translation

        def world_joints(
            _config,
            _record,
            _path,
            *,
            return_normalization=False,
            _joints=joints,
            _translation=translation,
        ):
            if return_normalization:
                return _joints, 100.0, {"source_to_normalized_translation_m": [0.0, 0.0, _translation]}
            return _joints, 100.0

        monkeypatch.setattr(dataset_pipeline, "_world_joints", world_joints)
        dataset_pipeline.fit_record_terrain(config, record)
        published.append(calls[-1]["seat_support_heights"])

    np.testing.assert_allclose(published[0], published[1])


def test_posed_seat_detection_uses_the_boundary_support_setting(monkeypatch):
    from terra import reconstruction

    observed = []

    def detect(*_args, **kwargs):
        observed.append(kwargs["allow_boundary_truncation"])
        return []

    monkeypatch.setattr(reconstruction, "detect_seat_rests", detect)

    options = reconstruction.add_posed_seat_support(
        {"allow_boundary_truncated_support": False},
        np.zeros((2, len(reconstruction.SMPLH_DEMO_JOINTS), 3)),
        100.0,
        {},
        "smpl",
        "shape.pkl",
        use_fitted_shape=True,
        calibrate_sites=True,
    )

    assert observed == [False]
    assert options == {"allow_boundary_truncated_support": False}


def _mock_dataset_fit(monkeypatch, tmp_path):
    import terra.terrain as terrain_module
    from terra import dataset_pipeline, reconstruction
    from terra._musclemimic import TerrainSpec

    joints = np.zeros((10, 2, 3), dtype=float)
    config = SimpleNamespace(
        name="study",
        terrain_mode="fit",
        terrain_fit={},
        contact_joints=DEFAULT_CONTACT_JOINTS,
        smpl_model_path=tmp_path / "smpl",
        cache_root=tmp_path / "cache",
        env_name="MyoFullBody",
        calibrate_sites=True,
    )
    source = tmp_path / "motion.npz"
    calibration = tmp_path / "flat.npz"
    source.touch()
    calibration.touch()
    record = SimpleNamespace(
        motion="Study/motion",
        source_path=source,
        calibration_path=calibration,
    )

    def world_joints(_config, _record, _path, *, return_normalization=False):
        if return_normalization:
            return joints, 100.0, {"source_to_normalized_translation_m": [0.0, 0.0, 0.0]}
        return joints, 100.0

    monkeypatch.setattr(dataset_pipeline, "_world_joints", world_joints)
    monkeypatch.setattr(
        terrain_module,
        "calibrate_neutral_foot_pitch",
        lambda *_args, **_kwargs: dict(NEUTRAL),
    )
    monkeypatch.setattr(terrain_module, "detect_stance_events", lambda *_args, **_kwargs: [object()])
    monkeypatch.setattr(terrain_module, "joint_surface_offsets", lambda _events: dict(OFFSETS))
    monkeypatch.setattr(terrain_module, "paired_sole_offsets", lambda offsets: dict(offsets))
    monkeypatch.setattr(reconstruction, "detect_seat_rests", lambda *_args, **_kwargs: [])
    calls = []

    def fit(_joints, _names, _fps, **kwargs):
        calls.append(kwargs)
        return TerrainSpec(), {
            "model": "flat",
            "joint_offsets": dict(OFFSETS),
            "pelvis_seat_offset": 0.25,
        }

    monkeypatch.setattr(terrain_module, "fit_terrain_from_motion", fit)
    monkeypatch.setattr(terrain_module, "validate_terrain", lambda *_args, **_kwargs: {"passed": True})
    return dataset_pipeline, config, record, calls


def test_explicit_full_matches_implicit_production_options_but_adds_profile_metadata(
    monkeypatch,
    tmp_path,
):
    dataset_pipeline, config, record, calls = _mock_dataset_fit(monkeypatch, tmp_path)

    _terrain, implicit, *_rest = dataset_pipeline.fit_record_terrain(config, record)
    _terrain, explicit, *_rest = dataset_pipeline.fit_record_terrain(
        config,
        record,
        reconstruction_profile="full",
    )

    assert calls[0] == calls[1]
    assert calls[1]["neutral_foot_pitch"] == NEUTRAL
    assert "reconstruction_profile" not in implicit
    assert explicit["reconstruction_profile"]["name"] == TERRA_FULL_PROFILE
    assert explicit["unsupported_support_kinds"] == []
    assert explicit["input"]["joint_order"]


@pytest.mark.parametrize(
    "profile, frame, height",
    [
        (None, None, 0.42),
        (None, "apparatus", 0.42),
        (None, "normalized", 0.45),
        ("full", None, 0.45),
        ("full", "normalized", 0.45),
        ("full", "apparatus", 0.42),
        ("no-physical-cues", None, 0.45),
        ("no-physical-cues", "apparatus", 0.42),
    ],
)
def test_dataset_seat_height_uses_the_declared_reconstruction_frame(monkeypatch, tmp_path, profile, frame, height):
    from terra import dataset_pipeline, reconstruction

    config = SimpleNamespace(
        terrain_mode="fit",
        terrain_fit={},
        env_name="MyoFullBody",
        smpl_model_path=tmp_path,
        cache_root=tmp_path,
        calibrate_sites=True,
        name="example",
        calibration_mode="none",
        posed_seat_frame=frame,
    )
    record = SimpleNamespace(source_path=tmp_path / "motion.npz", calibration_path=None)
    joints = np.zeros((3, len(reconstruction.SMPLH_DEMO_JOINTS), 3))
    monkeypatch.setattr(
        dataset_pipeline,
        "_world_joints",
        lambda *_a, **_k: (
            joints,
            50.0,
            {"source_to_normalized_translation_m": [0.0, 0.0, 0.03]},
        ),
    )
    monkeypatch.setattr(dataset_pipeline, "_record_calibration", lambda *_a: (None, {}))
    monkeypatch.setattr(dataset_pipeline, "load_smplh_motion", lambda *_a: {})
    monkeypatch.setattr(reconstruction, "detect_seat_rests", lambda *_a, **_k: [object()])
    monkeypatch.setattr(reconstruction, "posed_seat_support_heights", lambda *_a, **_k: np.array([0.45]))
    observed = []

    def fit(request):
        observed.extend(request.fit_options["seat_support_heights"])
        return SimpleNamespace(terrain=None, report={}, validation={})

    monkeypatch.setattr(dataset_pipeline, "reconstruct_terrain", fit)
    dataset_pipeline.fit_record_terrain(config, record, reconstruction_profile=profile)
    assert observed == pytest.approx([height])
