"""Tests for the registered Voronoi reconstruction adapter."""

from __future__ import annotations

import numpy as np
import pytest

from terra.benchmarking.reconstruction.core import PreparedMotion
from terra.benchmarking.reconstruction.methods.common import MotionLandmarks
from terra.benchmarking.reconstruction.methods.voronoi import VoronoiMethod, VoronoiState


@pytest.mark.parametrize("frame, height", [("normalized", 0.42), ("apparatus", 0.36)])
def test_voronoi_adapter_applies_the_shared_surface_to_its_own_intervals(monkeypatch, tmp_path, frame, height):
    import terra.benchmarking.reconstruction.methods.voronoi as module
    import terra.runtime as runtime
    import terra.smplh as smplh

    names = ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle", "Pelvis")
    frames = 21
    joints = np.zeros((frames, len(names), 3), dtype=float)
    joints[:, names.index("L_Toe"), :2] = (-0.1, 0.1)
    joints[:, names.index("R_Toe"), :2] = (0.1, -0.1)
    joints[:, names.index("L_Ankle"), :2] = (-0.1, 0.1)
    joints[:, names.index("R_Ankle"), :2] = (0.1, -0.1)
    joints[:, names.index("Pelvis")] = (0.5, 0.0, 0.8)

    source = tmp_path / "motion.npz"
    shape = tmp_path / "shape.pkl"
    source.touch()
    shape.touch()
    observed = {}

    monkeypatch.setattr(smplh, "load_smplh_motion", lambda path: {"path": str(path)})
    monkeypatch.setattr(runtime, "shape_cache_path", lambda *_args: shape)

    def posed(_motion, _model, _shape, solver_joints, demo_joints, rests, **kwargs):
        observed["rest_intervals"] = [(rest.start, rest.end) for rest in rests]
        observed["yaw"] = [rest.yaw for rest in rests]
        assert solver_joints is joints
        assert tuple(demo_joints) == names
        assert kwargs == {"calibrate_sites": True}
        return np.array([0.42])

    def validate(*_args, **kwargs):
        observed["validation_rests"] = [(rest.start, rest.end) for rest in kwargs["seat_rests"]]
        observed["validation_heights"] = kwargs["seat_support_heights"]
        return {"passed": True}

    monkeypatch.setattr(module, "posed_seat_support_heights", posed)
    monkeypatch.setattr(module, "validate_terrain", validate)

    method = VoronoiMethod(
        motions=("Study/A",),
        posed_seat_frame=frame,
        source_paths={"Study/A": source},
        smpl_model_path=tmp_path / "smpl",
        cache_root=tmp_path / "cache",
    )
    result = method.fit(
        "Study/A",
        PreparedMotion(
            VoronoiState(MotionLandmarks(joints, 20.0, names), dict.fromkeys(method.terrain_links, 0.0)),
            {
                "input_stage": "fitted_myofullbody_landmarks",
                "normalization": {"source_to_normalized_translation_m": [0.0, 0.0, 0.06]},
            },
        ),
    )

    assert observed["rest_intervals"] == [(0, frames)]
    assert observed["validation_rests"] == observed["rest_intervals"]
    assert observed["validation_heights"] == pytest.approx([height])
    assert observed["yaw"] == pytest.approx([np.pi])
    pelvis = [item for item in result.fit["support_intervals"] if item["kind"] == "pelvis"]
    assert [(item["start"], item["end"]) for item in pelvis] == observed["rest_intervals"]
    assert [item["surface_height_m"] for item in pelvis] == pytest.approx([height])
    assert result.fit["contact_source"] == "kinematic"
    assert result.fit["pelvis_support_height_source"] == "shared_posed_posterior_body_surface"
