"""Ownership and separation invariants for TERRA's comparison baselines."""

import os
from types import MappingProxyType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from terra.baselines import (
    BASELINES,
    GMR_BASELINE,
    OMNIRETARGET_BASELINE,
    SMPL_BASELINE,
    baseline_spec,
    gmr,
    omniretarget,
    smpl,
)
from terra.baselines.temporal import resample_smplh_motion
from terra.profiles import active_qp_terms


def test_all_three_baselines_have_explicit_terra_owned_definitions():
    assert tuple(BASELINES) == ("omniretarget", "gmr", "smpl")
    assert baseline_spec("omniretarget") is OMNIRETARGET_BASELINE
    assert baseline_spec("gmr") is GMR_BASELINE
    assert baseline_spec("smpl") is SMPL_BASELINE
    assert {spec.label for spec in BASELINES.values()} == {
        "OmniRetarget",
        "GMR",
        "MuscleMimic SMPL-fit",
    }
    assert "terra" not in BASELINES


def test_baseline_configs_are_immutable_and_resolve_by_copy():
    assert isinstance(GMR_BASELINE.config, MappingProxyType)
    with pytest.raises(TypeError):
        GMR_BASELINE.config["solver"] = "other"
    resolved = GMR_BASELINE.resolved_config({"damping": 0.25})
    assert resolved["damping"] == 0.25
    assert GMR_BASELINE.config["damping"] == 0.5


def test_gmr_adapter_delegates_with_default_configuration(monkeypatch):
    captured = {}

    def fake_fit(*args):
        captured["args"] = args
        return "trajectory", "analysis"

    monkeypatch.setattr(gmr, "_fit_gmr_motion", fake_fit)
    result = gmr.fit_motion("env", "robot", "motion", "logger", {"damping": 0.25})
    config = captured["args"][-1]
    assert result == ("trajectory", "analysis")
    assert config["solver"] == "daqp"
    assert config["damping"] == 0.25
    assert config["use_velocity_limit"] is True
    assert "algorithm" not in config
    assert "allow_cache_download" not in config


def test_gmr_adapter_scopes_backend_paths_and_restores_environment(monkeypatch, tmp_path):
    seen = {}

    def fake_fit(*args):
        seen["cache"] = os.environ["CONVERTED_AMASS_PATH"]
        seen["models"] = os.environ["SMPL_MODEL_PATH"]
        return "trajectory", "analysis"

    monkeypatch.setenv("CONVERTED_AMASS_PATH", "original-cache")
    monkeypatch.setenv("SMPL_MODEL_PATH", "original-models")
    monkeypatch.setattr(gmr, "_fit_gmr_motion", fake_fit)

    result = gmr.fit_motion(
        "env",
        "robot",
        "motion",
        "logger",
        {"cache_root": tmp_path / "cache", "smpl_model_path": tmp_path / "models"},
    )

    assert result == ("trajectory", "analysis")
    assert seen == {
        "cache": str((tmp_path / "cache").resolve()),
        "models": str((tmp_path / "models").resolve()),
    }
    assert os.environ["CONVERTED_AMASS_PATH"] == "original-cache"
    assert os.environ["SMPL_MODEL_PATH"] == "original-models"


def test_gmr_shape_is_prepared_once_under_the_explicit_cache(monkeypatch, tmp_path):
    from loco_mujoco.smpl import retargeting

    shape = tmp_path / "cache/MyoFullBody/gmr/myofullbody_shape.pkl"
    metadata = shape.with_name("myofullbody_shape_metadata.json")

    def fake_prepare(*, env_name, iterations):
        assert env_name == "MyoFullBody"
        assert iterations == 17
        assert os.environ["CONVERTED_AMASS_PATH"] == str((tmp_path / "cache").resolve())
        assert os.environ["SMPL_MODEL_PATH"] == str((tmp_path / "models").resolve())
        shape.parent.mkdir(parents=True)
        shape.write_bytes(b"shape")
        metadata.write_text("{}")
        return str(shape)

    monkeypatch.setattr(retargeting, "prepare_gmr_fitted_shape", fake_prepare)

    prepared = gmr.prepare_fitted_shape(
        "MyoFullBody",
        tmp_path / "cache",
        tmp_path / "models",
        iterations=17,
    )

    assert prepared == shape.resolve()


def test_smpl_adapter_delegates_to_musclemimic_public_api(monkeypatch):
    captured = {}

    def fake_fit(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return "trajectory", "analysis"

    monkeypatch.setattr(smpl, "_fit_smpl_motion", fake_fit)
    result = smpl.fit_motion("env", "robot", "models", "motion", "shape", "logger", visualize=True)
    assert result == ("trajectory", "analysis")
    assert captured["args"] == ("env", "robot", "models", "motion", "shape", "logger")
    assert captured["kwargs"] == {"skip_steps": True, "visualize": True}


def test_exact_smpl_rate_resamples_every_joint_without_subsampling(monkeypatch):
    captured = {}
    pose = np.zeros((3, 6), dtype=np.float32)
    pose[-1, 2] = np.pi / 2
    motion = {
        "pose_aa": pose,
        "trans": np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "gender": "neutral",
        "fps": 50.0,
    }

    def fake_fit(*args, **kwargs):
        captured.update(motion=args[3], kwargs=kwargs)
        return "trajectory", {"native_fps": 100.0}

    monkeypatch.setattr(smpl, "_fit_smpl_motion", fake_fit)
    trajectory, analysis = smpl.fit_motion(
        "env",
        "robot",
        "models",
        motion,
        "shape",
        "logger",
        {"target_fps": 100.0},
    )

    assert trajectory == "trajectory"
    assert captured["kwargs"] == {"skip_steps": False, "visualize": False}
    assert captured["motion"]["fps"] == 100.0
    assert captured["motion"]["pose_aa"].shape == (5, 6)
    assert captured["motion"]["trans"][:, 0] == pytest.approx([0, 0.5, 1, 1.5, 2])
    assert Rotation.from_rotvec(captured["motion"]["pose_aa"][3, :3]).magnitude() == pytest.approx(np.pi / 4)
    assert analysis["solve_rate_source_fps"] == 50.0
    assert analysis["solve_rate_target_fps"] == 100.0


def test_gmr_exact_rate_materializes_a_true_upsampled_archive(monkeypatch, tmp_path):
    source = tmp_path / "motion.npz"
    poses = np.zeros((3, 66), dtype=np.float32)
    poses[-1, 2] = np.pi / 2
    np.savez(
        source,
        poses=poses,
        trans=np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        betas=np.zeros(16, dtype=np.float32),
        gender=np.asarray("neutral"),
        mocap_framerate=np.asarray(50.0),
    )
    captured = {}

    def fake_fit(_env, _robot, resampled_source, _logger, config):
        with np.load(resampled_source, allow_pickle=False) as archive:
            captured.update(
                poses=np.asarray(archive["poses"]),
                root_orient=np.asarray(archive["root_orient"]),
                pose_body=np.asarray(archive["pose_body"]),
                left_hand_pose=np.asarray(archive["left_hand_pose"]),
                right_hand_pose=np.asarray(archive["right_hand_pose"]),
                trans=np.asarray(archive["trans"]),
                fps=float(np.asarray(archive["mocap_framerate"])),
                standard_fps=float(np.asarray(archive["mocap_frame_rate"])),
                config=config,
            )
        return "trajectory", {"native_fps": 100.0}

    monkeypatch.setattr(gmr, "_fit_gmr_motion", fake_fit)
    trajectory, analysis = gmr.fit_motion(
        "env",
        "robot",
        str(source),
        "logger",
        {"target_fps": 100.0, "exact_target_fps": True},
    )

    assert trajectory == "trajectory"
    assert captured["poses"].shape == (5, 156)
    assert captured["root_orient"].shape == (5, 3)
    assert captured["pose_body"].shape == (5, 63)
    assert captured["left_hand_pose"].shape == (5, 45)
    assert captured["right_hand_pose"].shape == (5, 45)
    assert captured["trans"][:, 0] == pytest.approx([0, 0.5, 1, 1.5, 2])
    assert captured["fps"] == 100.0
    assert captured["standard_fps"] == 100.0
    assert captured["config"]["target_fps"] == 100.0
    assert "exact_target_fps" not in captured["config"]
    assert analysis["solve_rate_target_frames"] == 5


@pytest.mark.parametrize("target", (0, -1, np.nan, True, "100"))
def test_temporal_resampling_rejects_invalid_target_rate(target):
    motion = {
        "pose_aa": np.zeros((2, 3)),
        "trans": np.zeros((2, 3)),
        "fps": 50.0,
    }
    with pytest.raises(ValueError, match="target_fps"):
        resample_smplh_motion(motion, target)


def test_omniretarget_adapter_forces_contribution_free_profile(monkeypatch):
    captured = {}

    def fake_fit(env_name, robot_conf, motion_data, logger, config):
        captured["config"] = config
        return "trajectory", "analysis"

    monkeypatch.setattr("terra.pipeline.fit_terra_motion", fake_fit)
    result = omniretarget.fit_motion("env", "robot", "motion", "logger", {"orient_weight": 99})
    assert result == ("trajectory", "analysis")
    assert captured["config"]["method_profile"] == "omniretarget"
    assert active_qp_terms(captured["config"]) == []


def test_omniretarget_always_uses_fitted_shape_without_terra_terms():
    from terra.profiles import resolve_method_profile

    fitted = resolve_method_profile("omniretarget", {})
    attempted_override = resolve_method_profile("omniretarget", {"use_fitted_shape": False, "calibrate_sites": False})

    assert (fitted["use_fitted_shape"], fitted["calibrate_sites"]) == (True, True)
    assert (attempted_override["use_fitted_shape"], attempted_override["calibrate_sites"]) == (True, True)
    assert active_qp_terms(fitted) == []

    with pytest.raises(ValueError, match="unknown TERRA solver configuration"):
        resolve_method_profile("omniretarget", {"omniretarget_source_convention": "native"})


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_spec("terra")
