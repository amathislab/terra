"""Contract tests for TERRA's public file-oriented retargeting API."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import terra.api as api
from loco_mujoco.core.terrain import BoxSpec, TerrainSpec
from terra.runtime import shape_cache_path
from terra.terrain.metadata import TerrainMetadata


def _write_smplh(path: Path, *, canonical: bool = False) -> None:
    poses = np.arange(3 * 156, dtype=float).reshape(3, 156) / 100.0
    fields = {
        "trans": np.zeros((3, 3)),
        "betas": np.linspace(-0.1, 0.1, 16),
        "gender": np.asarray(b"female"),
    }
    if canonical:
        fields.update(pose_aa=poses[:, :72], fps=np.asarray(60.0))
    else:
        fields.update(poses=poses, mocap_framerate=np.asarray(120.0))
    np.savez(path, **fields)


def _load_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _terrain() -> TerrainSpec:
    return TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.1), size=(1.0, 1.0, 0.1), name="terrain_box_0"),))


def test_public_boundary_loads_and_validates_environment_registrations():
    api.ensure_environment_registered("MyoFullBody")

    with pytest.raises(ValueError, match="unknown MuscleMimic environment"):
        api.ensure_environment_registered("NotARegisteredEnvironment")


def test_load_smplh_motion_accepts_amass_and_canonical_archives(tmp_path):
    amass_path = tmp_path / "amass.npz"
    canonical_path = tmp_path / "canonical.npz"
    _write_smplh(amass_path)
    _write_smplh(canonical_path, canonical=True)

    amass = api.load_smplh_motion(amass_path)
    canonical = api.load_smplh_motion(canonical_path)

    assert amass["pose_aa"].shape == (3, 72)
    np.testing.assert_allclose(amass["pose_aa"][:, 66:], 0.0)
    assert amass["fps"] == 120.0
    assert amass["gender"] == "female"
    assert canonical["pose_aa"].shape == (3, 72)
    assert canonical["fps"] == 60.0


def test_load_smplh_motion_accepts_amass_frame_rate_alias_and_single_row_betas(tmp_path):
    path = tmp_path / "alias.npz"
    _write_smplh(path)
    fields = _load_fields(path)
    fields["mocap_frame_rate"] = fields.pop("mocap_framerate")
    fields["betas"] = fields["betas"][None, :]
    fields["gender"] = np.asarray("NEUTRAL")
    np.savez(path, **fields)

    motion = api.load_smplh_motion(path)

    assert motion["fps"] == 120.0
    assert motion["betas"].shape == (16,)
    assert motion["gender"] == "neutral"


@pytest.mark.parametrize(
    ("missing", "message"),
    [
        ("trans", "missing required field.*trans"),
        ("betas", "missing required field.*betas"),
        ("gender", "missing required field.*gender"),
        ("poses", "must contain 'poses' or 'pose_aa'"),
        ("mocap_framerate", "missing fps/mocap_framerate"),
    ],
)
def test_load_smplh_motion_reports_missing_fields(tmp_path, missing, message):
    path = tmp_path / "missing.npz"
    _write_smplh(path)
    fields = _load_fields(path)
    fields.pop(missing)
    np.savez(path, **fields)

    with pytest.raises(ValueError, match=message):
        api.load_smplh_motion(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pose_aa", np.zeros((3, 71)), r"pose_aa must have shape \(frames, 72\)"),
        ("pose_aa", np.zeros((2, 72)), "at least 3 frames"),
        ("trans", np.zeros((3, 2)), r"trans must have shape \(3, 3\)"),
        ("betas", np.zeros(9), "at least 10 coefficients"),
        ("fps", np.asarray(0.0), "positive and finite"),
        ("fps", np.asarray([30.0, 60.0]), "frame rate must be scalar"),
        ("gender", np.asarray("robot"), "unsupported SMPL-H gender"),
        ("gender", np.asarray(["female", "male"]), "gender must be scalar"),
    ],
)
def test_load_smplh_motion_reports_invalid_fields(tmp_path, field, value, message):
    path = tmp_path / "invalid.npz"
    _write_smplh(path)
    fields = _load_fields(path)
    fields[field] = value
    np.savez(path, **fields)

    with pytest.raises(ValueError, match=message):
        api.load_smplh_motion(path)


def test_load_smplh_motion_rejects_nonfinite_motion(tmp_path):
    path = tmp_path / "bad.npz"
    _write_smplh(path)
    fields = _load_fields(path)
    fields["trans"][1, 2] = np.nan
    np.savez(path, **fields)

    with pytest.raises(ValueError, match="non-finite"):
        api.load_smplh_motion(path)


def test_load_smplh_motion_rejects_per_frame_shape_coefficients(tmp_path):
    path = tmp_path / "bad_betas.npz"
    _write_smplh(path)
    fields = _load_fields(path)
    fields["betas"] = np.zeros((3, 16))
    np.savez(path, **fields)

    with pytest.raises(ValueError, match="single-row"):
        api.load_smplh_motion(path)


@pytest.mark.parametrize("method", api.SUPPORTED_METHODS)
def test_retarget_smplh_dispatches_each_method_to_its_real_adapter(monkeypatch, tmp_path, method):
    source = tmp_path / "motion.npz"
    model_root = tmp_path / "models"
    cache_root = tmp_path / "cache"
    model_root.mkdir()
    _write_smplh(source)
    terrain = _terrain()
    calls = []

    monkeypatch.setattr(api, "load_robot_conf_file", lambda _env: SimpleNamespace(env_params={"x": 1}))

    def fake_shape(*args):
        calls.append(("shape", args[3]))
        args[3].parent.mkdir(parents=True, exist_ok=True)
        args[3].write_bytes(b"fitted shape")

    monkeypatch.setattr(api, "ensure_robot_shape", fake_shape)
    monkeypatch.setattr(
        api,
        "extend_motion",
        lambda env, params, trajectory, logger: ("extended", env, params, trajectory, logger.name),
    )

    def fake_gmr(*args):
        calls.append(("gmr", args))
        return "gmr-trajectory", {"adapter": "gmr"}

    def fake_smpl(*args):
        calls.append(("smpl", args))
        return "smpl-trajectory", {"adapter": "smpl"}

    def fake_terra(*args, **kwargs):
        calls.append(("terra", args, kwargs))
        return "terra-trajectory", {"adapter": args[4]["method_profile"]}

    monkeypatch.setattr(api, "fit_gmr_baseline", fake_gmr)
    monkeypatch.setattr(api, "fit_smpl_baseline", fake_smpl)
    monkeypatch.setattr(api, "fit_terra_motion", fake_terra)

    result = api.retarget_smplh(
        source,
        stability_policy="off",
        method=method,
        terrain=terrain,
        config=(
            {"damping": 0.25} if method == "gmr" else {"step_size": 0.15} if method in {"terra", "omniretarget"} else {}
        ),
        smpl_model_path=model_root,
        cache_root=cache_root,
    )

    assert result.method == method
    assert result.source_path == source.resolve()
    assert result.terrain == terrain
    assert result.analysis["source_format"] == "smplh"
    assert result.analysis["retargeting_method"] == method
    assert result.analysis["smpl_model_path"] == str(model_root.resolve())
    assert result.trajectory[0] == "extended"
    adapters = [call[0] for call in calls if call[0] in {"gmr", "smpl", "terra"}]
    assert adapters == ["terra" if method in {"terra", "omniretarget"} else method]

    if method == "gmr":
        gmr_config = calls[0][1][-1]
        assert gmr_config["target_fps"] == 30
        assert gmr_config["damping"] == 0.25
        assert gmr_config["smpl_model_path"] == str(model_root.resolve())
        assert gmr_config["cache_root"] == str(cache_root.resolve())
        assert gmr_config["terrain"] == terrain.to_dict()
    elif method in {"terra", "omniretarget"}:
        terra_call = next(call for call in calls if call[0] == "terra")
        assert terra_call[1][4]["method_profile"] == method
        assert terra_call[2]["smpl_model_path"] == model_root.resolve()
        assert terra_call[2]["fitted_shape_path"] == shape_cache_path("MyoFullBody", cache_root)
    if method != "gmr":
        assert result.analysis["fitted_shape_path"] == str(shape_cache_path("MyoFullBody", cache_root))


def test_retarget_smplh_can_reuse_shape_outside_experiment_cache(monkeypatch, tmp_path):
    source = tmp_path / "motion.npz"
    model_root = tmp_path / "models"
    cache_root = tmp_path / "experiment-cache"
    fitted_shape = tmp_path / "canonical-cache" / "MyoFullBody" / "shape_optimized.pkl"
    model_root.mkdir()
    fitted_shape.parent.mkdir(parents=True)
    fitted_shape.write_bytes(b"canonical fitted shape")
    _write_smplh(source)
    captured = {}

    monkeypatch.setattr(api, "load_robot_conf_file", lambda _env: SimpleNamespace(env_params={}))
    monkeypatch.setattr(
        api,
        "ensure_robot_shape",
        lambda *_args: pytest.fail("an explicit fitted shape must not be regenerated"),
    )
    monkeypatch.setattr(api, "extend_motion", lambda _env, _params, trajectory, _logger: trajectory)

    def fake_fit(*_args, **kwargs):
        captured.update(kwargs)
        trajectory = SimpleNamespace(
            data=SimpleNamespace(qpos=np.zeros((3, 7))),
            info=SimpleNamespace(frequency=100.0),
        )
        analysis = {
            "pos_error": np.zeros((3, 2)),
            "native_fps": 100.0,
            "numerical_envelope": {"max_root_step_m": 0.0},
        }
        return trajectory, analysis

    monkeypatch.setattr(api, "fit_terra_motion", fake_fit)

    result = api.retarget_smplh(
        source,
        terrain=_terrain(),
        smpl_model_path=model_root,
        cache_root=cache_root,
        fitted_shape_path=fitted_shape,
    )

    assert captured["fitted_shape_path"] == fitted_shape.resolve()
    assert result.analysis["fitted_shape_path"] == str(fitted_shape.resolve())
    assert result.analysis["stability_check"]["requires_retry"] is False


@pytest.mark.parametrize("suffix", (".c3d", ".trc", ".mat"))
def test_retarget_rejects_explicit_fitted_shape_for_marker_input(tmp_path, suffix):
    with pytest.raises(ValueError, match=r"already fitted \.npz"):
        api.retarget(tmp_path / f"motion{suffix}", fitted_shape_path=tmp_path / "shape.pkl")


@pytest.mark.parametrize("method", api.SUPPORTED_METHODS)
def test_retarget_c3d_forwards_method_specific_configuration(monkeypatch, tmp_path, method):
    source = tmp_path / "walk.c3d"
    c3d_models = tmp_path / "smplx"
    smplh_models = tmp_path / "smplh"
    cache_root = tmp_path / "cache"
    source.touch()
    c3d_models.mkdir()
    smplh_models.mkdir()
    captured = {}

    def fake_retarget(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return "trajectory", {"terrain": TerrainMetadata.from_terrain(_terrain()).to_dict()}

    monkeypatch.setattr(api, "retarget_c3d_to_trajectory", fake_retarget)
    result = api.retarget_c3d(
        source,
        stability_policy="off",
        method=method,
        terrain=_terrain(),
        config=(
            {"damping": 0.25} if method == "gmr" else {"step_size": 0.15} if method in {"terra", "omniretarget"} else {}
        ),
        c3d_model_path=c3d_models,
        smpl_model_path=smplh_models,
        cache_root=cache_root,
        c3d_options={"stage2_iters": 12},
    )

    assert captured["args"] == (str(source.resolve()), "MyoFullBody")
    options = captured["kwargs"]
    assert options["retargeting_method"] == method
    assert options["c3d_fit_model_path"] == str(c3d_models.resolve())
    assert options["retarget_smpl_model_path"] == str(smplh_models.resolve())
    assert options["cache_root"] == str(cache_root.resolve())
    assert options["device"] == "cpu"
    assert options["enforce_knee_hinge"] is False
    assert options["gender"] == "neutral"
    assert options["least_avail_markers"] == 1.0
    assert options["n_ref_frames"] == 12
    assert options["optimize_toes"] is True
    assert options["seed"] == 100
    assert options["stage1_iters"] == 320
    assert options["stage1_shape_solver"] == "joint_dogleg_jax"
    assert options["stage1_state_path"] is None
    assert options["stage2_iters"] == 12
    assert options["stage2_marker_weight_overrides"] is None
    assert options["stage2_solver"] == "frame_lbfgs"
    assert options["stage2_torso_frame_weight"] == 0.0
    assert options["strict_frame_picking"] is True
    assert options["surface_model_type"] == "smplx"
    assert options["target_fps"] is None
    assert options["wrist_markers_on_stick"] is False
    assert result.terrain == _terrain()
    assert result.analysis["source_format"] == "c3d"

    if method == "gmr":
        assert options["gmr_config"]["target_fps"] == 30
        assert options["gmr_config"]["damping"] == 0.25
        assert options["gmr_config"]["smpl_model_path"] == str(smplh_models.resolve())
        assert options["gmr_config"]["cache_root"] == str(cache_root.resolve())
        assert options["terra_config"] is None
    elif method in {"terra", "omniretarget"}:
        assert options["gmr_config"] is None
        assert options["terra_config"]["step_size"] == 0.15
        assert options["terra_config"]["terrain"] == _terrain().to_dict()
    else:
        assert options["gmr_config"] is None
        assert options["terra_config"] is None


@pytest.mark.parametrize("option", ["retargeting_method", "cache_root", "c3d_fit_model_path"])
def test_retarget_c3d_rejects_managed_option_override(tmp_path, option):
    source = tmp_path / "walk.c3d"
    source.touch()

    with pytest.raises(ValueError, match="cannot override"):
        api.retarget_c3d(
            source,
            terrain=_terrain(),
            c3d_options={option: "managed"},
        )


def test_retarget_c3d_rejects_unknown_option_before_entering_dependency(monkeypatch, tmp_path):
    source = tmp_path / "walk.c3d"
    source.touch()
    monkeypatch.setattr(
        api,
        "retarget_c3d_to_trajectory",
        lambda *_args, **_kwargs: pytest.fail("dependency adapter should not be called"),
    )

    with pytest.raises(ValueError, match=r"unknown c3d_options field\(s\): stage3_iters"):
        api.retarget_c3d(
            source,
            terrain=_terrain(),
            c3d_options={"stage3_iters": 12},
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("gender", "robot", "gender must be one of"),
        ("gender", 1, "gender must be a string"),
        ("surface_model_type", "smpl", "surface_model_type must be one of"),
        ("surface_model_type", None, "surface_model_type must be a string"),
        ("optimize_toes", 1, "optimize_toes must be a boolean"),
        ("clear_cache", "false", "clear_cache must be a boolean"),
        ("enforce_knee_hinge", 1, "enforce_knee_hinge must be a boolean"),
        ("strict_frame_picking", None, "strict_frame_picking must be a boolean"),
        ("wrist_markers_on_stick", "yes", "wrist_markers_on_stick must be a boolean"),
        ("n_ref_frames", 0, "n_ref_frames must be a positive integer"),
        ("stage1_iters", 0, "stage1_iters must be a positive integer"),
        ("stage1_iters", True, "stage1_iters must be a positive integer"),
        ("stage2_iters", 1.5, "stage2_iters must be a positive integer"),
        ("stage1_shape_solver", "lbfgs", "stage1_shape_solver must be one of"),
        ("stage2_solver", "dogleg", "stage2_solver must be one of"),
        ("seed", -1, "seed must be an integer"),
        ("seed", True, "seed must be an integer"),
        ("target_fps", 0, "target_fps must be positive and finite or null"),
        ("target_fps", float("nan"), "target_fps must be positive and finite or null"),
        ("least_avail_markers", 0, r"least_avail_markers must be in \(0, 1\]"),
        ("least_avail_markers", 1.1, r"least_avail_markers must be in \(0, 1\]"),
        ("stage2_torso_frame_weight", -1, "must be finite and non-negative"),
        ("device", "gpu", "device must be 'cpu', 'cuda', or 'cuda:<index>'"),
        ("stage2_marker_weight_overrides", [], "must be an object or null"),
        ("stage2_marker_weight_overrides", {"LHEE": 0}, "must be positive and finite"),
        ("stage2_marker_weight_overrides", {1: 2}, "labels must be non-empty strings"),
        ("converted_c3d_name", 12, "converted_c3d_name must be a string or path"),
        ("head_marker_corr_path", {}, "head_marker_corr_path must be a string or path"),
        ("pose_body_prior_path", [], "pose_body_prior_path must be a string or path"),
    ],
)
def test_retarget_c3d_rejects_invalid_option_before_entering_dependency(
    monkeypatch,
    tmp_path,
    option,
    value,
    message,
):
    source = tmp_path / "walk.c3d"
    source.touch()
    monkeypatch.setattr(
        api,
        "retarget_c3d_to_trajectory",
        lambda *_args, **_kwargs: pytest.fail("dependency adapter should not be called"),
    )

    with pytest.raises(ValueError, match=message):
        api.retarget_c3d(
            source,
            terrain=_terrain(),
            c3d_options={option: value},
        )


def test_retarget_c3d_forwards_advanced_fit_options(monkeypatch, tmp_path):
    source = tmp_path / "ramp.c3d"
    source.touch()
    c3d_models = tmp_path / "smplx"
    smplh_models = tmp_path / "smplh"
    cache_root = tmp_path / "cache"
    c3d_models.mkdir()
    smplh_models.mkdir()
    stage1_state = tmp_path / "subject_stage1.npz"
    stage1_state.touch()
    captured = {}

    def fake_retarget(*_args, **kwargs):
        captured.update(kwargs)
        return "trajectory", {}

    monkeypatch.setattr(api, "retarget_c3d_to_trajectory", fake_retarget)

    api.retarget_c3d(
        source,
        stability_policy="off",
        terrain=_terrain(),
        c3d_model_path=c3d_models,
        smpl_model_path=smplh_models,
        cache_root=cache_root,
        c3d_options={
            "device": "CUDA:1",
            "enforce_knee_hinge": True,
            "least_avail_markers": 0.8,
            "n_ref_frames": 16,
            "seed": 7,
            "stage1_state_path": stage1_state,
            "stage2_marker_weight_overrides": {"LHEE": 2},
            "stage2_solver": "BATCHED_LBFGS",
            "stage2_torso_frame_weight": 3,
            "strict_frame_picking": False,
            "target_fps": 50,
            "wrist_markers_on_stick": True,
        },
    )

    assert captured["device"] == "cuda:1"
    assert captured["enforce_knee_hinge"] is True
    assert captured["least_avail_markers"] == 0.8
    assert captured["n_ref_frames"] == 16
    assert captured["seed"] == 7
    assert captured["stage1_state_path"] == str(stage1_state.resolve())
    assert captured["stage2_marker_weight_overrides"] == {"LHEE": 2.0}
    assert captured["stage2_solver"] == "batched_lbfgs"
    assert captured["stage2_torso_frame_weight"] == 3.0
    assert captured["strict_frame_picking"] is False
    assert captured["target_fps"] == 50.0
    assert captured["wrist_markers_on_stick"] is True


def test_retarget_c3d_requires_explicit_marker_model_root(tmp_path):
    source = tmp_path / "walk.c3d"
    source.touch()

    with pytest.raises(ValueError, match="c3d_model_path is required"):
        api.retarget_c3d(source, terrain=_terrain())


def test_retarget_trc_prepares_marker_archive_and_forwards_to_shared_fitter(monkeypatch, tmp_path):
    source = tmp_path / "walk.trc"
    marker_archive = tmp_path / "cache" / ".trc_marker_cache" / "markers.npz"
    c3d_models = tmp_path / "smplx"
    smplh_models = tmp_path / "smplh"
    cache_root = tmp_path / "cache"
    source.touch()
    marker_archive.parent.mkdir(parents=True)
    marker_archive.touch()
    c3d_models.mkdir()
    smplh_models.mkdir()
    captured = {}

    def fake_prepare(path, root, *, up_axis):
        captured.update(prepared_path=path, prepared_root=root, up_axis=up_axis)
        return marker_archive

    def fake_retarget(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return "trajectory", {}

    monkeypatch.setattr(api, "prepare_trc_marker_archive", fake_prepare)
    monkeypatch.setattr(api, "retarget_c3d_to_trajectory", fake_retarget)

    result = api.retarget_trc(
        source,
        stability_policy="off",
        terrain=_terrain(),
        c3d_model_path=c3d_models,
        smpl_model_path=smplh_models,
        cache_root=cache_root,
        trc_up_axis="z",
    )

    assert captured["prepared_path"] == source.resolve()
    assert captured["prepared_root"] == cache_root.resolve()
    assert captured["up_axis"] == "z"
    assert captured["args"] == (str(marker_archive.resolve()), "MyoFullBody")
    assert result.source_path == source.resolve()
    assert result.analysis["source"] == "trc"
    assert result.analysis["source_format"] == "trc"
    assert result.analysis["trc_up_axis"] == "z"
    assert result.analysis["source_marker_archive_path"] == str(marker_archive.resolve())


def test_retarget_trc_rejects_unknown_vertical_axis_before_model_resolution(tmp_path):
    source = tmp_path / "walk.trc"
    source.touch()

    with pytest.raises(ValueError, match="trc_up_axis"):
        api.retarget_trc(source, trc_up_axis="x")  # type: ignore[arg-type]


def test_retarget_mat_prepares_marker_archive_and_forwards_to_shared_fitter(monkeypatch, tmp_path):
    source = tmp_path / "walk.mat"
    marker_archive = tmp_path / "cache" / ".mat_marker_cache" / "markers.npz"
    marker_models = tmp_path / "smplx"
    smplh_models = tmp_path / "smplh"
    cache_root = tmp_path / "cache"
    schema = {"positions_path": "markers", "labels": ["LANK"], "fps": 100}
    selectors = {"trial": 2}
    source.touch()
    marker_archive.parent.mkdir(parents=True)
    marker_archive.touch()
    marker_models.mkdir()
    smplh_models.mkdir()
    captured = {}

    def fake_prepare(path, received_schema, root, *, selectors):
        captured.update(
            prepared_path=path,
            schema=received_schema,
            prepared_root=root,
            selectors=selectors,
        )
        return marker_archive

    def fake_retarget(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return "trajectory", {}

    monkeypatch.setattr(api, "prepare_mat_marker_archive", fake_prepare)
    monkeypatch.setattr(api, "retarget_c3d_to_trajectory", fake_retarget)

    result = api.retarget_mat(
        source,
        stability_policy="off",
        mat_schema=schema,
        mat_selectors=selectors,
        terrain=_terrain(),
        c3d_model_path=marker_models,
        smpl_model_path=smplh_models,
        cache_root=cache_root,
    )

    assert captured["prepared_path"] == source.resolve()
    assert captured["prepared_root"] == cache_root.resolve()
    assert captured["schema"] == schema
    assert captured["selectors"] == selectors
    assert captured["args"] == (str(marker_archive.resolve()), "MyoFullBody")
    assert result.source_path == source.resolve()
    assert result.analysis["source_format"] == "mat"
    assert result.analysis["mat_schema"] == schema
    assert result.analysis["mat_selectors"] == selectors
    assert result.analysis["source_marker_archive_path"] == str(marker_archive.resolve())


def test_retarget_c3d_rejects_missing_stage1_state_before_dependency(monkeypatch, tmp_path):
    source = tmp_path / "walk.c3d"
    source.touch()
    monkeypatch.setattr(
        api,
        "retarget_c3d_to_trajectory",
        lambda *_args, **_kwargs: pytest.fail("dependency adapter should not be called"),
    )

    with pytest.raises(FileNotFoundError, match="C3D Stage-I state not found"):
        api.retarget_c3d(
            source,
            terrain=_terrain(),
            c3d_options={"stage1_state_path": tmp_path / "missing.npz"},
        )


@pytest.mark.parametrize("entrypoint", [api.retarget_smplh, api.retarget_c3d, api.retarget_trc])
def test_smpl_baseline_rejects_ignored_configuration(entrypoint, tmp_path):
    suffix = ".npz" if entrypoint is api.retarget_smplh else ".trc" if entrypoint is api.retarget_trc else ".c3d"
    source = tmp_path / f"walk{suffix}"
    source.touch()

    with pytest.raises(ValueError, match="does not accept configuration overrides"):
        entrypoint(source, method="smpl", terrain=_terrain(), config={"ignored": True})


def test_retarget_dispatches_by_file_extension(monkeypatch, tmp_path):
    npz = tmp_path / "walk.NPZ"
    c3d = tmp_path / "walk.C3D"
    trc = tmp_path / "walk.TRC"
    mat = tmp_path / "walk.MAT"
    monkeypatch.setattr(api, "retarget_smplh", lambda path, **kwargs: ("smplh", path, kwargs))
    monkeypatch.setattr(api, "retarget_c3d", lambda path, **kwargs: ("c3d", path, kwargs))
    monkeypatch.setattr(api, "retarget_trc", lambda path, **kwargs: ("trc", path, kwargs))
    monkeypatch.setattr(api, "retarget_mat", lambda path, **kwargs: ("mat", path, kwargs))

    assert api.retarget(npz, method="gmr")[0] == "smplh"
    assert api.retarget(c3d, method="terra", stability_policy="off")[0] == "c3d"
    assert api.retarget(trc, method="terra", stability_policy="off")[0] == "trc"
    assert api.retarget(mat, mat_schema={}, method="terra", stability_policy="off")[0] == "mat"
    with pytest.raises(ValueError, match=r"expected \.npz, \.c3d, \.trc, or \.mat"):
        api.retarget(tmp_path / "walk.bvh")


def test_retarget_rejects_c3d_options_for_smplh_input(tmp_path):
    with pytest.raises(ValueError, match=r"require a marker source"):
        api.retarget(tmp_path / "walk.npz", c3d_options={"stage2_iters": 12})


def test_retarget_requires_schema_for_mat_input(tmp_path):
    with pytest.raises(ValueError, match="mat_schema is required"):
        api.retarget(tmp_path / "walk.mat", stability_policy="off")


def test_retarget_rejects_mat_schema_for_other_inputs(tmp_path):
    with pytest.raises(ValueError, match=r"require a \.mat source"):
        api.retarget(tmp_path / "walk.trc", mat_schema={})


def test_public_api_exports_are_importable():
    from terra import (
        RetargetArtifacts,
        RetargetResult,
        ValidatedRetargetArtifacts,
        artifacts,
        load_retarget_analysis,
        load_smplh_motion,
        normalize_motion_name,
        retarget,
        retarget_c3d,
        retarget_mat,
        retarget_smplh,
        retarget_trc,
        save_retarget_result,
        smplh,
        validate_retarget_artifacts,
    )

    assert RetargetArtifacts is artifacts.RetargetArtifacts
    assert RetargetArtifacts is api.RetargetArtifacts
    assert RetargetResult is api.RetargetResult
    assert ValidatedRetargetArtifacts is artifacts.ValidatedRetargetArtifacts
    assert ValidatedRetargetArtifacts is api.ValidatedRetargetArtifacts
    assert load_retarget_analysis is artifacts.load_retarget_analysis
    assert load_retarget_analysis is api.load_retarget_analysis
    assert load_smplh_motion is smplh.load_smplh_motion
    assert load_smplh_motion is api.load_smplh_motion
    assert normalize_motion_name is artifacts.normalize_motion_name
    assert normalize_motion_name is api.normalize_motion_name
    assert retarget is api.retarget
    assert retarget_c3d is api.retarget_c3d
    assert retarget_mat is api.retarget_mat
    assert retarget_smplh is api.retarget_smplh
    assert retarget_trc is api.retarget_trc
    assert save_retarget_result is artifacts.save_retarget_result
    assert save_retarget_result is api.save_retarget_result
    assert validate_retarget_artifacts is artifacts.validate_retarget_artifacts
    assert validate_retarget_artifacts is api.validate_retarget_artifacts


class _SerializableTrajectory:
    def save(self, path: str) -> None:
        np.savez(path, qpos=np.zeros((4, 8)), qvel=np.ones((4, 7)), frequency=np.asarray(100.0))


def test_save_retarget_result_rejects_conflicting_terrain_representations(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    recorded_terrain = _terrain()
    result = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={"terrain": TerrainMetadata.from_terrain(recorded_terrain).to_dict()},
        terrain=TerrainSpec(boxes=(BoxSpec(pos=(0.0, 0.0, 0.2), size=(1.0, 1.0, 0.2), name="different"),)),
        method="terra",
        source_path=source,
    )

    with pytest.raises(ValueError, match="does not match"):
        api.save_retarget_result(result, tmp_path / "cache", "Study/Trial")


def test_save_retarget_result_rejects_analysis_terrain_without_playback_terrain(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    result = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={"terrain": TerrainMetadata.from_terrain(_terrain()).to_dict()},
        terrain=None,
        method="terra",
        source_path=source,
    )

    with pytest.raises(ValueError, match=r"result\.terrain is None"):
        api.save_retarget_result(result, tmp_path / "cache", "Study/Trial")


def test_published_artifacts_remain_valid_after_cache_relocation(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    terrain = _terrain()
    result = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={"terrain": TerrainMetadata.from_terrain(terrain).to_dict()},
        terrain=terrain,
        method="terra",
        source_path=source,
    )
    original_root = tmp_path / "original"
    relocated_root = tmp_path / "relocated"
    api.save_retarget_result(result, original_root, "Study/Subject/Trial")
    shutil.copytree(original_root, relocated_root)

    validated = api.validate_retarget_artifacts(
        relocated_root,
        "Study/Subject/Trial",
        require_nonflat_terrain=True,
    )

    assert validated.trajectory_path.is_relative_to(relocated_root)
    assert validated.terrain_path is not None
    assert validated.terrain_path.is_relative_to(relocated_root)


def test_save_retarget_result_discards_stale_terrain_metadata_for_flat_result(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    result = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={
            "terrain_file": "stale_terrain.json",
            "terrain_path": "/stale/upstream/terrain.json",
        },
        terrain=None,
        method="terra",
        source_path=source,
    )

    artifacts = api.save_retarget_result(result, tmp_path / "cache", "Study/Flat")

    analysis = api.load_retarget_analysis(artifacts.analysis_path)
    assert "terrain_file" not in analysis
    assert "terrain_path" not in analysis
    assert artifacts.terrain_path is None
    validated = api.validate_retarget_artifacts(tmp_path / "cache", "Study/Flat")
    assert validated.terrain_path is None
    assert validated.nonflat_terrain is False


def test_save_retarget_result_overwrites_nonflat_artifacts_with_clean_flat_set(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    cache_root = tmp_path / "cache"
    motion_name = "Study/Replacement"
    terrain = _terrain()
    nonflat = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={"terrain": TerrainMetadata.from_terrain(terrain).to_dict()},
        terrain=terrain,
        method="terra",
        source_path=source,
    )
    old_artifacts = api.save_retarget_result(nonflat, cache_root, motion_name)
    assert old_artifacts.terrain_path is not None

    flat = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={
            "terrain_path": str(old_artifacts.terrain_path),
        },
        terrain=None,
        method="terra",
        source_path=source,
    )
    artifacts = api.save_retarget_result(
        flat,
        cache_root,
        motion_name,
        overwrite=True,
    )

    assert artifacts.terrain_path is None
    assert not old_artifacts.terrain_path.exists()
    validated = api.validate_retarget_artifacts(cache_root, motion_name)
    assert validated.terrain_path is None
    assert validated.nonflat_terrain is False


def test_save_retarget_result_does_not_publish_partial_files(monkeypatch, tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    terrain = _terrain()
    result = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={"terrain": TerrainMetadata.from_terrain(terrain).to_dict()},
        terrain=terrain,
        method="terra",
        source_path=source,
    )
    paths = api.retarget_cache_paths(tmp_path / "cache", "Study/Trial")

    def fail_terrain_save(_self, _path):
        raise RuntimeError("simulated terrain write failure")

    monkeypatch.setattr(TerrainMetadata, "save", fail_terrain_save)

    with pytest.raises(RuntimeError, match="terrain write failure"):
        api.save_retarget_result(result, tmp_path / "cache", "Study/Trial")

    assert not paths.trajectory_path.exists()
    assert not paths.analysis_path.exists()
    assert not paths.terrain_path.exists()


def test_save_retarget_result_rejects_invalid_trajectory_before_publication(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)

    class InvalidTrajectory:
        def save(self, path: str) -> None:
            np.savez(path, qpos=np.zeros((4, 8)), qvel=np.zeros((3, 7)), frequency=np.asarray(100.0))

    result = api.RetargetResult(
        trajectory=InvalidTrajectory(),
        analysis={},
        terrain=None,
        method="terra",
        source_path=source,
    )
    paths = api.retarget_cache_paths(tmp_path / "cache", "Study/Trial")

    with pytest.raises(ValueError, match="trajectory qvel must have shape"):
        api.save_retarget_result(result, tmp_path / "cache", "Study/Trial")

    assert not paths.trajectory_path.exists()
    assert not paths.analysis_path.exists()


def test_save_retarget_result_rejects_complex_trajectory_state(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)

    class ComplexTrajectory:
        def save(self, path: str) -> None:
            np.savez(
                path,
                qpos=np.zeros((4, 8), dtype=np.complex128),
                qvel=np.zeros((4, 7)),
                frequency=np.asarray(100.0),
            )

    result = api.RetargetResult(
        trajectory=ComplexTrajectory(),
        analysis={},
        terrain=None,
        method="terra",
        source_path=source,
    )
    paths = api.retarget_cache_paths(tmp_path / "cache", "Study/Trial")

    with pytest.raises(ValueError, match="real numeric arrays"):
        api.save_retarget_result(result, tmp_path / "cache", "Study/Trial")

    assert not paths.trajectory_path.exists()
    assert not paths.analysis_path.exists()


def test_save_retarget_result_identifies_unserializable_analysis_field(tmp_path):
    source = tmp_path / "source.npz"
    _write_smplh(source)
    result = api.RetargetResult(
        trajectory=_SerializableTrajectory(),
        analysis={"unsupported": object()},
        terrain=None,
        method="terra",
        source_path=source,
    )

    with pytest.raises(ValueError, match="analysis value 'unsupported' is not safely serializable"):
        api.save_retarget_result(result, tmp_path / "cache", "Study/Trial")


@pytest.mark.parametrize(
    "name",
    [
        "../trial",
        "/tmp/trial",
        "subject/../trial",
        "subject/./trial",
        "subject//trial",
        "subject/trial:1",
        "C:/trial",
        "",
        ".",
    ],
)
def test_motion_name_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match="motion name"):
        api.normalize_motion_name(name)


def test_motion_name_accepts_internal_spaces():
    name = "EyesJapan/subject/sitdown-04-chair elbow_poses.npz"
    assert api.normalize_motion_name(name).as_posix() == ("EyesJapan/subject/sitdown-04-chair elbow_poses")


def test_motion_name_strips_trc_extension():
    assert api.normalize_motion_name("Study/Subject/Trial.trc").as_posix() == "Study/Subject/Trial"


def test_motion_name_strips_mat_extension():
    assert api.normalize_motion_name("Study/Subject/Trial.mat").as_posix() == "Study/Subject/Trial"


@pytest.mark.parametrize("name", ["subject /trial", "subject/ trial"])
def test_motion_name_rejects_component_edge_spaces(name):
    with pytest.raises(ValueError, match="motion name"):
        api.normalize_motion_name(name)


@pytest.mark.parametrize(
    ("function_name", "extra"),
    (("retarget_c3d", {}), ("retarget_trc", {}), ("retarget_mat", {"mat_schema": {}})),
)
def test_direct_marker_api_retries_an_unstable_terra_trajectory(monkeypatch, function_name, extra):
    configs = []

    def fake_marker(*_args, **kwargs):
        config = kwargs["config"] or {}
        configs.append(config)
        unstable = float(config.get("step_size", 0.2)) > 0.1
        qpos = np.zeros((3, 7))
        qpos[1:, 0] = 0.04 if unstable else 0.01
        errors = np.full((3, 2), 0.04)
        errors[0, 0] = 0.4 if unstable else 0.2
        return SimpleNamespace(
            trajectory=SimpleNamespace(
                data=SimpleNamespace(qpos=qpos),
                info=SimpleNamespace(frequency=100.0),
            ),
            analysis={
                "pos_error": errors,
                "native_fps": 100.0,
                "numerical_envelope": {"max_root_step_m": 0.01},
            },
        )

    monkeypatch.setattr(api, "_retarget_marker_trajectory", fake_marker)
    result = getattr(api, function_name)("motion.c3d", config={"step_size": 0.2}, **extra)

    assert configs == [{"step_size": 0.2}, {"step_size": 0.1}]
    assert result.analysis["stability_retry"]["triggered"] is True
