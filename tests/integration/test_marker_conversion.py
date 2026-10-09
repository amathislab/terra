"""Read real marker file formats and run the retained SMPL-H surface fitter."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from musclemimic.web_viewer.c3d_to_smpl import fit_smpl_to_c3d, save_motion_data_as_amass_smplh_npz
from terra.datasets.marker_fitting import _validate_smplh
from terra.mat import prepare_mat_marker_archive
from terra.trc import prepare_trc_marker_archive

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synthetic_markers(tmp_path_factory):
    models = os.environ.get("TERRA_TEST_MODEL_ROOT")
    if not models:
        pytest.skip("set TERRA_TEST_MODEL_ROOT to licensed neutral SMPL-H models")
    from musclemimic.web_viewer.c3d.markers import MOSHPP_SMPLH_MARKER_VIDS, build_marker_layout
    from musclemimic.web_viewer.c3d.smpl_models import _make_surface_model
    from musclemimic.web_viewer.c3d.surface_markers import SurfaceMarkerModel

    model, pose_dim = _make_surface_model(
        surface_model_type="smplh", smpl_model_path=models, gender="neutral", device="cpu"
    )
    layout = build_marker_layout(MOSHPP_SMPLH_MARKER_VIDS, surface_model_type="smplh")
    markers = SurfaceMarkerModel.from_layout(model, layout, pose_dim=pose_dim)
    positions = markers.initial_latents[:, [0, 2, 1]].copy()
    positions[:, 1] *= -1
    positions[:, 2] -= positions[:, 2].min() - 0.02
    positions = np.repeat(positions[None], 12, axis=0)
    positions[:, :, 0] += np.arange(len(positions))[:, None] * 0.001
    return tmp_path_factory.mktemp("marker-formats"), Path(models), positions, list(markers.labels)


@pytest.mark.parametrize("format_name", ["trc", "mat", "c3d"])
def test_marker_file_reaches_real_surface_fit(format_name, synthetic_markers):
    root, models, positions, labels = synthetic_markers
    source = root / f"markers.{format_name}"
    if format_name == "trc":
        header = (
            "PathFileType\t4\t(X/Y/Z)\tmarkers.trc\n"
            "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\n"
            f"100\t100\t{len(positions)}\t{len(labels)}\tmm\n"
            "Frame#\tTime\t" + "\t\t\t".join(labels) + "\t\t\n"
            "\t\t" + "\t".join(f"{axis}{i + 1}" for i in range(len(labels)) for axis in "XYZ") + "\n"
        )
        rows = [
            f"{i + 1}\t{i / 100:.2f}\t" + "\t".join(map(str, frame.ravel() * 1000)) for i, frame in enumerate(positions)
        ]
        source.write_text(header + "\n".join(rows) + "\n")
        prepared = prepare_trc_marker_archive(source, root / "cache", up_axis="z")
    elif format_name == "mat":
        savemat(source, {"markers": positions, "labels": np.asarray(labels), "fps": 100.0})
        schema = {
            "version": 1,
            "positions_path": "markers",
            "labels_path": "labels",
            "fps_path": "fps",
            "axis_order": "tmc",
            "units": "m",
            "axes": ["x", "y", "z"],
        }
        prepared = prepare_mat_marker_archive(source, schema, root / "cache")
    else:
        if importlib.util.find_spec("ezc3d") is None:
            pytest.skip("install the c3d extra")
        import ezc3d

        recording = ezc3d.c3d()
        recording["parameters"]["POINT"]["RATE"]["value"] = [100.0]
        recording["parameters"]["POINT"]["UNITS"]["value"] = ["mm"]
        recording["parameters"]["POINT"]["LABELS"]["value"] = labels
        points = np.ones((4, len(labels), len(positions)))
        points[:3] = positions.transpose(2, 1, 0) * 1000
        recording["data"]["points"] = points
        recording.write(str(source))
        prepared = source
    motion = fit_smpl_to_c3d(
        str(prepared),
        str(models),
        surface_model_type="smplh",
        gender="neutral",
        target_fps=50.0,
        device="cpu",
        stage1_iters=3,
        stage2_iters=5,
        n_ref_frames=3,
        stage1_shape_solver="joint_dogleg_jax",
        stage2_solver="batched_lbfgs",
        strict_frame_picking=False,
        enforce_knee_hinge=True,
    )
    output = root / f"{format_name}_poses.npz"
    save_motion_data_as_amass_smplh_npz(motion, output)
    frames, fps = _validate_smplh(output, enforce_knee_hinge=True)
    assert frames >= 3 and fps == 50.0
    error = motion["debug"]["marker_error"]["mean_mm"]
    assert np.isfinite(error) and error < 50.0
