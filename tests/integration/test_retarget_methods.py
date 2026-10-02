"""Run each published method through the actual CLI, solver, and artifact reader."""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from terra import validate_retarget_artifacts
from terra.benchmarking.reconstruction.cli import cohort_main
from terra.cli import main as retarget_main
from terra.commands.run import main as run_main
from terra.evaluation.dataset import main as evaluate_main

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def motion_inputs(tmp_path_factory):
    source = os.environ.get("TERRA_TEST_MOTION")
    models = os.environ.get("TERRA_TEST_MODEL_ROOT")
    if not source or not models:
        pytest.skip("set TERRA_TEST_MOTION and TERRA_TEST_MODEL_ROOT to licensed local inputs")
    root = tmp_path_factory.mktemp("retarget-methods")
    # A short sequence keeps all four real solvers practical in the release checks.
    with np.load(source, allow_pickle=False) as archive:
        data = dict(archive)
    frames = len(data["trans"])
    data = {key: value[:240] if value.ndim and len(value) == frames else value for key, value in data.items()}
    motion = root / "AMASS/Study/Subject/motion_poses.npz"
    motion.parent.mkdir(parents=True)
    np.savez(motion, **data)
    shape = os.environ.get("TERRA_TEST_SHAPE")
    if shape:
        target = root / "MyoFullBody/shape_optimized.pkl"
        target.parent.mkdir(parents=True)
        shutil.copy2(shape, target)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TERRA_DATA_ROOT", str(root))
        assert (
            cohort_main(
                [
                    "--motion",
                    str(motion),
                    "--dataset-config",
                    "amass",
                    "--cache-root",
                    str(root),
                    "--smpl-model-path",
                    str(models),
                    "--output-dir",
                    str(root / "reconstruction"),
                ]
            )
            == 0
        )
    record = json.loads((root / "reconstruction/Study__Subject__motion_poses.json").read_text())
    assert record["terrain"]["boxes"]
    terrain = root / "terrain.json"
    terrain.write_text(json.dumps(record["terrain"]))
    return motion, Path(models), root, terrain


@pytest.mark.parametrize("method", ["terra", "omniretarget", "smpl", "gmr"])
def test_retarget_method_loads_solves_and_publishes(method, motion_inputs, capsys):
    motion, models, root, terrain_path = motion_inputs
    if method == "gmr" and importlib.util.find_spec("general_motion_retargeting") is None:
        pytest.skip("install the baselines extra to exercise GMR")
    motion_id = "Study/Subject/motion_poses"
    terrain = "auto" if method in {"terra", "omniretarget"} else str(terrain_path)
    assert (
        retarget_main(
            [
                str(motion),
                "--method",
                method,
                "--terrain",
                terrain,
                "--smpl-model-path",
                str(models),
                "--output-root",
                str(root),
                "--name",
                motion_id,
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    payload = json.loads(text[text.rfind("\n{") + 1 :])
    assert Path(payload["trajectory_path"]).is_file()
    artifacts = validate_retarget_artifacts(root, motion_id, method=method)
    assert artifacts.num_frames > 0
    assert artifacts.frequency > 0
    assert artifacts.nonflat_terrain
    with np.load(artifacts.analysis_path, allow_pickle=False) as analysis:
        errors = analysis["pos_error"]
        assert errors.size and np.isfinite(errors).all()

    selection = root / "motions.txt"
    selection.write_text(motion_id + "\n")
    run_root = root / "collection" / method
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("TERRA_DATA_ROOT", str(root))
        patch.setenv("TERRA_MODEL_ROOT", str(models))
        assert (
            run_main(
                [
                    "amass",
                    "--selection-manifest",
                    str(selection),
                    "--cache-root",
                    str(root),
                    "--run-root",
                    str(run_root),
                    "--method",
                    method,
                ]
            )
            == 0
        )
        report = json.loads((run_root / "run.json").read_text())
        assert report["reference_cache_root"] == str(root)
        if method == "terra":
            evaluation = root / "evaluation"
            assert (
                evaluate_main(
                    [
                        "amass",
                        "--manifest",
                        str(run_root / "manifest.csv"),
                        "--cache-root",
                        str(root),
                        "--output-root",
                        str(evaluation),
                    ]
                )
                == 0
            )
            result = json.loads((evaluation / "evaluation.json").read_text())
            assert result["stages"]["metrics"]["exit_code"] == 0
