"""Tests for the config-driven per-dataset pipeline contract."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terra.commands.run import read_motion_selection, read_motion_selection_rows
from terra.contracts import RetargetResult
from terra.dataset_pipeline import (
    MotionRecord,
    _selected_terrain_family,
    benchmark_timing_context,
    load_dataset_config,
    load_motion_records,
    run_motion,
)
from terra.datasets.config import bundled_dataset_config
from terra.datasets.prism.conversion import export_take
from terra.evaluation.dataset import RETARGET_EVALUATION_SCHEMA
from terra.evaluation.dataset import main as evaluate_dataset
from terra.runtime import shape_cache_path
from terra.smplh import load_smplh_motion
from terra.terrain.stance import DEFAULT_CONTACT_JOINTS

REPO = Path(__file__).resolve().parents[2]


def test_benchmark_timing_context_records_worker_and_thread_allocation(monkeypatch):
    for name, value in {
        "TERRA_CPU_REQUEST": "16",
        "TERRA_RUN_WORKERS": "8",
        "TERRA_THREADS_PER_WORKER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    }.items():
        monkeypatch.setenv(name, value)

    context = benchmark_timing_context()

    assert context["cpu_request"] == 16
    assert context["worker_processes"] == 8
    assert context["threads_per_worker"] == 1
    assert context["omp_num_threads"] == 1
    assert context["mkl_num_threads"] == 1
    assert context["openblas_num_threads"] == 1
    assert context["xla_flags"] == "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    assert isinstance(context["cpu_model"], str)


def test_selected_terrain_family_uses_only_fitted_output():
    ramp = SimpleNamespace(boxes=(object(),))
    raised = SimpleNamespace(boxes=(object(),))
    empty = SimpleNamespace(boxes=())

    assert _selected_terrain_family({"model": "ramp"}, ramp) == "ramp"
    assert _selected_terrain_family({"model": "stair_flight"}, raised) == "steps"
    assert _selected_terrain_family({"model": "per_level"}, raised) == "steps"
    assert _selected_terrain_family({"model": "flat"}, empty) == "flat"


def test_motion_selection_reads_csv_and_txt_without_losing_order(tmp_path):
    csv_path = tmp_path / "cohort.csv"
    csv_path.write_text("motion,dataset\nB/trial2,b\nA/trial1,a\n")
    txt_path = tmp_path / "cohort.txt"
    txt_path.write_text("B/trial2\nA/trial1\n")

    assert read_motion_selection(csv_path) == ["B/trial2", "A/trial1"]
    assert read_motion_selection(txt_path) == ["B/trial2", "A/trial1"]
    assert read_motion_selection_rows(csv_path)[0] == {
        "motion": "B/trial2",
        "dataset": "b",
    }


def test_motion_selection_finds_motion_column_anywhere_in_csv_header(tmp_path):
    csv_path = tmp_path / "cohort.csv"
    csv_path.write_text(
        "dataset,subject,condition,motion,fit_passed\n"
        "gait120,S01,ramp_up,Gait120/S01/trial2,true\n"
        "darmstadt,S02,stairs_up,Darmstadt/S02/trial1,true\n"
    )

    assert read_motion_selection(csv_path) == [
        "Gait120/S01/trial2",
        "Darmstadt/S02/trial1",
    ]
    assert read_motion_selection_rows(csv_path)[0]["dataset"] == "gait120"


@pytest.mark.parametrize("name", ("gait120", "vielemeyer", "darmstadt", "amass", "prism"))
def test_repository_dataset_configs_share_one_schema(name):
    path = bundled_dataset_config(name)
    config = load_dataset_config(path)
    source = path.read_text()

    assert config.name == name
    assert config.method == "terra"
    assert config.env_name == "MyoFullBody"
    assert config.smpl_model_path == config.storage_roots.model_root
    assert config.run_root.is_absolute()
    assert config.cache_root.is_absolute()
    assert config.reference_cache_root == config.cache_root
    assert config.terrain_source_dir is None
    assert config.terrain_source_method is None
    assert config.method_overrides == {}
    assert config.contact_joints == DEFAULT_CONTACT_JOINTS
    assert config.posed_seat_frame == "apparatus"
    assert not hasattr(config, "benchmark")
    assert not hasattr(config, "evaluation")
    assert not hasattr(config, "workers")
    assert "[benchmark]" not in source
    assert "[evaluation]" not in source
    assert "sample" not in source
    assert "workers" not in source


@pytest.mark.parametrize("name", ("gait120", "darmstadt", "vielemeyer", "amass", "prism"))
def test_all_datasets_use_the_same_kinematic_contact_policy(name):
    path = bundled_dataset_config(name)
    config = load_dataset_config(path)

    assert config.contact_joints == DEFAULT_CONTACT_JOINTS


def test_gait120_uses_standard_four_probe_kinematic_frontend():
    config = load_dataset_config(bundled_dataset_config("gait120"))

    assert config.contact_joints == DEFAULT_CONTACT_JOINTS


def _write_config(root: Path, body: str) -> Path:
    (root / "pyproject.toml").write_text("[project]\nname='test'\nversion='0'\n")
    path = root / "configs" / "dataset.toml"
    path.parent.mkdir(parents=True)
    path.write_text(body)
    return path


def test_manifest_normalizes_output_paths_and_calibration_template(tmp_path):
    input_root = tmp_path / "converted"
    motion = input_root / "Gait120" / "S001" / "SlopeAscent" / "Trial01" / "AllSteps_stageii.npz"
    calibration = input_root / "Gait120" / "S001" / "LevelWalking" / "Trial01" / "AllSteps_stageii.npz"
    motion.parent.mkdir(parents=True)
    calibration.parent.mkdir(parents=True)
    motion.touch()
    calibration.touch()
    manifest = input_root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("motion", "subject", "movement", "output_path", "status"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "motion": "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii",
                "subject": "1",
                "movement": "SlopeAscent",
                "output_path": motion.relative_to(input_root).as_posix(),
                "status": "generated",
            }
        )
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "gait120"
[input]
root = "converted"
manifest = "converted/manifest.csv"
[terrain]
calibration = "manifest"
calibration_template = "Gait120/S{subject_padded}/LevelWalking/Trial01/AllSteps_stageii"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )

    records = load_motion_records(load_dataset_config(config_path))

    assert len(records) == 1
    assert records[0].source_path == motion
    assert records[0].calibration_path == calibration
    assert records[0].condition == "SlopeAscent"
    assert records[0].fit_passed is True


def test_dataset_config_can_isolate_outputs_from_reference_terrain_cache(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "amass"
[input]
root = "converted"
[terrain]
mode = "flat"
calibration = "none"
contact_source = "kinematic"
[retarget]
method = "gmr"
smpl_model_path = "models"
cache_root = "rate100/cache"
reference_cache_root = "canonical/cache"
run_root = "rate100/run"
[retarget.method_overrides]
target_fps = 100.0
exact_target_fps = true
""",
    )

    config = load_dataset_config(config_path)

    assert config.cache_root == (tmp_path / "rate100/cache").resolve()
    assert config.reference_cache_root == (tmp_path / "canonical/cache").resolve()
    assert config.method_overrides == {"target_fps": 100.0, "exact_target_fps": True}


def test_dataset_config_resolves_precomputed_reconstruction_terrain(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "converted"
[terrain]
mode = "precomputed"
calibration = "none"
contact_source = "kinematic"
source_dir = "reconstruction/terra/terrain"
source_method = "TERRA"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "experiment/cache"
reference_cache_root = "canonical/cache"
run_root = "experiment/run"
""",
    )

    config = load_dataset_config(config_path)

    assert config.terrain_mode == "precomputed"
    assert config.terrain_source_dir == (tmp_path / "reconstruction/terra/terrain").resolve()
    assert config.terrain_source_method == "terra"
    assert config.reference_cache_root == (tmp_path / "canonical/cache").resolve()


def test_precomputed_terrain_requires_source_directory_and_method(tmp_path):
    path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "bad"
[input]
root = "input"
[terrain]
mode = "precomputed"
calibration = "none"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )

    with pytest.raises(ValueError, match="source_dir"):
        load_dataset_config(path)


def test_amass_config_discovers_smplh_without_a_conversion_manifest(tmp_path):
    config = load_dataset_config(bundled_dataset_config("amass"))
    fixture = tmp_path / "amass"
    fixture.mkdir()
    np.savez(fixture / "WSUF03_poses.npz", pose_aa=np.zeros((2, 156)), fps=np.asarray(60.0))
    config = replace(config, input_root=fixture)

    records = load_motion_records(config)

    assert [record.motion for record in records] == ["WSUF03_poses"]
    assert records[0].source_path == fixture / "WSUF03_poses.npz"
    assert records[0].calibration_path is None
    assert config.terrain_mode == "fit"


def test_explicit_runner_filter_can_promote_calibration_rows(tmp_path):
    input_root = tmp_path / "converted"
    flat = input_root / "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii.npz"
    ramp = input_root / "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii.npz"
    flat.parent.mkdir(parents=True)
    ramp.parent.mkdir(parents=True)
    flat.touch()
    ramp.touch()
    manifest = input_root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("motion", "output_path", "terrain_class", "role", "status"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "motion": "Gait120/S001/LevelWalking/Trial01/AllSteps_stageii",
                    "output_path": str(flat.relative_to(input_root)),
                    "terrain_class": "flat",
                    "role": "calibration",
                    "status": "generated",
                },
                {
                    "motion": "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii",
                    "output_path": str(ramp.relative_to(input_root)),
                    "terrain_class": "ramp_up",
                    "role": "retarget",
                    "status": "generated",
                },
            )
        )
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "gait120"
[input]
root = "converted"
manifest = "converted/manifest.csv"
[terrain]
calibration = "self"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    config = load_dataset_config(config_path)

    assert [record.terrain_class for record in load_motion_records(config)] == ["ramp_up"]
    promoted = load_motion_records(config, include_non_retarget=True)
    assert [record.terrain_class for record in promoted] == ["flat", "ramp_up"]


def test_prism_export_produces_smplh_motion_only(tmp_path):
    frames = 8
    contacts = np.ones((frames, 2), dtype=bool)
    take = {
        "info": {"data_info": {"fps": 100.0}},
        "smpl_params": {
            "poses": np.zeros((frames, 72), dtype=np.float32),
            "trans": np.zeros((frames, 3), dtype=np.float32),
            "root_offset": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "gender": "neutral",
        },
        "insole": {
            "L_Foot": {"contacts": contacts, "CoP_world": np.zeros((frames, 3))},
            "R_Foot": {"contacts": contacts, "CoP_world": np.zeros((frames, 3))},
        },
        # The exporter must not inspect or serialize reference object geometry.
        "objects": {"secret": object()},
    }
    source = tmp_path / "subj001" / "take001.pkl"
    source.parent.mkdir()
    with source.open("wb") as handle:
        pickle.dump(take, handle)

    take["smpl_params"]["poses"][:, 66:72] = 1.0
    with source.open("wb") as handle:
        pickle.dump(take, handle)
    row = export_take(source, tmp_path / "converted", overwrite=False)
    with np.load(Path(str(row["output_path"]))) as archive:
        assert archive["poses"].shape == (frames, 156)
        np.testing.assert_array_equal(archive["poses"][:, 66:], 0.0)
    motion = load_smplh_motion(Path(str(row["output_path"])))

    assert motion["pose_aa"].shape == (frames, 72)
    np.testing.assert_allclose(motion["trans"], np.array([[1.0, 2.0, 3.0]] * frames))
    assert motion["betas"].shape == (16,)
    assert motion["fps"] == 100.0
    assert "contact_path" not in row
    assert not Path(str(row["output_path"])).with_suffix(".contacts.json").exists()


def test_config_rejects_unknown_contact_adapter(tmp_path):
    path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "bad"
[input]
root = "input"
[terrain]
contact_source = "dataset_magic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )

    with pytest.raises(ValueError, match="contact_source"):
        load_dataset_config(path)


def test_config_rejects_contact_metadata_as_reconstruction_input(tmp_path):
    path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "bad"
[input]
root = "input"
[terrain]
contact_source = "events"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )

    with pytest.raises(ValueError, match="must be 'kinematic'"):
        load_dataset_config(path)


def test_config_rejects_nonstandard_contact_joints(tmp_path):
    path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "bad"
[input]
root = "input"
[terrain]
contact_source = "kinematic"
contact_joints = ["L_Toe", "R_Toe"]
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )

    with pytest.raises(ValueError, match="contact_joints"):
        load_dataset_config(path)


def test_manifestless_selection_resolves_exact_sources_without_recursive_discovery(tmp_path, monkeypatch):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "amass"
[input]
root = "input"
glob = "**/*.npz"
[terrain]
mode = "flat"
calibration = "none"
contact_source = "kinematic"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    config = load_dataset_config(config_path)
    motion = "KIT/Subject/walk_poses"
    source = config.input_root / f"{motion}.npz"
    source.parent.mkdir(parents=True)
    source.touch()

    monkeypatch.setattr(
        Path,
        "glob",
        lambda *_args, **_kwargs: pytest.fail("exact selections must not scan the raw dataset"),
    )

    records = load_motion_records(config, selected_motions=[motion])

    assert [record.motion for record in records] == [motion]
    assert records[0].source_path == source


def test_baseline_run_consumes_terra_terrain_without_overrides(tmp_path, monkeypatch):
    from terra import dataset_pipeline

    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "self"
contact_source = "kinematic"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
[retarget.overrides]
foot_orient_weight = 0.0
""",
    )
    config = replace(load_dataset_config(config_path), method="gmr")
    source = config.input_root / "Example" / "Trial01.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    record = MotionRecord(
        motion="Example/Trial01",
        dataset="example",
        source_path=source,
        terrain_class="ramp_up",
        expected_family="ramp",
    )
    metadata_path = tmp_path / "terra_terrain.json"
    metadata_path.write_text("{}")
    terrain = SimpleNamespace(boxes=(object(),), to_dict=lambda: {"boxes": [{}]})
    seen = {}

    monkeypatch.setattr(
        dataset_pipeline,
        "retarget_cache_paths",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=tmp_path / "missing.npz",
            analysis_path=tmp_path / "missing_analysis.npz",
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline,
        "validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(
            terrain_path=metadata_path,
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline.TerrainMetadata,
        "load",
        lambda _path: SimpleNamespace(terrain=terrain),
    )

    def fake_retarget(_path, **kwargs):
        seen.update(kwargs)
        return RetargetResult(
            trajectory=SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros((3, 2))),
                info=SimpleNamespace(frequency=100.0),
            ),
            analysis={"pos_error": np.zeros((3, 1))},
            terrain=terrain,
            method="gmr",
            source_path=source,
        )

    monkeypatch.setattr(dataset_pipeline, "retarget", fake_retarget)
    published_analysis = {}

    def fake_save(result, *_args, **_kwargs):
        published_analysis.update(result.analysis)
        return SimpleNamespace(
            trajectory_path=tmp_path / "gmr.npz",
            analysis_path=tmp_path / "gmr_analysis.npz",
            terrain_path=tmp_path / "gmr_terrain.json",
        )

    monkeypatch.setattr(dataset_pipeline, "save_retarget_result", fake_save)

    result = run_motion(config, record)

    assert result["status"] == "ok"
    assert result["terrain_model"] == "shared_terra_metadata"
    assert seen["method"] == "gmr"
    assert seen["terrain"] == metadata_path
    assert seen["config"] == {}
    assert published_analysis["benchmark_solved_frames"] == 3
    assert published_analysis["benchmark_retarget_elapsed_s"] > 0
    assert published_analysis["benchmark_timing_context"]["worker_processes"] >= 1


@pytest.mark.parametrize("method", ["terra", "smpl", "gmr", "omniretarget"])
def test_retargeting_consumes_precomputed_terrain_without_run_provenance(tmp_path, monkeypatch, method):
    from terra import dataset_pipeline

    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "precomputed"
calibration = "none"
contact_source = "kinematic"
source_dir = "reconstruction/terra/terrain"
source_method = "terra"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "experiment/cache"
reference_cache_root = "canonical/cache"
run_root = "experiment/results"
""",
    )
    config = replace(load_dataset_config(config_path), method=method)
    source = config.input_root / "Example" / "Trial01.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    record = MotionRecord(motion="Example/Trial01", dataset="example", source_path=source)
    assert config.terrain_source_dir is not None
    config.terrain_source_dir.mkdir(parents=True)
    terrain_record = config.terrain_source_dir / "Example__Trial01.json"
    terrain_record.write_text(
        json.dumps(
            {
                "method": "terra",
                "motion": record.motion,
                "terrain": {"boxes": []},
                "validation": {"passed": True},
            }
        )
    )
    terrain = SimpleNamespace(boxes=(object(),), to_dict=lambda: {"boxes": [{}]})
    seen = {}

    def no_reference_trajectory(*_args, **_kwargs):
        pytest.fail("Precomputed scenes must not require a reference TERRA trajectory")

    monkeypatch.setattr(dataset_pipeline, "validate_retarget_artifacts", no_reference_trajectory)

    monkeypatch.setattr(
        dataset_pipeline.TerrainMetadata,
        "from_dict",
        lambda _value: SimpleNamespace(terrain=terrain),
    )
    monkeypatch.setattr(
        dataset_pipeline,
        "retarget_cache_paths",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=tmp_path / "missing.npz",
            analysis_path=tmp_path / "missing_analysis.npz",
        ),
    )

    def fake_retarget(_path, **kwargs):
        seen.update(kwargs)
        return RetargetResult(
            trajectory=SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros((3, 2))),
                info=SimpleNamespace(frequency=100.0),
            ),
            analysis={"pos_error": np.zeros((3, 1))},
            terrain=terrain,
            method=method,
            source_path=source,
        )

    monkeypatch.setattr(dataset_pipeline, "retarget", fake_retarget)

    def fake_save(result, *_args, **_kwargs):
        return SimpleNamespace(
            trajectory_path=tmp_path / "terra.npz",
            analysis_path=tmp_path / "terra_analysis.npz",
            terrain_path=tmp_path / "terra_terrain.json",
        )

    monkeypatch.setattr(dataset_pipeline, "save_retarget_result", fake_save)

    result = run_motion(config, record)

    assert result["status"] == "ok"
    assert result["terrain_model"] == "terra"
    assert seen["terrain"] is terrain
    assert seen["method"] == method
    assert seen["fitted_shape_path"] == (
        shape_cache_path("MyoFullBody", config.reference_cache_root) if method == "terra" else None
    )
    if method != "terra":
        assert seen["config"] == config.method_overrides


def test_baseline_run_accepts_implicit_flat_scene(tmp_path, monkeypatch):
    from terra import dataset_pipeline

    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "flat-example"
[input]
root = "input"
[terrain]
mode = "flat"
calibration = "none"
contact_source = "kinematic"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    config = replace(load_dataset_config(config_path), method="gmr")
    source = config.input_root / "Example" / "Trial01.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    record = MotionRecord(
        motion="Example/Trial01",
        dataset="flat-example",
        source_path=source,
        terrain_class="flat",
        expected_family="flat",
    )
    seen = {}

    monkeypatch.setattr(
        dataset_pipeline,
        "retarget_cache_paths",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=tmp_path / "missing.npz",
            analysis_path=tmp_path / "missing_analysis.npz",
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline,
        "validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(terrain_path=None),
    )

    def fake_retarget(_path, **kwargs):
        seen.update(kwargs)
        return RetargetResult(
            trajectory=SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros((3, 2))),
                info=SimpleNamespace(frequency=100.0),
            ),
            analysis={"pos_error": np.zeros((3, 1))},
            terrain=None,
            method="gmr",
            source_path=source,
        )

    monkeypatch.setattr(dataset_pipeline, "retarget", fake_retarget)
    monkeypatch.setattr(
        dataset_pipeline,
        "save_retarget_result",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=tmp_path / "gmr.npz",
            analysis_path=tmp_path / "gmr_analysis.npz",
            terrain_path=None,
        ),
    )

    result = run_motion(config, record)

    assert result["status"] == "ok"
    assert result["terrain_model"] == "flat"
    assert result["terrain_family_selected"] == "flat"
    assert seen["terrain"] is None
    assert seen["config"] == {}


def test_omniretarget_dataset_run_discards_terra_overrides(tmp_path, monkeypatch):
    from terra import dataset_pipeline

    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "self"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
[retarget.overrides]
foot_orient_weight = 999.0
""",
    )
    config = replace(load_dataset_config(config_path), method="omniretarget")
    source = config.input_root / "Example" / "Trial01.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    record = MotionRecord(
        motion="Example/Trial01",
        dataset="example",
        source_path=source,
        terrain_class="ramp_up",
        expected_family="ramp",
    )
    metadata_path = tmp_path / "terra_terrain.json"
    metadata_path.write_text("{}")
    terrain = SimpleNamespace(boxes=(object(),), to_dict=lambda: {"boxes": [{}]})
    seen = {}

    monkeypatch.setattr(
        dataset_pipeline,
        "retarget_cache_paths",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=tmp_path / "missing.npz",
            analysis_path=tmp_path / "missing_analysis.npz",
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline,
        "validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(
            terrain_path=metadata_path,
        ),
    )
    monkeypatch.setattr(
        dataset_pipeline.TerrainMetadata,
        "load",
        lambda _path: SimpleNamespace(terrain=terrain),
    )

    def fake_retarget(_path, **kwargs):
        seen.update(kwargs)
        return RetargetResult(
            trajectory=SimpleNamespace(
                data=SimpleNamespace(qpos=np.zeros((3, 2))),
                info=SimpleNamespace(frequency=100.0),
            ),
            analysis={"pos_error": np.zeros((3, 1))},
            terrain=terrain,
            method="omniretarget",
            source_path=source,
        )

    monkeypatch.setattr(dataset_pipeline, "retarget", fake_retarget)
    monkeypatch.setattr(
        dataset_pipeline,
        "save_retarget_result",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=tmp_path / "omni.npz",
            analysis_path=tmp_path / "omni_analysis.npz",
            terrain_path=tmp_path / "omni_terrain.json",
        ),
    )

    result = run_motion(config, record)

    assert result["status"] == "ok"
    assert seen["config"] == {}


@pytest.mark.parametrize("table", ("benchmark", "evaluation"))
def test_config_rejects_benchmark_and_evaluation_policy(tmp_path, table):
    path = _write_config(
        tmp_path,
        f"""
schema_version = 1
[dataset]
name = "bad"
[input]
root = "input"
[terrain]
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
[{table}]
sample_size = 10
""",
    )

    with pytest.raises(ValueError, match="processing policy only"):
        load_dataset_config(path)


def test_evaluation_is_a_separate_non_mutating_dry_run(tmp_path, monkeypatch, capsys):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "none"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    config = load_dataset_config(config_path)
    config.run_root.mkdir()
    (config.run_root / "manifest.csv").write_text(
        "motion,dataset,terrain_class,passed\nExample/Trial01,example,ramp,1\n"
    )
    (config.run_root / "run.json").write_text(
        json.dumps(
            {
                "input_root": str(config.input_root),
                "cache_root": str(config.cache_root),
                "run_root": str(config.run_root),
                "method": "terra",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terra evaluate dataset",
            str(config_path),
            "--manifest",
            str(config.run_root / "manifest.csv"),
            "--output-root",
            str(config.run_root / "evaluation"),
            "--dry-run",
        ],
    )

    assert evaluate_dataset() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["methods"] == {"TERRA": "terra"}
    assert payload["manifest"] == str(config.run_root / "manifest.csv")
    assert payload["schema"] == RETARGET_EVALUATION_SCHEMA
    assert "reconstruction_dir" not in payload
    assert "prism_data_root" not in payload
    assert not (config.run_root / "evaluation").exists()


def test_evaluation_dry_run_accepts_declared_flat_motion_names(tmp_path, monkeypatch, capsys):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "flat"
calibration = "none"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("motion,dataset,terrain_class,passed\nKIT/go_over_beam01_poses,example,flat,1\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terra evaluate dataset",
            str(config_path),
            "--manifest",
            str(manifest),
            "--output-root",
            str(tmp_path / "evaluation"),
            "--dry-run",
        ],
    )

    assert evaluate_dataset() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest"] == str(manifest)


def test_evaluation_dry_run_accepts_isolated_experiment_cache(tmp_path, monkeypatch, capsys):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "none"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "canonical/cache"
run_root = "results"
""",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("motion,dataset,terrain_class,passed\nExample/Trial01,example,ramp,1\n")
    experiment_cache = tmp_path / "experiment/cache"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terra evaluate dataset",
            str(config_path),
            "--manifest",
            str(manifest),
            "--cache-root",
            str(experiment_cache),
            "--output-root",
            str(tmp_path / "evaluation"),
            "--dry-run",
        ],
    )

    assert evaluate_dataset() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_root"] == str(experiment_cache.resolve())


def test_evaluation_uses_explicit_config_paths_without_environment_mutation(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "none"
contact_source = "kinematic"
[retarget]
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    config = load_dataset_config(config_path)

    assert config.input_root == (tmp_path / "input").resolve()
    assert config.cache_root == (tmp_path / "cache").resolve()
    assert config.smpl_model_path == (tmp_path / "models").resolve()


def test_baseline_evaluation_uses_method_specific_run_root_and_label(tmp_path, monkeypatch, capsys):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "none"
contact_source = "kinematic"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    base = load_dataset_config(config_path)
    run_root = base.run_root / "gmr"
    run_root.mkdir(parents=True)
    (run_root / "manifest.csv").write_text("motion,dataset,terrain_class,passed\nExample/Trial01,example,ramp,1\n")
    (run_root / "run.json").write_text(
        json.dumps(
            {
                "input_root": str(base.input_root),
                "cache_root": str(base.cache_root),
                "run_root": str(run_root),
                "method": "gmr",
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terra evaluate dataset",
            str(config_path),
            "--manifest",
            str(run_root / "manifest.csv"),
            "--method",
            "GMR=gmr",
            "--output-root",
            str(run_root / "evaluation"),
            "--dry-run",
        ],
    )

    assert evaluate_dataset() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_root"] == str(run_root / "evaluation")
    assert payload["methods"] == {"GMR": "gmr"}


def test_multi_method_evaluation_uses_one_explicit_shared_manifest(tmp_path, monkeypatch, capsys):
    config_path = _write_config(
        tmp_path,
        """
schema_version = 1
[dataset]
name = "example"
[input]
root = "input"
[terrain]
mode = "fit"
calibration = "none"
contact_source = "kinematic"
[retarget]
method = "terra"
smpl_model_path = "models"
cache_root = "cache"
run_root = "results"
""",
    )
    base = load_dataset_config(config_path)
    for method in ("terra", "omniretarget", "gmr", "smpl"):
        run_root = base.run_root if method == "terra" else base.run_root / method
        run_root.mkdir(parents=True)
        motions = ["Example/Trial01", "Example/Trial02"]
        if method == "gmr":
            motions = motions[:1]
        with (run_root / "manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("motion", "dataset", "terrain_class", "passed"),
            )
            writer.writeheader()
            for motion in motions:
                writer.writerow(
                    {
                        "motion": motion,
                        "dataset": "example",
                        "terrain_class": "ramp",
                        "passed": 1,
                    }
                )
        (run_root / "run.json").write_text(
            json.dumps(
                {
                    "input_root": str(base.input_root),
                    "cache_root": str(base.cache_root),
                    "run_root": str(run_root),
                    "method": method,
                    "motions": 2,
                }
            )
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "terra evaluate dataset",
            str(config_path),
            "--manifest",
            str(base.run_root / "manifest.csv"),
            "--method",
            "TERRA=terra",
            "--method",
            "OmniRetarget=omniretarget",
            "--method",
            "GMR=gmr",
            "--method",
            "MuscleMimic SMPL-fit=smpl",
            "--output-root",
            str(base.run_root / "evaluation"),
            "--dry-run",
        ],
    )

    assert evaluate_dataset() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["methods"] == {
        "TERRA": "terra",
        "OmniRetarget": "omniretarget",
        "GMR": "gmr",
        "MuscleMimic SMPL-fit": "smpl",
    }
    assert not (base.run_root / "evaluation").exists()


def test_foot_calibration_cache_keeps_fitted_body_shapes_separate(monkeypatch, tmp_path):
    import terra.dataset_pipeline as pipeline
    import terra.terrain as terrain

    monkeypatch.setattr(pipeline, "_FOOT_CALIBRATION_CACHE", {})
    record = SimpleNamespace(calibration_path=tmp_path / "walking.npz", source_path=tmp_path / "stairs.npz")
    config = SimpleNamespace(
        smpl_model_path=tmp_path / "model",
        env_name="MyoFullBody",
        calibrate_sites=True,
        terrain_mode="fit",
        cache_root=tmp_path / "shape-a",
    )
    calls = []

    def world(config, *_args):
        calls.append(config.cache_root)
        return np.full((3, 17, 3), 1.0 if config.cache_root.name == "shape-a" else 2.0), 50.0

    monkeypatch.setattr(pipeline, "_world_joints", world)
    monkeypatch.setattr(
        terrain,
        "calibrate_neutral_foot_pitch",
        lambda joints, *_args: dict.fromkeys(("L", "R"), float(joints[0, 0, 0])),
    )
    monkeypatch.setattr(terrain, "detect_stance_events", lambda joints, *_args: joints)
    monkeypatch.setattr(
        terrain, "joint_surface_offsets", lambda joints: dict.fromkeys(DEFAULT_CONTACT_JOINTS, float(joints[0, 0, 0]))
    )
    monkeypatch.setattr(terrain, "paired_sole_offsets", dict)
    first = pipeline._record_calibration(config, record, np.zeros((3, 17, 3)), 50.0)
    config.cache_root = tmp_path / "shape-b"
    second = pipeline._record_calibration(config, record, np.zeros((3, 17, 3)), 50.0)
    assert first == ({"L": 1.0, "R": 1.0}, dict.fromkeys(DEFAULT_CONTACT_JOINTS, 1.0))
    assert second == ({"L": 2.0, "R": 2.0}, dict.fromkeys(DEFAULT_CONTACT_JOINTS, 2.0))
    assert pipeline._record_calibration(config, record, np.zeros((3, 17, 3)), 50.0) == second
    assert calls == [tmp_path / "shape-a", tmp_path / "shape-b"]
