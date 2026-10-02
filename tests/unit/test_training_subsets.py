import csv
import json
from pathlib import Path
from types import SimpleNamespace

from terra._musclemimic import OPTIMIZED_SHAPE_FILE_NAME
from terra.commands.materialize import materialize_subset
from terra.commands.selection import (
    build_selection,
    publish_selection,
)
from terra.commands.selection import main as select_main
from terra.paths import StorageRoots


def test_materialize_subset_hardlinks_and_writes_git_commit(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    shape = source / "MyoFullBody" / OPTIMIZED_SHAPE_FILE_NAME
    shape.parent.mkdir(parents=True)
    shape.write_bytes(b"shape")
    motion = "Gait120/S001/SlopeAscent/Trial01/AllSteps_stageii"
    base = source / "MyoFullBody" / "terra" / motion
    base.parent.mkdir(parents=True)
    trajectory = base.with_suffix(".npz")
    analysis = base.with_name(f"{base.name}_analysis.npz")
    terrain = base.with_name(f"{base.name}_terrain.json")
    trajectory.write_bytes(b"trajectory")
    analysis.write_bytes(b"analysis")
    terrain.write_bytes(b"terrain")
    row = {
        "motion": motion,
        "dataset": "gait120",
        "source_cache_root": str(source),
        "trajectory_relpath": str(trajectory.relative_to(source)),
        "analysis_relpath": str(analysis.relative_to(source)),
        "terrain_relpath": str(terrain.relative_to(source)),
    }

    monkeypatch.setattr(
        "terra.commands.materialize.validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    payload = materialize_subset([row], destination, link_mode="hardlink")

    copied = destination / row["trajectory_relpath"]
    assert copied.is_file()
    assert copied.stat().st_ino == trajectory.stat().st_ino
    assert payload["robot_shape"] is None
    assert not (destination / "MyoFullBody" / OPTIMIZED_SHAPE_FILE_NAME).exists()
    assert payload["motions"][0]["motion"] == motion
    assert payload["motions"][0]["split"] == "train"
    assert len((destination / "GIT_COMMIT").read_text().strip()) == 40


def test_materialize_mixed_split_keeps_test_artifacts_out_of_training(monkeypatch, tmp_path):
    source = tmp_path / "source"
    rows = []
    for motion, split, with_terrain in (
        ("Gait120/S001/LevelWalking/Trial01/AllSteps_stageii", "train", False),
        ("Gait120/S002/SlopeAscent/Trial01/AllSteps_stageii", "test", True),
    ):
        base = source / "MyoFullBody/terra" / motion
        base.parent.mkdir(parents=True, exist_ok=True)
        trajectory = base.with_suffix(".npz")
        analysis = base.with_name(f"{base.name}_analysis.npz")
        terrain = base.with_name(f"{base.name}_terrain.json")
        trajectory.write_bytes(b"trajectory")
        analysis.write_bytes(b"analysis")
        if with_terrain:
            terrain.write_bytes(b"terrain")
        rows.append(
            {
                "motion": motion,
                "dataset": "gait120",
                "source_cache_root": str(source),
                "trajectory_relpath": str(trajectory.relative_to(source)),
                "analysis_relpath": str(analysis.relative_to(source)),
                "terrain_relpath": str(terrain.relative_to(source)) if with_terrain else "",
                "motion_type": "ramp_up" if with_terrain else "flat_locomotion",
                "split": split,
            }
        )

    monkeypatch.setattr(
        "terra.commands.materialize.validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    payload = materialize_subset(
        rows,
        tmp_path / "destination",
        terrain_mode="mixed",
    )

    assert payload["terrain_mode"] == "mixed"
    assert payload["split_counts"] == {"train": 1, "evaluation": 0, "test": 1}
    assert [row["split"] for row in payload["motions"]] == ["train", "test"]
    assert payload["motions"][1]["paths"] == {}
    heldout_motion = tmp_path / "destination" / rows[1]["trajectory_relpath"]
    assert not heldout_motion.exists()


def test_materialize_subset_records_and_validates_flat_terrain(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    motion = "KIT/1/walk01_poses"
    base = source / "MyoFullBody/terra" / motion
    base.parent.mkdir(parents=True)
    trajectory = base.with_suffix(".npz")
    analysis = base.with_name(f"{base.name}_analysis.npz")
    terrain = base.with_name(f"{base.name}_terrain.json")
    for path in (trajectory, analysis, terrain):
        path.write_bytes(path.name.encode())
    row = {
        "motion": motion,
        "dataset": "amass",
        "source_cache_root": str(source),
        "trajectory_relpath": str(trajectory.relative_to(source)),
        "analysis_relpath": str(analysis.relative_to(source)),
        "terrain_relpath": str(terrain.relative_to(source)),
    }

    def validate(_cache_root, validated_motion, **kwargs):
        assert validated_motion == motion
        assert kwargs["require_nonflat_terrain"] is False
        return SimpleNamespace(terrain_path=terrain, nonflat_terrain=False)

    monkeypatch.setattr("terra.commands.materialize.validate_retarget_artifacts", validate)
    payload = materialize_subset([row], destination, terrain_mode="flat")

    assert payload["terrain_mode"] == "flat"
    assert payload["motions"][0]["motion"] == motion


def test_materialize_subset_accepts_canonical_flat_artifacts_without_terrain_sidecar(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    motion = "KIT/1/walk01_poses"
    base = source / "MyoFullBody/terra" / motion
    base.parent.mkdir(parents=True)
    trajectory = base.with_suffix(".npz")
    analysis = base.with_name(f"{base.name}_analysis.npz")
    terrain = base.with_name(f"{base.name}_terrain.json")
    trajectory.write_bytes(b"trajectory")
    analysis.write_bytes(b"analysis")
    row = {
        "motion": motion,
        "dataset": "amass",
        "source_cache_root": str(source),
        "trajectory_relpath": str(trajectory.relative_to(source)),
        "analysis_relpath": str(analysis.relative_to(source)),
        "terrain_relpath": str(terrain.relative_to(source)),
    }

    def validate(cache_root, validated_motion, **kwargs):
        assert cache_root == destination
        assert validated_motion == motion
        assert kwargs["require_nonflat_terrain"] is False
        assert not (destination / row["terrain_relpath"]).exists()
        return SimpleNamespace(terrain_path=None, nonflat_terrain=False)

    monkeypatch.setattr("terra.commands.materialize.validate_retarget_artifacts", validate)
    payload = materialize_subset([row], destination, terrain_mode="flat")

    paths = payload["motions"][0]["paths"]
    assert set(paths) == {"trajectory_relpath", "analysis_relpath"}
    assert (destination / row["trajectory_relpath"]).is_file()
    assert (destination / row["analysis_relpath"]).is_file()


def test_materialize_subset_rebases_manifest_and_destination_artifact_roots(monkeypatch, tmp_path):
    artifact_root = tmp_path / "artifacts"
    source = artifact_root / "gait120/cache"
    destination = artifact_root / "training/cache"
    shape = source / "MyoFullBody" / OPTIMIZED_SHAPE_FILE_NAME
    shape.parent.mkdir(parents=True)
    shape.write_bytes(b"shape")
    motion = "Gait120/S001/ramp"
    base = source / "MyoFullBody/terra" / motion
    base.parent.mkdir(parents=True)
    trajectory = base.with_suffix(".npz")
    analysis = base.with_name(f"{base.name}_analysis.npz")
    terrain = base.with_name(f"{base.name}_terrain.json")
    trajectory.write_bytes(b"trajectory")
    analysis.write_bytes(b"analysis")
    terrain.write_bytes(b"terrain")
    row = {
        "motion": motion,
        "dataset": "gait120",
        "source_cache_root": "runs/gait120/cache",
        "trajectory_relpath": str(trajectory.relative_to(source)),
        "analysis_relpath": str(analysis.relative_to(source)),
        "terrain_relpath": str(terrain.relative_to(source)),
    }
    roots = StorageRoots.from_environment(
        Path(__file__).parents[2],
        environment={"TERRA_ARTIFACT_ROOT": str(artifact_root)},
    )
    monkeypatch.setattr(
        "terra.commands.materialize.validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    payload = materialize_subset(
        [row],
        Path("runs/training/cache"),
        link_mode="hardlink",
        storage_roots=roots,
    )

    assert payload["destination_cache"] == str(destination)
    assert payload["source_caches"] == {"gait120": str(source)}
    assert payload["storage_roots"]["artifact_root"] == str(artifact_root)
    assert (destination / row["trajectory_relpath"]).is_file()


def _current_run(tmp_path: Path, dataset: str, motions: tuple[str, ...]) -> tuple[Path, Path]:
    run_root = tmp_path / dataset / "retarget"
    cache_root = tmp_path / dataset / "cache"
    run_root.mkdir(parents=True)
    manifest = run_root / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("motion", "dataset", "passed"))
        writer.writeheader()
        writer.writerows({"motion": motion, "dataset": dataset, "passed": 1} for motion in motions)
    status = run_root / "status.csv"
    status.write_text("motion,status\n" + "".join(f"{motion},ok\n" for motion in motions))
    run = run_root / "run.json"
    run.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "method": "terra",
                "env_name": "MyoFullBody",
                "cache_root": str(cache_root),
                "run_root": str(run_root),
                "output_manifest": str(manifest),
                "status": str(status),
            }
        )
    )
    return run_root, cache_root


def test_training_selection_validates_runs_and_preserves_explicit_order(monkeypatch, tmp_path):
    first = "Gait120/S001/ramp"
    second = "PRISM/subj001/take024_poses"
    gait_run, gait_cache = _current_run(tmp_path, "gait120", (first,))
    prism_run, prism_cache = _current_run(tmp_path, "prism", (second,))

    def validated(cache_root, motion, **kwargs):
        assert kwargs == {
            "method": "terra",
            "env_name": "MyoFullBody",
            "require_nonflat_terrain": True,
        }
        base = cache_root / "MyoFullBody" / "terra" / motion
        return SimpleNamespace(
            trajectory_path=base.with_suffix(".npz"),
            analysis_path=base.with_name(f"{base.name}_analysis.npz"),
            terrain_path=base.with_name(f"{base.name}_terrain.json"),
        )

    monkeypatch.setattr("terra.commands.selection.validate_retarget_artifacts", validated)
    rows = build_selection(
        [gait_run, prism_run / "run.json"],
        motions=[second, first],
    )

    assert [row["motion"] for row in rows] == [second, first]
    assert rows[0]["source_cache_root"] == str(prism_cache)
    assert rows[1]["source_cache_root"] == str(gait_cache)

    output = tmp_path / "selection.csv"
    report_output = tmp_path / "selection_report.json"
    payload = publish_selection(output, rows, report_path=report_output)
    assert payload["motions"] == [second, first]
    assert payload["report"] == str(report_output)
    assert json.loads(report_output.read_text()) == payload
    assert len((tmp_path / "GIT_COMMIT").read_text().strip()) == 40
    assert not output.with_suffix(".json").exists()
    with output.open(newline="") as handle:
        assert [row["motion"] for row in csv.DictReader(handle)] == [second, first]


def test_training_selection_accepts_only_flat_artifacts_in_flat_mode(monkeypatch, tmp_path):
    motion = "KIT/1/walk01_poses"
    run_root, cache_root = _current_run(tmp_path, "amass", (motion,))

    def validated(root, validated_motion, **kwargs):
        assert root == cache_root
        assert validated_motion == motion
        assert kwargs["require_nonflat_terrain"] is False
        base = root / "MyoFullBody/terra" / motion
        return SimpleNamespace(
            trajectory_path=base.with_suffix(".npz"),
            analysis_path=base.with_name(f"{base.name}_analysis.npz"),
            terrain_path=None,
            nonflat_terrain=False,
        )

    monkeypatch.setattr("terra.commands.selection.validate_retarget_artifacts", validated)
    rows = build_selection([run_root], terrain_mode="flat")

    assert [row["motion"] for row in rows] == [motion]
    assert rows[0]["terrain_relpath"] == ""


def test_training_selection_cli_accepts_one_validated_retargeted_motion(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    motion = "FirstRun/motion"
    base = cache / "MyoFullBody/terra" / motion

    def validated(root, selected_motion, **kwargs):
        assert root == cache
        assert selected_motion == motion
        assert kwargs["require_nonflat_terrain"] is False
        return SimpleNamespace(
            trajectory_path=base.with_suffix(".npz"),
            analysis_path=base.with_name(f"{base.name}_analysis.npz"),
            terrain_path=None,
            nonflat_terrain=False,
        )

    monkeypatch.setattr("terra.commands.selection.validate_retarget_artifacts", validated)
    selection = tmp_path / "selection.csv"
    assert (
        select_main(
            [
                "--cache-root",
                str(cache),
                "--motion",
                motion,
                "--dataset",
                "first-run",
                "--terrain-mode",
                "mixed",
                "--out",
                str(selection),
            ]
        )
        == 0
    )
    with selection.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["source_cache_root"] == str(cache)
    assert rows[0]["trajectory_relpath"] == "MyoFullBody/terra/FirstRun/motion.npz"
    assert rows[0]["split"] == "train"


def test_training_selection_can_preserve_first_artifact_for_cross_run_duplicates(monkeypatch, tmp_path):
    motion = "KIT/1/walk01_poses"
    nonflat_run, nonflat_cache = _current_run(tmp_path / "nonflat", "amass", (motion,))
    flat_run, _flat_cache = _current_run(tmp_path / "flat", "amass", (motion,))

    def validated(cache_root, selected_motion, **_kwargs):
        base = cache_root / "MyoFullBody/terra" / selected_motion
        return SimpleNamespace(
            trajectory_path=base.with_suffix(".npz"),
            analysis_path=base.with_name(f"{base.name}_analysis.npz"),
            terrain_path=base.with_name(f"{base.name}_terrain.json"),
            nonflat_terrain=True,
        )

    monkeypatch.setattr("terra.commands.selection.validate_retarget_artifacts", validated)
    rows = build_selection(
        [nonflat_run, flat_run],
        terrain_mode="mixed",
        duplicate_policy="first",
    )

    assert len(rows) == 1
    assert rows[0]["dataset"] == "amass"
    assert rows[0]["source_cache_root"] == str(nonflat_cache)


def test_training_selection_uses_published_manifest_without_status_table(monkeypatch, tmp_path):
    motion = "Gait120/S001/ramp"
    run_root, cache = _current_run(tmp_path, "gait120", (motion,))
    (run_root / "status.csv").unlink()
    base = cache / "MyoFullBody" / "terra" / motion
    monkeypatch.setattr(
        "terra.commands.selection.validate_retarget_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(
            trajectory_path=base.with_suffix(".npz"),
            analysis_path=base.with_name(f"{base.name}_analysis.npz"),
            terrain_path=None,
            nonflat_terrain=False,
        ),
    )

    rows = build_selection([run_root], terrain_mode="flat")
    assert [row["motion"] for row in rows] == [motion]
