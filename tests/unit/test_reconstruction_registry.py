"""Tests for the package-owned reconstruction registry and cohort contract."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from terra.benchmarking.reconstruction import (
    RECONSTRUCTION_METHODS,
    available_methods,
    cli,
    create_method,
    run_cohort,
)
from terra.benchmarking.reconstruction.core import (
    PreparedMotion,
    ReconstructionResult,
    load_selection,
)
from terra.benchmarking.reconstruction.matrix import (
    DEFAULT_MATRIX,
    build_jobs,
    load_matrix,
    preflight,
)


def _selection(path: Path, *motions: str) -> Path:
    path.write_text("dataset,motion\n" + "".join(f"test,{motion}\n" for motion in motions))
    return path


@dataclass
class FakeMethod:
    fail_motion: str | None = None
    value: int = 1
    validation_passed: bool = True

    name = "fake"
    display_name = "Fake reconstruction"
    description = "Deterministic test plugin"

    @property
    def options(self) -> dict[str, Any]:
        return {"value": self.value}

    def prepare(self, motion: str) -> PreparedMotion:
        return PreparedMotion({"motion": motion}, {"source": motion})

    def fit(self, motion, prepared):
        if motion == self.fail_motion:
            raise RuntimeError("deliberate fit failure")
        return ReconstructionResult(
            terrain={"boxes": [], "provenance": {"test": True}},
            fit={"value": self.value},
            validation={
                "passed": self.validation_passed,
                "influenced_reconstruction": False,
            },
        )

    def summarize(self, result):
        return {"value": result.fit["value"]}


def test_registry_is_the_complete_supported_method_inventory():
    assert available_methods() == (
        "contact-least-squares",
        "voronoi",
        "terra-no-physical-cues",
        "terra",
    )
    with pytest.raises(ValueError, match="unknown reconstruction method"):
        create_method("unknown", motions=("Study/A",))


def test_registry_uses_the_paper_method_labels():
    assert RECONSTRUCTION_METHODS["contact-least-squares"].display_name == "Contact least squares"
    assert RECONSTRUCTION_METHODS["voronoi"].display_name == "Voronoi"
    assert RECONSTRUCTION_METHODS["terra"].display_name == "TERRA"
    assert RECONSTRUCTION_METHODS["terra-no-physical-cues"].display_name == "TERRA w/o physical cues"


def test_voronoi_registry_freezes_the_paper_configuration():
    method = create_method("voronoi", motions=("Study/A",))

    assert method.name == "voronoi"
    assert method.config.grid_size_m == pytest.approx(0.10)
    assert method.config.plateau_side_m == pytest.approx(1.0)
    assert method.config.height_merge_tolerance_m == pytest.approx(0.10)
    assert method.config.minimum_box_height_m == pytest.approx(0.0001)
    assert method.contact_joints == ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle")
    assert method.terrain_links == ("L_Toe", "R_Toe", "Pelvis")
    assert method.seat_height_source == "shared_posed_posterior_body_surface"
    assert method.link_offsets == {
        "L_Toe": 0.0,
        "R_Toe": 0.0,
    }


def test_cohort_cli_accepts_a_long_inline_options_object():
    payload = '{"config":{"max_merge_candidate_pairs":100000000},"padding":"' + "x" * 300 + '"}'

    assert cli._json_object(payload)["config"]["max_merge_candidate_pairs"] == 100_000_000


def test_motion_baselines_use_the_four_default_contact_joints():
    least_squares = create_method("contact-least-squares", motions=("Study/A",))
    voronoi = create_method("voronoi", motions=("Study/A",))

    expected = ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle")
    assert least_squares.options["contact_joints"] == list(expected)
    assert voronoi.contact_joints == expected


def test_least_squares_contact_joints_cannot_be_overridden():
    with pytest.raises(ValueError, match=r"unsupported option.*contact_links"):
        create_method(
            "contact-least-squares",
            motions=("Study/A",),
            options={"contact_links": ["L_Toe"]},
        )


def test_production_terrain_namespace_excludes_comparison_implementations():
    import terra.terrain as terrain

    for name in (
        "ContactBoxConfig",
        "LeastSquaresPlaneConfig",
        "VoronoiConfig",
        "TIPTerrainConfig",
        "fit_flat_ground_control",
        "fit_voronoi_terrain_from_motion",
    ):
        assert not hasattr(terrain, name)


def test_selection_requires_unique_canonical_collision_free_motion_ids(tmp_path):
    valid = load_selection(_selection(tmp_path / "valid.csv", "Study/A", "Study/B"))
    assert valid.motions == ("Study/A", "Study/B")

    for name, motions, match in (
        ("alias.csv", ("Study/A.npz",), "already be canonical"),
        ("duplicate.csv", ("Study/A", "Study/A"), "duplicate"),
        ("collision.csv", ("A/B__C", "A__B/C"), "flatten"),
        ("case.csv", ("Study/A", "study/a"), "flatten"),
    ):
        with pytest.raises(ValueError, match=match):
            load_selection(_selection(tmp_path / name, *motions))


def test_cohort_publishes_one_schema_checkpoints_failures_and_preserves_order(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A", "Study/Bad", "Study/B")
    output = tmp_path / "out"
    result = run_cohort(FakeMethod(fail_motion="Study/Bad"), selection, output, progress=False)

    assert result.exit_code == 2
    with result.status_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["motion"] for row in rows] == ["Study/A", "Study/Bad", "Study/B"]
    assert [row["status"] for row in rows] == ["ok", "failed", "ok"]
    assert "RuntimeError: deliberate fit failure" in rows[1]["error"]
    assert not (output / "Study__Bad.json").exists()

    record = json.loads((output / "Study__A.json").read_text())
    assert set(record) == {
        "method",
        "method_display_name",
        "motion",
        "provenance",
        "terrain",
        "fit",
        "validation",
    }
    assert set(rows[0]) == {
        "motion",
        "method",
        "status",
        "output",
        "elapsed_seconds",
        "error",
        "summary_json",
    }
    run = json.loads((output / "run.json").read_text())
    assert run["counts"] == {"ok": 2, "cached": 0, "failed": 1}
    assert run["provenance"]["scientific_identity_sha256"] == record["provenance"][
        "scientific_identity_sha256"
    ]
    assert len((output / "GIT_COMMIT").read_text().strip()) == 40


def test_metric_threshold_does_not_mark_a_successful_method_as_failed(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    output = tmp_path / "out"

    result = run_cohort(
        FakeMethod(validation_passed=False),
        selection,
        output,
        progress=False,
    )

    assert result.exit_code == 0
    assert result.records[0]["status"] == "ok"
    assert result.records[0]["error"] == ""
    record = json.loads((output / "Study__A.json").read_text())
    assert record["validation"]["passed"] is False
    run = json.loads((output / "run.json").read_text())
    assert run["counts"] == {"ok": 1, "cached": 0, "failed": 0}


def test_cache_reuses_complete_output_and_overwrite_is_atomic(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    output = tmp_path / "out"
    first = run_cohort(FakeMethod(value=1), selection, output, progress=False)
    original = (output / "Study__A.json").read_bytes()
    assert first.exit_code == 0

    cached = run_cohort(FakeMethod(value=1), selection, output, progress=False)
    assert cached.records[0]["status"] == "cached"
    assert (output / "Study__A.json").read_bytes() == original

    with pytest.raises(FileExistsError, match="scientific identity"):
        run_cohort(FakeMethod(value=2), selection, output, progress=False)
    assert (output / "Study__A.json").read_bytes() == original
    assert json.loads((output / "run.json").read_text())["options"] == {"value": 1}

    failed_overwrite = run_cohort(
        FakeMethod(fail_motion="Study/A", value=2),
        selection,
        output,
        overwrite=True,
        progress=False,
    )
    assert failed_overwrite.exit_code == 2
    assert (output / "Study__A.json").read_bytes() == original

    replaced = run_cohort(FakeMethod(value=2), selection, output, overwrite=True, progress=False)
    assert replaced.exit_code == 0
    assert json.loads((output / "Study__A.json").read_text())["fit"]["value"] == 2


def test_cache_rejects_a_changed_matrix_identity_before_mutating_metadata(tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    matrix = tmp_path / "matrix.toml"
    matrix.write_text("schema_version = 1\n")
    output = tmp_path / "out"
    run_cohort(
        FakeMethod(value=1),
        selection,
        output,
        matrix_path=matrix,
        progress=False,
    )
    original_record = (output / "Study__A.json").read_bytes()
    original_run = (output / "run.json").read_bytes()

    matrix.write_text("schema_version = 2\n")
    with pytest.raises(FileExistsError, match="scientific identity"):
        run_cohort(
            FakeMethod(value=1),
            selection,
            output,
            matrix_path=matrix,
            progress=False,
        )

    assert (output / "Study__A.json").read_bytes() == original_record
    assert (output / "run.json").read_bytes() == original_run


def test_cohort_cli_forwards_registered_method_inputs(monkeypatch, tmp_path):
    selection = _selection(tmp_path / "selection.csv", "Study/A")
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return 17

    monkeypatch.setattr(cli, "run_registered", run)
    assert (
        cli.cohort_main(
            [
                "--method",
                "contact-least-squares",
                "--motions",
                str(selection),
                "--output-dir",
                str(tmp_path / "out"),
                "--options",
                '{"config":{"velocity_threshold_m_s":0.2}}',
            ]
        )
        == 17
    )
    assert calls[0][0][0] == "contact-least-squares"
    assert calls[0][1]["options"]["config"]["velocity_threshold_m_s"] == pytest.approx(0.20)

    assert (
        cli.cohort_main(
            [
                "--method",
                "terra",
                "--motions",
                str(selection),
                "--dataset-config",
                str(tmp_path / "dataset.toml"),
                "--output-dir",
                str(tmp_path / "terra"),
            ]
        )
        == 17
    )
    assert calls[1][0][0] == "terra"


def test_single_matrix_schema_selects_registered_methods_and_builds_separate_jobs(tmp_path):
    matrix = load_matrix(DEFAULT_MATRIX)
    assert tuple(matrix.methods) == available_methods()
    voronoi = matrix.methods["voronoi"].options
    assert voronoi["config"] == {
        "grid_size_m": 0.10,
        "plateau_side_m": 1.0,
        "height_merge_tolerance_m": 0.10,
        "minimum_box_height_m": 0.0001,
        "max_merge_candidate_pairs": 100_000_000,
    }
    assert voronoi["link_offsets"] == {
        "L_Toe": 0.0,
        "R_Toe": 0.0,
    }
    assert voronoi["seat_height_source"] == "shared_posed_posterior_body_surface"
    assert set(matrix.datasets) == {"amass", "gait120", "darmstadt", "vielemeyer", "prism"}
    assert all(dataset.selection is None for dataset in matrix.datasets.values())
    current_manifest = _selection(tmp_path / "current.csv", "Study/A")
    jobs = build_jobs(
        matrix,
        datasets=("gait120",),
        methods=("contact-least-squares", "terra"),
        output_root=tmp_path,
        manifests={"gait120": current_manifest},
    )
    assert [(job.dataset.name, job.method.name) for job in jobs] == [
        ("gait120", "contact-least-squares"),
        ("gait120", "terra"),
    ]
    assert jobs[0].output_dir == tmp_path / "gait120" / "contact-least-squares" / "terrain"
    assert jobs[1].output_dir == tmp_path / "gait120" / "terra" / "terrain"
    assert {check["name"] for check in preflight(jobs[0])} == {"selection", "dataset_config"}
    assert {check["name"] for check in preflight(jobs[1])} == {
        "selection",
        "dataset_config",
    }


def test_matrix_default_output_uses_artifact_root(tmp_path):
    artifact_root = tmp_path / "artifacts"
    matrix = load_matrix(
        DEFAULT_MATRIX,
        environment={"TERRA_ARTIFACT_ROOT": str(artifact_root)},
    )
    assert matrix.output_root == artifact_root / "terrain-reconstruction"
    manifest = _selection(tmp_path / "current.csv", "Study/A")
    job = build_jobs(
        matrix,
        datasets=("gait120",),
        methods=("contact-least-squares",),
        manifests={"gait120": manifest},
    )[0]
    assert job.output_dir == artifact_root / "terrain-reconstruction/gait120/contact-least-squares/terrain"

    explicit = (tmp_path / "absolute-output").resolve()
    overridden = build_jobs(
        matrix,
        datasets=("gait120",),
        methods=("contact-least-squares",),
        manifests={"gait120": manifest},
        output_root=explicit,
    )[0]
    assert overridden.output_dir.is_relative_to(explicit)
    assert overridden.publication_root == explicit
    assert overridden.publication_root_source == "absolute_cli_override"


def test_matrix_rejects_shell_commands_and_unregistered_methods(tmp_path):
    path = tmp_path / "matrix.toml"
    path.write_text(
        """
schema_version = 2
[run]
output_root = "out"
[methods.mystery]
fit_command = ["python", "unknown.py"]
[datasets.test]
selection = "selection.csv"
methods = ["mystery"]
"""
    )
    with pytest.raises(ValueError, match="not registered"):
        load_matrix(path, repo_root=tmp_path)
