"""Tests for the installed, package-owned dataset conversion surface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from terra.datasets import cli, darmstadt, gait120, vielemeyer


def test_convert_help_is_lazy(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected import: {name}"),
    )

    assert cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    assert all(dataset in output for dataset in ("gait120", "darmstadt", "vielemeyer", "prism"))
    assert "MAT conversion is dataset-specific" in output


@pytest.mark.parametrize(
    ("dataset", "module_name"),
    (
        ("gait120", "terra.datasets.gait120"),
        ("darmstadt", "terra.datasets.darmstadt"),
        ("vielemeyer", "terra.datasets.vielemeyer"),
        ("prism", "terra.datasets.prism.conversion"),
    ),
)
def test_convert_dispatches_only_selected_dataset(monkeypatch, dataset, module_name) -> None:
    calls = []

    def imported(name):
        calls.append(name)
        return SimpleNamespace(main=lambda argv: 23 if list(argv) == ["--help"] else 0)

    monkeypatch.setattr(cli.importlib, "import_module", imported)

    assert cli.main([dataset, "--help"]) == 23
    assert calls == [module_name]


def test_gait120_merge_is_a_package_owned_nested_command(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        "terra.datasets.gait120_merge.main",
        lambda argv: calls.append(list(argv)) or 29,
    )

    assert gait120.main(["merge", "shard-a", "--destination", "merged"]) == 29
    assert calls == [["shard-a", "--destination", "merged"]]


def test_converter_defaults_follow_external_storage_roots(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "source-data"
    artifact_root = tmp_path / "generated"
    model_root = tmp_path / "body-models"
    monkeypatch.setenv("TERRA_DATA_ROOT", str(data_root))
    monkeypatch.setenv("TERRA_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("TERRA_MODEL_ROOT", str(model_root))

    gait = gait120._build_parser().parse_args([])
    darmstadt_args = darmstadt._parse_args([])
    vielemeyer_args = vielemeyer._parse_args([])

    assert gait.original_root == data_root / "Gait120-original" / "extracted"
    assert gait.emg_root == data_root / "Gait120-EMG"
    assert gait.output_root == artifact_root / "gait120" / "smplh"
    assert gait.smpl_model_path == model_root
    assert darmstadt_args.input_root == data_root / "Darmstadt-Stair-Ambulation"
    assert darmstadt_args.output_root == artifact_root / "darmstadt" / "smplh"
    assert darmstadt_args.smpl_model_path == model_root
    assert vielemeyer_args.input_root == data_root / "Vielemeyer-Ramp-Walking"
    assert vielemeyer_args.output_root == artifact_root / "vielemeyer" / "smplh"
    assert vielemeyer_args.smpl_model_path == model_root


def test_prism_conversion_resolves_storage_aliases(tmp_path: Path, monkeypatch) -> None:
    from terra.datasets.prism import conversion

    data_root = tmp_path / "data-root"
    artifact_root = tmp_path / "artifact-root"
    monkeypatch.setenv("TERRA_DATA_ROOT", str(data_root))
    monkeypatch.setenv("TERRA_ARTIFACT_ROOT", str(artifact_root))
    observed = {}
    take = data_root / "PRISM/subj001/take001.pkl"

    def selected(root, names=()):
        observed["input"] = (root, list(names))
        return [take]

    def exported(path, output_root, *, overwrite):
        observed["output"] = (path, output_root, overwrite)
        return {
            "motion": "PRISM/subj001/take001_poses",
            "dataset": "prism",
            "subject": "subj001",
            "condition": "take001",
            "role": "retarget",
            "fit_passed": True,
            "status": "converted",
        }

    monkeypatch.setattr(conversion, "selected_take_paths", selected)
    monkeypatch.setattr(conversion, "export_take", exported)

    assert conversion.main(["data/PRISM", "--output-root", "runs/converted", "--take", "take001"]) == 0
    assert observed["input"] == (data_root / "PRISM", ["take001"])
    assert observed["output"] == (take, artifact_root / "converted", False)
    assert (artifact_root / "converted/manifest.csv").is_file()
    assert len((artifact_root / "converted/GIT_COMMIT").read_text().strip()) == 40


def test_package_converters_do_not_import_workspace_scripts() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src/terra/datasets"
    for name in ("gait120.py", "darmstadt.py", "vielemeyer.py", "marker_fitting.py"):
        source = (package_root / name).read_text()
        assert "scripts." not in source
        assert "sys.path" not in source
