import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terra.datasets.marker_fitting import (
    DARMSTADT_MARKERS,
    Clip,
    _interpolate,
    _knee_nonhinge_max_deg,
    _quality_gate_ratio,
    _quality_passed,
    _should_retry_fit,
    _stage1_state_path,
    _touchdown_crop,
    _validate_smplh,
    _write_manifest,
)
from terra.datasets.vielemeyer import _fit_subject_clips


def test_interpolate_fills_internal_and_boundary_marker_gaps() -> None:
    values = np.array(
        [
            [[np.nan, 1.0, 5.0]],
            [[2.0, np.nan, 6.0]],
            [[4.0, 3.0, np.nan]],
        ],
        dtype=np.float32,
    )

    result = _interpolate(values)

    np.testing.assert_allclose(result[:, 0, 0], [2.0, 2.0, 4.0])
    np.testing.assert_allclose(result[:, 0, 1], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result[:, 0, 2], [5.0, 6.0, 6.0])


def test_interpolate_rejects_wholly_unobserved_coordinate() -> None:
    with pytest.raises(ValueError, match="no finite samples"):
        _interpolate(np.full((3, 1, 3), np.nan, dtype=np.float32))


def test_touchdown_crop_converts_matlab_indices_and_adds_padding() -> None:
    annotation = SimpleNamespace(tdL=np.array([101, 301]), tdR=np.array([201, 401]))

    assert _touchdown_crop(annotation, n_frames=1000, fps=100.0, padding_s=0.5) == (50, 451)


def test_marker_quality_gate_requires_all_three_residual_bounds() -> None:
    args = SimpleNamespace(max_mean_mm=20.0, max_p95_mm=40.0, max_error_mm=60.0)

    assert _quality_passed({"mean_mm": 10.0, "p95_mm": 30.0, "max_mm": 50.0}, args)
    assert not _quality_passed({"mean_mm": 10.0, "p95_mm": 30.0, "max_mm": 61.0}, args)
    assert not _quality_passed({"mean_mm": 10.0, "p95_mm": 30.0}, args)
    assert _quality_gate_ratio({"mean_mm": 10.0, "p95_mm": 30.0, "max_mm": 61.0}, args) == pytest.approx(61.0 / 60.0)
    assert np.isinf(_quality_gate_ratio({"mean_mm": 10.0}, args))


def test_calibration_retry_is_not_limited_to_near_misses() -> None:
    args = SimpleNamespace(max_mean_mm=20.0, max_p95_mm=40.0, max_error_mm=60.0)
    far_failure = {"mean_mm": 15.0, "p95_mm": 30.0, "max_mm": 125.0}

    assert not _should_retry_fit(far_failure, args, retry_failed_fit=False)
    assert _should_retry_fit(far_failure, args, retry_failed_fit=True)


def test_smplh_quality_gate_checks_exported_nonhinge_knee_rotation(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    poses = np.zeros((3, 156), dtype=np.float32)

    def save() -> None:
        np.savez(
            path,
            poses=poses,
            trans=np.zeros((3, 3), dtype=np.float32),
            betas=np.zeros(16, dtype=np.float32),
            gender=np.array("neutral"),
            mocap_framerate=np.array(50.0, dtype=np.float32),
        )

    save()
    assert _validate_smplh(path, enforce_knee_hinge=True) == (3, 50.0)
    assert _knee_nonhinge_max_deg(path) == 0.0

    poses[:, 13] = np.radians(5.0)
    save()
    with pytest.raises(ValueError, match="knee-hinge gate failed"):
        _validate_smplh(path, enforce_knee_hinge=True)
    assert _knee_nonhinge_max_deg(path) == pytest.approx(5.0, abs=1e-5)


def test_darmstadt_stage1_state_isolated_by_exceptional_marker_layout(tmp_path: Path) -> None:
    marker_path = tmp_path / "markers.npz"
    full_labels = np.asarray([label for label, _field in DARMSTADT_MARKERS])
    clip = Clip(
        dataset="darmstadt",
        subject="D10",
        condition="riser_0.17",
        role="retarget",
        motion="Darmstadt/D10/test",
        source_path=str(tmp_path / "Marker10.mat"),
        marker_path=str(marker_path),
        output_path=str(tmp_path / "test.npz"),
        expected_family="steps",
    )
    args = SimpleNamespace(output_root=tmp_path / "outputs", stage1_state_root=None)

    np.savez(marker_path, labels=full_labels)
    assert _stage1_state_path(clip, args) == tmp_path / "outputs/.stage1/darmstadt/D10.npz"

    np.savez(marker_path, labels=full_labels[:-1])
    partial = _stage1_state_path(clip, args)
    assert partial.parent == tmp_path / "outputs/.stage1/darmstadt"
    assert partial.name.startswith("D10-markers-28-")
    assert partial.suffix == ".npz"


def test_external_manifest_is_atomic_and_relocatable(tmp_path: Path) -> None:
    clip = Clip(
        dataset="darmstadt",
        subject="D01",
        condition="riser_0.10",
        role="retarget",
        motion="Darmstadt/D01/test",
        source_path=str(tmp_path.parent / "Marker1.mat"),
        marker_path=str(tmp_path / ".markers/Darmstadt/D01/test.npz"),
        output_path=str(tmp_path / "Darmstadt/D01/test.npz"),
        expected_family="steps",
    )
    manifest = tmp_path / "manifest.csv"

    _write_manifest(manifest, [clip])

    with manifest.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["marker_path"] == ".markers/Darmstadt/D01/test.npz"
    assert row["output_path"] == "Darmstadt/D01/test.npz"
    assert not (tmp_path / ".manifest.csv.tmp").exists()


def test_vielemeyer_replaces_failed_calibration_and_refits_probe(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / ".stage1/vielemeyer/Ref09.npz"
    args = SimpleNamespace(
        output_root=tmp_path,
        stage1_state_root=tmp_path / ".stage1",
        max_mean_mm=20.0,
        max_p95_mm=40.0,
        max_error_mm=60.0,
    )

    def clip(name: str, family: str) -> Clip:
        return Clip(
            dataset="vielemeyer",
            subject="Ref09",
            condition="level_down" if family == "flat" else "ramp_10_up",
            role="retarget",
            motion=f"Vielemeyer/Ref09/{name}",
            source_path=str(tmp_path / f"{name}.c3d"),
            marker_path=str(tmp_path / f"{name}.c3d"),
            output_path=str(tmp_path / f"{name}.npz"),
            expected_family=family,
        )

    clips = [clip("level_a", "flat"), clip("level_b", "flat"), clip("ramp", "ramp")]
    calls = []

    def state_value(path: Path) -> str | None:
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return str(data["token"].item())

    def fitted(candidate, _args, *, force_refit=False, retry_failed_fit=False):
        candidate_state = _stage1_state_path(candidate, _args)
        calls.append(
            (
                candidate.motion,
                force_refit,
                retry_failed_fit,
                state_value(state_path),
            )
        )
        candidate_state.parent.mkdir(parents=True, exist_ok=True)
        is_probe = _args.stage1_state_root != tmp_path / ".stage1"
        if candidate.motion.endswith("level_a") and is_probe:
            np.savez(candidate_state, token=np.array("rejected"))
            candidate.fit_passed = False
        else:
            if candidate.motion.endswith("level_b"):
                np.savez(candidate_state, token=np.array("selected"))
            else:
                assert state_value(state_path) == "selected"
            candidate.fit_passed = True
        candidate.status = "generated"
        return candidate

    monkeypatch.setattr("terra.datasets.vielemeyer._fit_clip", fitted)

    results = list(_fit_subject_clips(clips, args))

    assert [result.fit_passed for result in results] == [True, True, True]
    assert {result.calibration_motion for result in results} == {"Vielemeyer/Ref09/level_b"}
    assert calls == [
        ("Vielemeyer/Ref09/level_a", True, True, None),
        ("Vielemeyer/Ref09/level_b", True, True, None),
        ("Vielemeyer/Ref09/level_a", True, True, "selected"),
        ("Vielemeyer/Ref09/ramp", True, True, "selected"),
    ]


def test_vielemeyer_calibration_failure_stops_subject(tmp_path: Path, monkeypatch) -> None:
    args = SimpleNamespace(
        output_root=tmp_path,
        stage1_state_root=tmp_path / ".stage1",
        max_mean_mm=20.0,
        max_p95_mm=40.0,
        max_error_mm=60.0,
    )
    level = Clip(
        dataset="vielemeyer",
        subject="Ref09",
        condition="level_down",
        role="retarget",
        motion="Vielemeyer/Ref09/level",
        source_path="level.c3d",
        marker_path="level.c3d",
        output_path=str(tmp_path / "level.npz"),
        expected_family="flat",
    )
    ramp = Clip(
        dataset="vielemeyer",
        subject="Ref09",
        condition="ramp_10_up",
        role="retarget",
        motion="Vielemeyer/Ref09/ramp",
        source_path="ramp.c3d",
        marker_path="ramp.c3d",
        output_path=str(tmp_path / "ramp.npz"),
        expected_family="ramp",
    )

    def failed(candidate, _args, **_kwargs):
        candidate.fit_passed = False
        candidate.status = "generated"
        return candidate

    monkeypatch.setattr("terra.datasets.vielemeyer._fit_clip", failed)

    results = list(_fit_subject_clips([level, ramp], args))

    assert results[0].status == "generated"
    assert results[1].status == "failed"
    assert "no same-subject level motion passed" in results[1].error


def test_darmstadt_mat_inventory_exports_markers_without_physiology(tmp_path):
    from scipy.io import savemat

    from terra.datasets import darmstadt

    source_root = tmp_path / "source"
    source_root.mkdir()
    touchdowns = source_root / "touchdowns/Processed/Touchdowns"
    touchdowns.mkdir(parents=True)
    markers = np.empty(6, dtype=object)
    annotations = np.empty(6, dtype=object)
    base = np.arange(90, dtype=float).reshape(30, 3) / 100
    fields = {field: base + index * 0.001 for index, (_label, field) in enumerate(DARMSTADT_MARKERS)}
    for index in range(6):
        markers[index] = np.asarray([fields, fields], dtype=object)
        annotations[index] = np.asarray([{"tdL": np.array([11, 16]), "tdR": np.array([14, 19])}] * 2, dtype=object)
    savemat(source_root / "Marker1.mat", {"Marker": markers, "Marker_fs": 100.0})
    savemat(touchdowns / "Touchdowns1.mat", {"TD_ascent": annotations, "TD_descent": annotations, "TD_fs": 100.0})
    rows = darmstadt._inventory(source_root, tmp_path / "output")
    assert len(rows) == 24
    row = rows[0]
    counts = darmstadt._export_subject([row], SimpleNamespace(redo=True, padding_s=0.02))
    assert counts[row["motion"]] == len(DARMSTADT_MARKERS)
    with np.load(row["marker_path"], allow_pickle=False) as archive:
        assert archive["positions"].shape == (13, len(DARMSTADT_MARKERS), 3)
        np.testing.assert_allclose(archive["positions"][:, 0], base[8:21])
        assert float(archive["fps"]) == 100.0
