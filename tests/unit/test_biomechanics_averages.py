"""Tests for subject/motion-type biomechanical validation traces."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from terra.datasets.biomechanics_averages import (
    _base_payload,
    _darmstadt_stride_map,
    _gait120_grf_cycles,
    _mean_std,
    _phase_resample,
    _vielemeyer_processed_traces,
    gait120_cycle_windows,
    validate_trial_average,
    write_trial_average,
)


def test_trial_average_schema_accepts_processed_emg_and_missing_grf(tmp_path: Path) -> None:
    payload = _base_payload(
        dataset="example",
        subject="S01",
        motion_type="walking",
        condition="level",
        direction="forward",
        phase_percent=np.linspace(0.0, 100.0, 101),
        phase_definition="gait cycle",
        stride_positions=("cycle",),
        stride_sides=("right",),
        source_motions=("Example/S01/walking/trial01", "Example/S01/walking/trial02"),
    )
    mean = np.linspace(0.0, 1.0, 101, dtype=np.float32)[None, :, None]
    payload.update(
        {
            "emg_available": np.array(True),
            "emg_mean": mean,
            "emg_std": np.zeros_like(mean),
            "emg_valid": np.ones_like(mean, dtype=bool),
            "emg_channels": np.asarray(("TA",)),
            "emg_muscles": np.asarray(("TibialisAnterior",)),
            "emg_channel_sides": np.asarray(("right",)),
            "emg_units": np.array("normalized"),
            "emg_processing": np.array("release processed"),
            "emg_source_kind": np.array("release_processed"),
            "emg_source_paths": np.asarray(("source.mat",)),
            "emg_n_source_traces": np.array((2,), dtype=np.int64),
        }
    )
    path = tmp_path / "trace.npz"

    write_trial_average(path, payload)

    assert validate_trial_average(path) == {
        "dataset": "example",
        "subject": "S01",
        "phase_samples": 101,
        "stride_positions": 1,
        "emg_available": True,
        "grf_available": False,
    }


def test_trial_average_validator_rejects_misaligned_emg(tmp_path: Path) -> None:
    payload = _base_payload(
        dataset="example",
        subject="S01",
        motion_type="walking",
        condition="level",
        direction="forward",
        phase_percent=np.linspace(0.0, 100.0, 101),
        phase_definition="gait cycle",
        stride_positions=("cycle",),
        stride_sides=("right",),
        source_motions=("trial01",),
    )
    payload["emg_mean"] = np.zeros((1, 100, 0), dtype=np.float32)
    path = tmp_path / "bad.npz"
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)

    with pytest.raises(ValueError, match="invalid EMG trace shape"):
        validate_trial_average(path)


def test_mean_std_ignores_only_missing_samples() -> None:
    mean, std = _mean_std(
        (
            np.array(((1.0, np.nan), (3.0, 5.0))),
            np.array(((3.0, 4.0), (5.0, 9.0))),
        )
    )

    np.testing.assert_allclose(mean, ((2.0, 4.0), (4.0, 7.0)))
    np.testing.assert_allclose(std, ((1.0, 0.0), (1.0, 2.0)))


def test_phase_resample_preserves_endpoints_and_shape() -> None:
    values = np.stack((np.arange(5), np.arange(5) ** 2), axis=-1)

    result = _phase_resample(values, 101)

    assert result.shape == (101, 2)
    np.testing.assert_array_equal(result[0], values[0])
    np.testing.assert_array_equal(result[-1], values[-1])


def test_gait120_grf_average_excludes_incomplete_plate_contacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import terra.datasets.gait120 as gait120_module

    native_time = np.linspace(0.0, 2.0, 201)
    valid = np.zeros((len(native_time), 2), dtype=bool)
    # Sidecar order is deliberately right, left. Each complete contact spans
    # roughly half of its intended step; the short portion in the neighboring
    # step is boundary spill and must not become a second zero-padded trace.
    valid[(native_time >= 0.99) & (native_time <= 1.60), 0] = True
    valid[(native_time >= 0.50) & (native_time <= 1.10), 1] = True
    force = np.zeros((len(native_time), 2, 3), dtype=np.float32)
    force[valid[:, 0], 0, 2] = 700.0
    force[valid[:, 1], 1, 2] = 600.0
    step01 = tmp_path / "Step01.trc"
    step02 = tmp_path / "Step02.trc"
    times = {step01: np.linspace(0.0, 1.0, 101), step02: np.linspace(1.0, 2.0, 101)}
    monkeypatch.setattr(gait120_module, "load_gait120_trc", lambda path: SimpleNamespace(times=times[path]))
    sidecar = {
        "source_start_time_s": np.array(0.0),
        "source_trc_paths": np.asarray((str(step01), str(step02))),
        "grf_available": np.array(True),
        "grf_native_time_s": native_time,
        "grf_valid_native": valid,
        "grf_channels": np.asarray(("right", "left")),
        "grf_force_native": force,
        "grf_moment_native": np.zeros_like(force),
        "grf_source_paths": np.asarray(("Step01.mot", "Step02.mot")),
    }
    sidecar_path = tmp_path / "biomechanics.npz"
    np.savez_compressed(sidecar_path, **sidecar)

    windows = gait120_cycle_windows(sidecar)
    forces, _moments, _sources = _gait120_grf_cycles(sidecar_path)
    mean, _std = _mean_std(forces)

    assert [window.complete_grf_sides for window in windows] == [("left",), ("right",)]
    assert np.isnan(forces[0][:, 1]).all()
    assert np.isnan(forces[1][:, 0]).all()
    assert np.nanmax(mean[:, 0, 2]) == pytest.approx(600.0)
    assert np.nanmax(mean[:, 1, 2]) == pytest.approx(700.0)


def test_gait120_grf_average_accepts_combined_chair_plate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import terra.datasets.gait120 as gait120_module

    native_time = np.linspace(0.0, 1.0, 101)
    force = np.zeros((len(native_time), 1, 3), dtype=np.float32)
    force[:, 0, 2] = 700.0
    step = tmp_path / "Step01.trc"
    monkeypatch.setattr(
        gait120_module,
        "load_gait120_trc",
        lambda path: SimpleNamespace(times=native_time) if path == step else None,
    )
    sidecar = {
        "source_start_time_s": np.array(0.0),
        "source_trc_paths": np.asarray((str(step),)),
        "grf_available": np.array(True),
        "grf_native_time_s": native_time,
        "grf_valid_native": np.ones((len(native_time), 1), dtype=bool),
        "grf_channels": np.asarray(("combined",)),
        "grf_force_native": force,
        "grf_moment_native": np.zeros_like(force),
        "grf_source_paths": np.asarray(("Step01.mot",)),
    }
    sidecar_path = tmp_path / "chair_biomechanics.npz"
    np.savez_compressed(sidecar_path, **sidecar)

    windows = gait120_cycle_windows(sidecar)
    forces, _moments, _sources = _gait120_grf_cycles(sidecar_path)

    assert [window.complete_grf_sides for window in windows] == [("combined",)]
    assert forces[0].shape == (101, 1, 3)
    np.testing.assert_allclose(forces[0][:, 0, 2], 700.0)


def test_gait120_cycle_windows_relocates_absolute_source_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import terra.datasets.gait120 as gait120_module

    data_root = tmp_path / "cluster-data"
    step = data_root / "Gait120-original" / "extracted" / "S001" / "Step01.trc"
    step.parent.mkdir(parents=True)
    step.touch()
    monkeypatch.setattr(
        gait120_module,
        "load_gait120_trc",
        lambda path: SimpleNamespace(times=np.asarray((0.0, 1.0))) if path == step else None,
    )
    sidecar = {
        "source_start_time_s": np.asarray(0.0),
        "source_trc_paths": np.asarray(
            ("/media/data/old-checkout/data/Gait120-original/extracted/S001/Step01.trc",)
        ),
        "grf_available": np.asarray(False),
    }

    windows = gait120_cycle_windows(sidecar, data_root=data_root)

    assert len(windows) == 1
    assert windows[0].source_path == step


def test_darmstadt_stride_map_preserves_published_setup_and_side_order() -> None:
    assert _darmstadt_stride_map("ascent", 2) == [
        (2, "r", 0),
        (2, "l", 0),
        (2, "r", 1),
        (2, "l", 1),
        (2, "r", 2),
        (2, "l", 2),
        (2, "r", 3),
        (5, "l", 1),
        (5, "r", 2),
        (5, "l", 2),
        (5, "r", 3),
    ]
    assert _darmstadt_stride_map("descent", 1)[0:5] == [
        (4, "l", 0),
        (4, "r", 0),
        (4, "l", 1),
        (4, "r", 1),
        (1, "l", 0),
    ]


def test_vielemeyer_loader_uses_calculated_contact_normalized_grf(tmp_path: Path) -> None:
    archive = tmp_path / "Ref01.zip"
    phase = np.linspace(0.0, 1.0, 101)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        **{
            f"GRF{axis}{contact}": phase + axis_index + contact
            for axis_index, axis in enumerate("xyz")
            for contact in (1, 2)
        },
    )
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("Ref01/ramp75_up/trial01.npz", buffer.getvalue())

    loaded = _vielemeyer_processed_traces(archive)

    traces, paths = loaded["ramp_75_up"]
    assert len(traces) == 1
    assert traces[0].shape == (101, 2, 3)
    np.testing.assert_allclose(traces[0][:, 0, 0], phase + 1)
    assert paths == [f"{archive.resolve()}!Ref01/ramp75_up/trial01.npz"]
