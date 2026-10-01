from pathlib import Path

import numpy as np
import pytest

from terra.datasets.biomechanics import (
    biomechanics_path,
    empty_grf,
    motion_clock,
    resample_linear,
    validate_biomechanics,
    write_biomechanics,
)
from terra.datasets.darmstadt import _preflight_biomechanics, _signal_path


def _motion(path: Path, *, frames: int = 4, fps: float = 50.0) -> None:
    np.savez(
        path,
        poses=np.zeros((frames, 156), dtype=np.float32),
        mocap_framerate=np.array(fps, dtype=np.float32),
    )


def test_sidecar_is_frame_synchronized_and_preserves_native_emg(tmp_path: Path) -> None:
    motion = tmp_path / "trial_stageii.npz"
    _motion(motion)
    clock, fps = motion_clock(motion)
    native_time = np.arange(13, dtype=np.float64) / 200.0
    native = np.stack((native_time, 2.0 * native_time), axis=-1).astype(np.float32)
    emg = {
        "emg_available": np.array(True),
        "emg": resample_linear(native_time, native, clock),
        "emg_native": native,
        "emg_native_time_s": native_time,
        "emg_channels": np.asarray(("left_test", "right_test")),
        "emg_muscles": np.asarray(("TestMuscle", "TestMuscle")),
        "emg_channel_types": np.asarray(("muscle", "muscle")),
        "emg_channel_sides": np.asarray(("left", "right")),
        "emg_units": np.array("mV"),
        "emg_processing": np.array("test"),
        "emg_source_paths": np.asarray(("source.mat",)),
    }
    sidecar = biomechanics_path(motion)

    write_biomechanics(
        sidecar,
        dataset="test",
        motion="Test/trial_stageii",
        motion_time_s=clock,
        motion_fps=fps,
        synchronization="shared start",
        emg=emg,
        grf=empty_grf(len(clock)),
    )

    assert validate_biomechanics(sidecar, motion_path=motion) == {
        "frames": 4,
        "fps": 50.0,
        "emg_available": True,
        "emg_channels": 2,
        "grf_available": False,
        "grf_channels": 0,
    }
    with np.load(sidecar, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["emg_native"], native)
        np.testing.assert_allclose(data["emg"][:, 0], clock)
        assert data["grf_force"].shape == (4, 0, 3)


def test_sidecar_rejects_a_different_motion_clock(tmp_path: Path) -> None:
    motion = tmp_path / "motion.npz"
    other = tmp_path / "other.npz"
    _motion(motion, frames=4)
    _motion(other, frames=5)
    clock, fps = motion_clock(motion)
    sidecar = biomechanics_path(motion)
    write_biomechanics(
        sidecar,
        dataset="test",
        motion="test",
        motion_time_s=clock,
        motion_fps=fps,
        synchronization="test",
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_biomechanics(sidecar, motion_path=other)


def test_resampling_rejects_an_unsynchronized_source_interval() -> None:
    with pytest.raises(ValueError, match="does not cover"):
        resample_linear(
            np.arange(5, dtype=float) / 100.0,
            np.arange(5, dtype=float)[:, None],
            np.arange(5, dtype=float) / 50.0,
        )


def test_darmstadt_preflight_requires_trial_level_sources(tmp_path: Path) -> None:
    marker = tmp_path / "Marker1.mat"
    touchdown = tmp_path / "Touchdowns1.mat"
    marker.touch()
    touchdown.touch()
    row = {
        "source_path": str(marker),
        "touchdown_path": str(touchdown),
        "emg_path": str(_signal_path(tmp_path, "EMG", 1)),
        "forces_path": str(_signal_path(tmp_path, "Forces", 1)),
    }

    with pytest.raises(FileNotFoundError, match="FullyProcessed 100-point group averages"):
        _preflight_biomechanics([row])
